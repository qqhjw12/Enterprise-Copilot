"""会话接口：SSE 流式对话 + HITL 恢复 + 线程状态查询。

事件协议（前端按 `event:` 字段分支即可）：
  start        运行开始          {thread_id, mode, offline_llm}
  route        意图路由结果       {intent}
  retrieval    检索统计           {chunks, filtered_out, timings_ms}
  citations    引用来源           {citations:[{title, snippet, score, ...}]}
  token        增量文本           {delta}
  tool_call    即将调用工具       {calls:[{tool,args,requires_approval}]}
  tool_result  工具返回           {tool, ok, data|error, latency_ms}
  interrupt    等待人工审批       {thread_id, approval:{calls:[...]}}   ← 前端弹窗
  approval_result 审批结果        {action}
  fallback     进入兜底           {reason}
  guardrail    安全闸门           {stage, action, rules, redactions}   ← 输入/输出/工具
  node         节点推进           {node, status}
  done         运行结束           {answer, citations, usage, trace_id, guardrail}
  error        异常               {code, message, status}

另有 `/v1/chat/ui-stream`：同一份事件流，翻译成 Vercel AI SDK 的 UI Message Stream v1
（见 runtime/api/ui_stream.py），给 useChat / assistant-ui 直接消费。
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from common.errors import Forbidden, NotFound, ServiceUnavailable
from common.logging import request_id_ctx
from common.scopes import SCOPE_CHAT_WRITE, SCOPE_RAG_READ
from runtime.api.deps import require_scope
from runtime.api.ui_stream import STREAM_HEADER, STREAM_VERSION, to_ui_message_stream
from runtime.graph.runner import RunInput
from runtime.mcp_layer.schemas import CallerContext
from runtime.rag import AccessContext, retrieve

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(default="", max_length=64, description="留空则新建会话")
    graph_key: str = Field(default="copilot_default", max_length=64)


class ResumeRequest(BaseModel):
    action: str = Field(pattern="^(approve|reject|edit)$")
    calls: list[dict[str, Any]] = Field(default_factory=list, description="edit 时的参数补丁")
    comment: str = Field(default="", max_length=500)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


def _runner(request: Request):
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise ServiceUnavailable("运行时尚未就绪（checkpointer 或 MCP 装配中）")
    return runner


def _bearer(request: Request) -> str:
    """原始令牌：仅用于向控制面拉取模型 Catalog，不落日志。"""
    header = request.headers.get("authorization", "")
    return header[7:].strip() if header.lower().startswith("bearer ") else ""


def _sse(event: dict[str, Any]) -> str:
    payload = json.dumps(event.get("data", {}), ensure_ascii=False, default=str)
    return f"event: {event.get('event', 'message')}\ndata: {payload}\n\n"


def _ui_response(events, thread_id: str) -> StreamingResponse:
    """AI SDK UI Message Stream：给 assistant-ui / useChat 直接消费。"""

    async def gen():
        async for chunk in to_ui_message_stream(events, message_id=thread_id):
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            STREAM_HEADER: STREAM_VERSION,
        },
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    caller: CallerContext = Depends(require_scope(SCOPE_CHAT_WRITE)),
) -> StreamingResponse:
    runner = _runner(request)
    thread_id = payload.thread_id or f"thr_{caller.user_id}_{uuid.uuid4().hex[:12]}"

    run = RunInput(
        message=payload.message,
        thread_id=thread_id,
        caller=caller,
        graph_key=payload.graph_key,
        request_id=request_id_ctx.get() or "",
        token=_bearer(request),
        trace_id=getattr(request.state, "trace_id", ""),
    )

    async def gen():
        async for event in runner.stream(run):
            yield _sse(event)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/ui-stream")
async def chat_ui_stream(
    payload: ChatRequest,
    request: Request,
    caller: CallerContext = Depends(require_scope(SCOPE_CHAT_WRITE)),
) -> StreamingResponse:
    """与 /chat/stream 同源，但输出 AI SDK UI Message Stream（前端零适配）。"""
    runner = _runner(request)
    thread_id = payload.thread_id or f"thr_{caller.user_id}_{uuid.uuid4().hex[:12]}"
    run = RunInput(
        message=payload.message,
        thread_id=thread_id,
        caller=caller,
        graph_key=payload.graph_key,
        request_id=request_id_ctx.get() or "",
        token=_bearer(request),
        trace_id=getattr(request.state, "trace_id", ""),
    )
    return _ui_response(runner.stream(run), thread_id)


@router.post("/chat")
async def chat_once(
    payload: ChatRequest,
    request: Request,
    caller: CallerContext = Depends(require_scope(SCOPE_CHAT_WRITE)),
) -> dict[str, Any]:
    """非流式版本：给 curl / 单测 / 自动化用，逻辑与流式完全一致。"""
    runner = _runner(request)
    thread_id = payload.thread_id or f"thr_{caller.user_id}_{uuid.uuid4().hex[:12]}"
    run = RunInput(
        message=payload.message,
        thread_id=thread_id,
        caller=caller,
        graph_key=payload.graph_key,
        request_id=request_id_ctx.get() or "",
        token=_bearer(request),
        trace_id=getattr(request.state, "trace_id", ""),
    )
    events: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    pending_interrupt: dict[str, Any] | None = None
    # 注意：这里**不能**在 interrupt 处 break。
    # 提前跳出会把 astream 生成器扔在半路，LangGraph 还没把 next=("approval",) 落盘，
    # 紧接着的 resume 就会读到"没有待审批操作"（真机上表现为用户手快就报错）。
    async for event in runner.stream(run):
        events.append(event)
        if event["event"] == "error":
            final = event
            break
        if event["event"] == "interrupt":
            pending_interrupt = event
        elif event["event"] == "done":
            final = event
            if pending_interrupt is not None:
                # 把审批负载合并进终止事件：对外只暴露一个"这一轮结束了"的对象
                final = {
                    "event": "interrupt",
                    "data": {**event["data"], "approval": pending_interrupt["data"].get("approval")},
                }
    return {
        "thread_id": thread_id,
        "final": final,
        "events": [{"event": e["event"], "data": e["data"]} for e in events if e["event"] != "token"],
    }


@router.post("/threads/{thread_id}/resume")
async def resume(
    thread_id: str,
    payload: ResumeRequest,
    request: Request,
    caller: CallerContext = Depends(require_scope(SCOPE_CHAT_WRITE)),
) -> StreamingResponse:
    runner = _runner(request)
    state = await runner.get_state(thread_id)
    if not state.get("awaiting_approval"):
        raise NotFound("该会话当前没有待审批的操作")

    decision = {"action": payload.action, "calls": payload.calls, "comment": payload.comment}
    run = RunInput(
        message="",
        thread_id=thread_id,
        caller=caller,
        request_id=request_id_ctx.get() or "",
        token=_bearer(request),
        trace_id=getattr(request.state, "trace_id", ""),
    )

    async def gen():
        async for event in runner.stream(run, resume=decision):
            yield _sse(event)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/threads/{thread_id}/resume/ui-stream")
async def resume_ui_stream(
    thread_id: str,
    payload: ResumeRequest,
    request: Request,
    caller: CallerContext = Depends(require_scope(SCOPE_CHAT_WRITE)),
) -> StreamingResponse:
    runner = _runner(request)
    state = await runner.get_state(thread_id)
    if not state.get("awaiting_approval"):
        raise NotFound("该会话当前没有待审批的操作")

    decision = {"action": payload.action, "calls": payload.calls, "comment": payload.comment}
    run = RunInput(
        message="",
        thread_id=thread_id,
        caller=caller,
        request_id=request_id_ctx.get() or "",
        token=_bearer(request),
        trace_id=getattr(request.state, "trace_id", ""),
    )
    return _ui_response(runner.stream(run, resume=decision), thread_id)


@router.get("/threads/{thread_id}")
async def thread_state(
    thread_id: str,
    request: Request,
    caller: CallerContext = Depends(require_scope(SCOPE_CHAT_WRITE)),
) -> dict[str, Any]:
    runner = _runner(request)
    state = await runner.get_state(thread_id)
    # 会话归属校验：thread_id 里带了发起人工号，越权查询直接拒绝
    if caller.has_scope("audit:read") or caller.has_scope("project:admin"):
        return state
    if not thread_id.startswith(f"thr_{caller.user_id}"):
        raise Forbidden("无权查看他人的会话")
    return state


@router.post("/rag/retrieve")
async def rag_retrieve(
    payload: RetrieveRequest,
    caller: CallerContext = Depends(require_scope(SCOPE_RAG_READ)),
) -> dict[str, Any]:
    """检索调试接口：把 dense/sparse/fused/rerank 四段结果都返回，便于定位召回问题。"""
    ctx = AccessContext(
        tenant_id=caller.tenant_id,
        user_id=caller.user_id,
        roles=list(caller.roles),
        scopes=list(caller.scopes),
    )
    result = await retrieve(payload.query, ctx, top_k=payload.top_k)
    return {
        "query": payload.query,
        "citations": result.citations(),
        "trace": result.trace.as_dict() if result.trace else None,
    }
