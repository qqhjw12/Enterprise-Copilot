"""把内部事件流翻译成 Vercel AI SDK 的 UI Message Stream（v1）。

为什么要做这层适配：
  - 内部事件是"领域事件"（route/retrieval/tool_call/interrupt），前端要写一堆 if；
  - AI SDK 的 useChat / assistant-ui 只认标准 part（text-delta / tool-* / data-*），
    对齐之后前端不用为每个 Agent 平台重写渲染层。

协议要点（https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol）：
  - 响应头 `x-vercel-ai-ui-message-stream: v1`
  - 每条 `data: {json}`，以 `data: [DONE]` 结束
  - 文本必须 start → delta* → end；工具必须 input-available → output-available
  - 自定义数据用 `data-<name>`，前端在 message.parts 里按 type 取
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any

STREAM_HEADER = "x-vercel-ai-ui-message-stream"
STREAM_VERSION = "v1"


def _part(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def to_ui_message_stream(
    events: AsyncIterator[dict[str, Any]],
    *,
    message_id: str,
) -> AsyncIterator[str]:
    text_id = "text_1"
    text_open = False
    step_open = False
    call_seq = 0
    # 同名工具可能并发/连续调用，用队列保证 result 与 call 一一对应
    pending_calls: dict[str, deque[str]] = defaultdict(deque)

    yield _part({"type": "start", "messageId": message_id})
    step_open = True
    yield _part({"type": "start-step"})

    async for event in events:
        name = event.get("event", "message")
        data = event.get("data") or {}

        if name == "token":
            delta = str(data.get("delta", ""))
            if not delta:
                continue
            if not text_open:
                text_open = True
                yield _part({"type": "text-start", "id": text_id})
            yield _part({"type": "text-delta", "id": text_id, "delta": delta})

        elif name == "tool_call":
            for call in data.get("calls") or []:
                call_seq += 1
                call_id = f"call_{call_seq}"
                pending_calls[str(call.get("tool", ""))].append(call_id)
                yield _part(
                    {
                        "type": "tool-input-available",
                        "toolCallId": call_id,
                        "toolName": call.get("tool", ""),
                        "input": call.get("args") or {},
                    }
                )
                if call.get("requires_approval"):
                    yield _part(
                        {
                            "type": "tool-approval-request",
                            "toolCallId": call_id,
                            "approvalId": f"ap_{call_id}",
                            "reason": "该操作会修改内部系统，需要人工确认",
                        }
                    )

        elif name == "tool_result":
            tool = str(data.get("tool", ""))
            call_id = pending_calls[tool].popleft() if pending_calls[tool] else f"call_unknown_{tool}"
            yield _part(
                {
                    "type": "tool-output-available",
                    "toolCallId": call_id,
                    "output": {
                        "ok": data.get("ok"),
                        "data": data.get("data"),
                        "error": data.get("error"),
                        "latency_ms": data.get("latency_ms"),
                    },
                }
            )

        elif name == "citations":
            yield _part({"type": "data-citations", "data": {"citations": data.get("citations") or []}})

        elif name == "guardrail":
            yield _part({"type": "data-guardrail", "data": data})

        elif name == "interrupt":
            # 审批负载单独给一份：前端可以渲染"改参数"表单，而不只是 approve/reject
            yield _part(
                {
                    "type": "data-approval",
                    "data": {
                        "thread_id": data.get("thread_id"),
                        "approval": data.get("approval"),
                        "budget": data.get("budget"),
                    },
                }
            )

        elif name == "route":
            yield _part({"type": "data-route", "data": {"intent": data.get("intent")}})

        elif name == "retrieval":
            yield _part({"type": "data-retrieval", "data": data})

        elif name == "fallback":
            yield _part({"type": "data-fallback", "data": data})

        elif name == "error":
            error_text = str(data.get("message") or "执行失败")
            yield _part({"type": "error", "errorText": error_text})
            if text_open:
                text_open = False
                yield _part({"type": "text-end", "id": text_id})
            if step_open:
                step_open = False
                yield _part({"type": "finish-step"})
            yield _part({"type": "finish"})
            yield "data: [DONE]\n\n"
            return

        elif name == "done":
            # 被安全策略拦截时没有 token 流，但用户必须看到"为什么没结果"。
            # 这里把 done.answer 补成一段正常文本，前端不用为 blocked 单独写渲染。
            answer = str(data.get("answer") or "")
            if not text_open and data.get("status") == "blocked" and answer:
                text_open = True
                yield _part({"type": "text-start", "id": text_id})
                yield _part({"type": "text-delta", "id": text_id, "delta": answer})
            if text_open:
                text_open = False
                yield _part({"type": "text-end", "id": text_id})
            if step_open:
                step_open = False
                yield _part({"type": "finish-step"})
            yield _part(
                {
                    "type": "data-done",
                    "data": {
                        "thread_id": data.get("thread_id"),
                        "status": data.get("status"),
                        "answer": answer,
                        "citations": data.get("citations") or [],
                        "usage": data.get("usage"),
                        "latency_ms": data.get("latency_ms"),
                        "trace_id": data.get("trace_id"),
                        "awaiting_approval": data.get("awaiting_approval", False),
                    },
                }
            )
            yield _part({"type": "finish"})
            yield "data: [DONE]\n\n"
            return

    # 没有 done（上游异常断流）也要给出合法的终止帧，避免前端永远转圈
    if text_open:
        yield _part({"type": "text-end", "id": text_id})
    if step_open:
        yield _part({"type": "finish-step"})
    yield _part({"type": "finish"})
    yield "data: [DONE]\n\n"
