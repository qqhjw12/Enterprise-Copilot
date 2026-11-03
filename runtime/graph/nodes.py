"""图节点实现。

节点只做三件事：读 state、调依赖、返回 state 增量。
不在这里做鉴权（在 API 层）、不做持久化（checkpointer 负责）、不发 HTTP（SSE 层负责）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from common.errors import BudgetExceeded
from runtime.graph.prompts import (
    AGENT_SYSTEM,
    APPROVAL_PROMPT,
    FALLBACK_TEMPLATES,
    KNOWLEDGE_SYSTEM,
)
from runtime.graph.state import AgentState, get_deps
from runtime.guardrails import get_guardrails
from runtime.llm.client import LLMError
from runtime.obs import metrics
from runtime.obs.spans import gen_ai_attrs, span
from runtime.rag import retrieve
from runtime.rag.types import AccessContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 意图路由

# 显式"要我去做"的动作
_ACTION_PATTERNS = [
    r"帮我(查|提交|创建|申请|报修|请)",
    r"我要(请|提交|创建|申请|报修)",
    r"请?(提交|创建|发起|报修)",
    r"(请|休)假",
    r"工单",
    r"资产",
    r"通讯录",
    r"查一下",
    r"查查",
]
# "在问规定/事实"的标记
_QUESTION_PATTERNS = [
    r"是什么", r"怎么", r"如何", r"多少", r"几天", r"标准", r"规定",
    r"制度", r"流程", r"政策", r"哪些", r"是否", r"能不能", r"可以吗",
]

# "只是在补一个参数"的标记：日期或时长，且不带新的疑问
_SUPPLEMENT_PATTERN = re.compile(
    r"\d{1,2}\s*[月\-/.]\s*\d{1,2}"  # 11-9 / 11月9
    r"|\d{1,2}\s*[日号]"  # 9日 / 9号
    r"|明天|后天|今天"  # 相对日期
    r"|\d+\s*(?:个)?(?:工作)?天"  # 请 3 天
)


def _base_intent(text: str) -> str:
    lowered = text.lower()
    has_action = any(re.search(p, lowered) for p in _ACTION_PATTERNS)
    has_question = any(re.search(p, lowered) for p in _QUESTION_PATTERNS)
    explicit = bool(re.search(r"(帮我|请帮我|我要|麻烦)", text)) or bool(
        re.search(r"\d{4}-\d{2}-\d{2}", text)
    )

    if has_action and (not has_question or explicit):
        return "operation"
    return "knowledge"


def classify_intent(text: str, *, previous: str = "") -> str:
    """规则路由：优先看"有没有明确动作"，其次看"是不是在问规定"。

    为什么不用 LLM 分类：多一次模型调用就多 500~1500ms 延迟和一份成本，
    而企业内部的意图分布其实很集中（操作类基本都绕着 请假/工单/资产/通讯录）。
    规则覆盖不到的长尾会落到 knowledge 分支，最多多答一句"这是操作类问题"。

    `previous` 用于多轮补全：用户先问"帮我提交年假从 11-3"、被追问结束日期后
    只回一句"到 11月9日"。这句话单看既没有动作也没有疑问，按上面的规则会掉进
    knowledge 分支，于是同一次请假被拆成两半——前半句在等日期，后半句去查制度。
    所以补参数的一轮要延续上一轮的意图。
    """
    intent = _base_intent(text)
    if intent == "operation":
        return intent
    if previous and _base_intent(previous) == "operation" and _is_supplement(text):
        return "operation"
    return intent


def _is_supplement(text: str) -> bool:
    """这一句是不是"只给了个参数"，而不是一个新的问题。"""
    lowered = text.lower()
    if any(re.search(p, lowered) for p in _QUESTION_PATTERNS):
        return False
    return bool(_SUPPLEMENT_PATTERN.search(text))


# ---------------------------------------------------------------- 工具函数

def _emit(event: dict[str, Any]) -> None:
    """向 SSE 层推送自定义事件；非流式执行时静默忽略。"""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(event)
    except Exception:  # noqa: BLE001 —— 事件推送失败绝不能影响主流程
        pass


def _last_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _previous_human_text(messages: list[AnyMessage]) -> str:
    """倒数第二条用户消息。用于判断"这一轮是不是在补上一轮的参数"。"""
    seen = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            seen += 1
            if seen == 2:
                return str(message.content)
    return ""


def to_openai_messages(messages: list[AnyMessage], system: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if isinstance(message, HumanMessage):
            converted.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage):
            item: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["args"], ensure_ascii=False)},
                    }
                    for call in message.tool_calls
                ]
            converted.append(item)
        elif isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.name or "tool",
                    "content": str(message.content),
                }
            )
        elif isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": str(message.content)})
    return converted


# ---------------------------------------------------------------- 节点

async def route_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    text = _last_human_text(messages)
    intent = classify_intent(text, previous=_previous_human_text(messages))
    _emit({"type": "route", "intent": intent})
    metrics.inc("agent_route_total", {"intent": intent})
    return {"intent": intent}  # type: ignore[typeddict-item]


async def knowledge_retrieve_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    query = _last_human_text(state.get("messages", []))
    ctx = AccessContext(
        tenant_id=state.get("tenant_id", ""),
        user_id=state.get("user_id", ""),
        roles=list(state.get("roles") or []),
        scopes=list(state.get("scopes") or []),
    )

    started = time.perf_counter()
    try:
        result = await retrieve(query, ctx, top_k=deps.rag_top_k)
    except Exception:  # noqa: BLE001
        logger.exception("rag_retrieve_failed")
        return {"fallback_reason": "tool_unavailable", "status": "fallback"}  # type: ignore[typeddict-item]
    elapsed = (time.perf_counter() - started) * 1000
    metrics.observe("rag_retrieve_ms", elapsed)

    trace = result.trace
    _emit(
        {
            "type": "retrieval",
            "chunks": len(result.chunks),
            "filtered_out": trace.filtered_out if trace else 0,
            "timings_ms": trace.timings_ms if trace else {},
            "total_ms": round(elapsed, 2),
        }
    )
    if result.chunks:
        _emit({"type": "citations", "citations": result.citations()})

    return {
        "retrieved_block": result.as_prompt_block(),
        "citations": result.citations(),
    }  # type: ignore[typeddict-item]


async def knowledge_answer_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    query = _last_human_text(state.get("messages", []))
    block = state.get("retrieved_block", "")
    citations = state.get("citations") or []

    if not citations:
        # 没有命中就别调用模型：省一次费用，也彻底断掉"没依据还硬编"的可能
        answer = "知识库中未收录该信息，建议联系对应部门或在 IT 服务台提交咨询工单。"
        _emit({"type": "token", "delta": answer})
        return {"messages": [AIMessage(answer)], "answer": answer, "status": "ok"}  # type: ignore[typeddict-item]

    messages = [
        {"role": "system", "content": KNOWLEDGE_SYSTEM},
        {"role": "user", "content": f"【检索上下文】\n{block}\n\n【问题】\n{query}"},
    ]

    try:
        stream = deps.llm.chat_stream(messages)
        if stream is not None:
            parts: list[str] = []
            async for delta in stream:
                parts.append(delta)
                _emit({"type": "token", "delta": delta})
            answer = "".join(parts)
        else:
            response = await deps.llm.chat(messages)
            answer = response.content
            _emit({"type": "token", "delta": answer})
    except BudgetExceeded as exc:
        return {
            "fallback_reason": "budget",
            "status": "fallback",
            "answer": exc.message,
        }  # type: ignore[typeddict-item]
    except LLMError as exc:
        logger.warning("llm_failed_in_knowledge_node", extra={"code": exc.code})
        return {
            "fallback_reason": "llm_error",
            "status": "fallback",
            "answer": str(exc),
        }  # type: ignore[typeddict-item]

    return {
        "messages": [AIMessage(answer)],
        "answer": answer,
        "status": "ok",
        "usage": deps.llm.budget.snapshot(),
    }  # type: ignore[typeddict-item]


async def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    settings = deps.settings

    iterations = int(state.get("iterations") or 0) + 1
    over_cap = iterations > settings.max_iterations
    if over_cap and not state.get("tool_events"):
        # 一次都没成功调用过工具就到上限，说明模型在原地打转，直接兜底
        return {
            "iterations": iterations,
            "fallback_reason": "max_iterations",
            "status": "fallback",
        }  # type: ignore[typeddict-item]

    caller = deps.caller
    system = AGENT_SYSTEM.format(
        user_id=caller.user_id,
        roles=", ".join(caller.roles) or "employee",
        scopes=", ".join(caller.scopes) or "-",
    )
    history = list(state.get("messages") or [])[-settings.history_window :]
    messages = to_openai_messages(history, system)

    try:
        # 超过轮次上限后不再给工具，逼模型用已有结果收尾，避免把有效结果一起丢掉
        response = await deps.llm.chat(messages, tools=None if over_cap else deps.tool_schemas())
    except BudgetExceeded as exc:
        return {
            "iterations": iterations,
            "fallback_reason": "budget",
            "status": "fallback",
            "answer": exc.message,
        }  # type: ignore[typeddict-item]
    except LLMError as exc:
        return {
            "iterations": iterations,
            "fallback_reason": "llm_error",
            "status": "fallback",
            "answer": str(exc),
        }  # type: ignore[typeddict-item]

    # 超轮次后即使模型还在请求工具，也不再执行：把已有结果交给它收尾
    effective_calls = [] if over_cap else response.tool_calls
    if not effective_calls:
        answer = response.content or "工具调用轮次已达上限，我先基于已获取的信息回答；如需继续请明确下一步。"
        _emit({"type": "token", "delta": answer})
        return {
            "messages": [AIMessage(answer)],
            "answer": answer,
            "iterations": iterations,
            "pending_tool_calls": [],
            "status": "ok",
            "degraded": over_cap,
            "usage": deps.llm.budget.snapshot(),
        }  # type: ignore[typeddict-item]

    specs = {spec.name: spec for spec in deps.toolbox.tool_specs()}
    pending: list[dict[str, Any]] = []
    for call in effective_calls:
        spec = specs.get(call.name)
        pending.append(
            {
                "id": call.id,
                "name": call.name,
                "args": call.arguments,
                "write": bool(spec and spec.write),
                "label": spec.scope if spec else "",
            }
        )

    ai_message = AIMessage(
        content=response.content or "",
        tool_calls=[{"id": c.id, "name": c.name, "args": c.arguments, "type": "tool_call"} for c in effective_calls],
    )
    _emit(
        {
            "type": "tool_call",
            "calls": [
                {"tool": p["name"], "args": p["args"], "requires_approval": p["write"]} for p in pending
            ],
        }
    )
    return {
        "messages": [ai_message],
        "pending_tool_calls": pending,
        "iterations": iterations,
        "status": "running",
    }  # type: ignore[typeddict-item]


async def approval_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """HITL 中断点。

    这个节点里**只能**有 interrupt 与决策落库，不能有副作用：
    interrupt 恢复时节点会从头重放，任何写操作都会被执行两次。
    """
    pending = state.get("pending_tool_calls") or []
    payload = {
        "type": "approval_required",
        "summary": "即将执行以下操作",
        "prompt": APPROVAL_PROMPT,
        "calls": [
            {
                "id": call["id"],
                "tool": call["name"],
                "args": call["args"],
                "write": call["write"],
            }
            for call in pending
        ],
    }
    decision = interrupt(payload) or {}
    action = str(decision.get("action", "reject")).lower()

    if action == "approve":
        return {
            "approvals": [{"action": "approve", "at": time.time()}],
            "status": "running",
        }  # type: ignore[typeddict-item]

    if action == "edit":
        edited = decision.get("calls") or []
        updated = []
        for call in pending:
            patch = next((e for e in edited if e.get("id") == call["id"]), None)
            if patch and isinstance(patch.get("args"), dict):
                call = {**call, "args": {**call["args"], **patch["args"]}}
            updated.append(call)
        return {
            "pending_tool_calls": updated,
            "approvals": [{"action": "edit", "at": time.time(), "patched": len(edited)}],
            "status": "running",
        }  # type: ignore[typeddict-item]

    # 拒绝：必须为每个 tool_call 补一条 ToolMessage，
    # 否则历史里会留下"悬空的 tool_calls"，下一轮请求会被模型服务直接判 400
    rejection_messages = [
        ToolMessage(
            content=json.dumps(
                {"ok": False, "error": {"code": "rejected_by_user", "message": "用户拒绝了该操作"}},
                ensure_ascii=False,
            ),
            tool_call_id=call["id"],
            name=call["name"],
        )
        for call in pending
    ]
    _emit({"type": "approval_result", "action": "reject"})
    # 拒绝是**终止态**，不再回到 agent：
    # 否则模型看到"工具报错"很可能换个参数再请求一次，等于绕过刚被拒绝的审批。
    cancel_text = "已按你的要求取消，本次操作未在内部系统中提交任何变更。"
    _emit({"type": "token", "delta": cancel_text})
    return {
        "messages": rejection_messages,
        "pending_tool_calls": [],
        "approvals": [{"action": "reject", "at": time.time()}],
        "answer": cancel_text,
        "status": "ok",
    }  # type: ignore[typeddict-item]


async def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    pending = state.get("pending_tool_calls") or []
    messages: list[ToolMessage] = []
    events: list[dict[str, Any]] = []
    failures = 0

    for call in pending:
        # 执行前最后一道参数校验：caller 只能由运行时注入，模型无权自带；
        # 参数里也不允许夹带密钥（模型可能从上下文里"顺手"抄到一段 token）。
        verdict = get_guardrails().check_tool_args(call["name"], call["args"])
        if verdict.tripped:
            failures += 1
            metrics.inc("guardrail_trip_total", {"stage": "tool"})
            payload = {
                "ok": False,
                "data": None,
                "error": {"code": "guardrail_blocked", "message": verdict.text, "rules": verdict.rules},
            }
            messages.append(
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
            _emit({"type": "guardrail", "data": verdict.as_event()})
            _emit({"type": "tool_result", "tool": call["name"], "ok": False, "error": payload["error"]})
            continue

        started = time.perf_counter()
        with span(
            "execute_tool",
            **gen_ai_attrs(operation="execute_tool", tool=call["name"], tool_call_id=call["id"]),
        ) as tool_span:
            outcome = await deps.toolbox.call(call["name"], call["args"], deps.caller)
            tool_span.set(**{"tool.result.ok": outcome.ok, "tool.result.code": (outcome.error or {}).get("code")})
        elapsed = (time.perf_counter() - started) * 1000
        metrics.observe("tool_call_ms", elapsed, {"tool": call["name"]})
        metrics.inc("tool_call_total", {"tool": call["name"], "ok": str(outcome.ok).lower()})

        if not outcome.ok:
            failures += 1

        payload = {"ok": outcome.ok, "data": outcome.data, "error": outcome.error}
        messages.append(
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                tool_call_id=call["id"],
                name=call["name"],
            )
        )
        event = {"type": "tool_result", **outcome.as_event()}
        events.append(event)
        _emit(event)

    previous_failures = int(state.get("consecutive_failures") or 0)
    consecutive = previous_failures + 1 if failures and failures == len(pending) else 0

    result: dict[str, Any] = {
        "messages": messages,
        "tool_events": events,
        "pending_tool_calls": [],
        "consecutive_failures": consecutive,
        "status": "running",
    }
    # 连续两轮全部失败（超时/熔断/内部错误）就不再让模型继续试，
    # 否则会出现"失败 → 重试 → 换个参数再失败"的钱包黑洞
    if consecutive >= deps.settings.max_consecutive_failures:
        result["fallback_reason"] = "tool_unavailable"
        result["status"] = "fallback"
    return result  # type: ignore[return-value]


async def fallback_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    reason = state.get("fallback_reason") or "default"
    template = FALLBACK_TEMPLATES.get(reason, FALLBACK_TEMPLATES["default"])

    detail = ""
    for event in reversed(state.get("tool_events") or []):
        if not event.get("ok"):
            detail = str((event.get("error") or {}).get("message", ""))[:80]
            break

    answer = template.format(detail=detail or "超时或连接失败", limit=deps.llm.budget.limit_usd)
    metrics.inc("agent_fallback_total", {"reason": reason})
    logger.warning("agent_fallback", extra={"reason": reason, "detail": detail})
    _emit({"type": "fallback", "reason": reason})
    _emit({"type": "token", "delta": answer})
    return {
        "messages": [AIMessage(answer)],
        "answer": answer,
        "status": "fallback",
        "usage": deps.llm.budget.snapshot(),
    }  # type: ignore[typeddict-item]
