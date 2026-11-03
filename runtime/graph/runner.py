"""Agent 运行器：驱动图执行并把内部事件翻译成 SSE 事件。

这一层是"执行面的边界"：
  - 向上：只吐事件（不含内部对象），前端/网关都能消费；
  - 向下：负责依赖注入、thread 恢复、终止态归一。
图本身不知道 HTTP 的存在。
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import Command

from common.errors import BudgetExceeded
from common.logging import trace_id_ctx
from runtime.config import RuntimeSettings, get_settings
from runtime.graph.registry import get_compiled
from runtime.graph.state import GraphDeps, initial_state
from runtime.guardrails import OUTPUT_BLOCK_MESSAGE, StreamGuard, get_guardrails
from runtime.llm.catalog import ModelCatalogClient
from runtime.llm.gateway import build_gateway
from runtime.mcp_layer.client import _BaseToolbox
from runtime.mcp_layer.schemas import CallerContext
from runtime.obs import metrics
from runtime.obs.spans import gen_ai_attrs, new_trace_id, span
from runtime.obs.tracing import run_config

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunInput:
    message: str
    thread_id: str
    caller: CallerContext
    graph_key: str = "copilot_default"
    request_id: str = ""
    token: str = ""  # 原始 JWT，用于向控制面拉取模型 Catalog
    trace_id: str = ""  # 由 HTTP 层透传；直接调用 runner 时为空则自动生成


class AgentRunner:
    def __init__(
        self,
        *,
        toolbox: _BaseToolbox,
        checkpointer: Any,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.toolbox = toolbox
        self.checkpointer = checkpointer
        self.settings = settings or get_settings()
        self.catalog = ModelCatalogClient(self.settings.platform_base_url)

    def _config(self, run: RunInput, deps: GraphDeps) -> dict[str, Any]:
        config = run_config(
            tenant_id=run.caller.tenant_id,
            user_id=run.caller.user_id,
            thread_id=run.thread_id,
            request_id=run.request_id,
            project=run.graph_key,
        )
        config["configurable"]["deps"] = deps
        # 图级保险：即使条件边写错，也不会无限循环烧钱
        config["recursion_limit"] = self.settings.max_iterations * 6 + 20
        return config

    async def stream(
        self,
        run: RunInput,
        *,
        resume: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        trace_token = trace_id_ctx.set(run.trace_id or new_trace_id())
        try:
            attrs = gen_ai_attrs(
                operation="invoke_agent",
                agent=run.graph_key,
                conversation_id=run.thread_id,
            )
            with span("invoke_agent", **attrs) as agent_span:
                async for event in self._run(run, resume=resume, agent_span=agent_span):
                    yield event
        finally:
            trace_id_ctx.reset(trace_token)

    async def _run(
        self,
        run: RunInput,
        *,
        resume: dict[str, Any] | None,
        agent_span: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        trace_id = trace_id_ctx.get() or ""
        guardrails = get_guardrails()
        metrics.inc("agent_run_total", {"graph": run.graph_key})

        # 输入闸门必须阻塞式、且在拉 Catalog / 调模型之前。
        # 并行跑虽然延迟更低，但模型可能已经把 token 花了、甚至把工具执行了
        # （openai-agents-python 的 guardrails 文档对这个取舍写得很清楚）。
        input_verdict = None
        if resume is None and run.message:
            input_verdict = guardrails.check_input(run.message)
            if input_verdict.tripped:
                metrics.inc("guardrail_trip_total", {"stage": "input"})
                elapsed_ms = (time.perf_counter() - started) * 1000
                agent_span.set(**{"agent.status": "blocked", "guardrail.rules": input_verdict.rules})
                yield {
                    "event": "start",
                    "data": {
                        "thread_id": run.thread_id,
                        "graph": run.graph_key,
                        "mode": "new",
                        "offline_llm": None,
                        "model_source": "guardrail",
                        "trace_id": trace_id,
                    },
                }
                yield {"event": "guardrail", "data": input_verdict.as_event()}
                yield {
                    "event": "done",
                    "data": {
                        "thread_id": run.thread_id,
                        "status": "blocked",
                        "answer": input_verdict.text,
                        "citations": [],
                        "intent": None,
                        "degraded": False,
                        "awaiting_approval": False,
                        "next": [],
                        "iterations": 0,
                        "usage": {
                            "limit_usd": self.settings.run_budget_usd,
                            "spent_usd": 0.0,
                            "ratio": 0.0,
                            "charges": [],
                        },
                        "latency_ms": round(elapsed_ms, 2),
                        "trace_id": trace_id,
                        "guardrail": input_verdict.as_event(),
                    },
                }
                return
            # 手机号/身份证在进入模型前脱敏，但不拦截：用户可能只是在陈述事实
            run.message = input_verdict.text

        spec = await self.catalog.resolve(run.token, run.caller.tenant_id) if run.token else None
        gateway = build_gateway(self.settings, spec=spec)
        deps = GraphDeps(
            llm=gateway,
            toolbox=self.toolbox,
            caller=run.caller,
            settings=self.settings,
        )
        graph = get_compiled(run.graph_key, self.checkpointer)
        config = self._config(run, deps)

        graph_input: Any = Command(resume=resume) if resume is not None else initial_state(
            message=run.message, caller=run.caller, thread_id=run.thread_id
        )

        yield {
            "event": "start",
            "data": {
                "thread_id": run.thread_id,
                "graph": run.graph_key,
                "mode": "resume" if resume is not None else "new",
                "offline_llm": gateway.offline,
                "model_source": "platform-catalog" if spec else "env",
                "trace_id": trace_id,
            },
        }
        if input_verdict is not None and input_verdict.redactions:
            yield {"event": "guardrail", "data": input_verdict.as_event()}

        interrupted = False
        stream_guard = StreamGuard(guardrails)
        stream_blocked = False
        try:
            async for mode, payload in graph.astream(
                graph_input, config, stream_mode=["updates", "custom"]
            ):
                if mode == "custom":
                    event = dict(payload)
                    name = str(event.pop("type", "message"))
                    if name == "token":
                        # 逐 token 过闸：密钥/金丝雀/系统提示会被扣住，不发给前端
                        safe, verdict = stream_guard.feed(str(event.get("delta", "")))
                        if verdict is not None and verdict.tripped:
                            stream_blocked = True
                            metrics.inc("guardrail_trip_total", {"stage": "output"})
                            yield {"event": "guardrail", "data": verdict.as_event()}
                        elif safe:
                            yield {"event": "token", "data": {"delta": safe}}
                        continue
                    yield {"event": name, "data": event}
                    continue

                if "__interrupt__" in payload:
                    interrupted = True
                    interrupt_payload = payload["__interrupt__"][0]
                    value = getattr(interrupt_payload, "value", interrupt_payload)
                    yield {
                        "event": "interrupt",
                        "data": {
                            "thread_id": run.thread_id,
                            "status": "awaiting_approval",
                            "approval": value,
                            "budget": gateway.budget.snapshot(),
                        },
                    }
                    continue

                for node_name, delta in payload.items():
                    if isinstance(delta, dict):
                        metrics.inc("agent_node_total", {"node": node_name})
                    yield {
                        "event": "node",
                        "data": {
                            "node": node_name,
                            "status": delta.get("status") if isinstance(delta, dict) else None,
                        },
                    }
        except BudgetExceeded as exc:
            metrics.inc("agent_budget_exceeded_total")
            agent_span.status = "error"
            agent_span.set(**{"agent.status": "budget_exceeded"})
            yield {"event": "error", "data": {"code": exc.code, "message": exc.message, "status": 402}}
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent_run_failed")
            metrics.inc("agent_run_failed_total")
            agent_span.status = "error"
            agent_span.error = type(exc).__name__
            yield {
                "event": "error",
                "data": {"code": "agent_run_failed", "message": f"执行失败：{exc}", "status": 500},
            }
            return

        # 释放流式闸门扣住的尾部；整体校验不过就丢弃（前面的安全前缀已下发，不含敏感串）
        tail, stream_verdict = stream_guard.finalize()
        if stream_verdict is not None and stream_verdict.tripped:
            stream_blocked = True
            metrics.inc("guardrail_trip_total", {"stage": "output"})
            yield {"event": "guardrail", "data": stream_verdict.as_event()}
        elif tail:
            yield {"event": "token", "data": {"delta": tail}}

        snapshot = await graph.aget_state(config)
        values = snapshot.values or {}
        answer = values.get("answer") or _last_ai_text(values.get("messages") or [])
        status = "awaiting_approval" if interrupted else values.get("status", "ok")

        # 收尾闸门：对完整 answer 再校验一次（非流式、以及流式闸门没覆盖的 PII 脱敏）
        guardrail_event = None
        if answer:
            verdict = guardrails.check_output(answer, tenant_id=run.caller.tenant_id)
            guardrail_event = verdict.as_event()
            if verdict.tripped:
                answer = OUTPUT_BLOCK_MESSAGE
                status = "blocked"
                if not stream_blocked:
                    metrics.inc("guardrail_trip_total", {"stage": "output"})
                    yield {"event": "guardrail", "data": guardrail_event}
            elif verdict.redactions:
                answer = verdict.text
                yield {"event": "guardrail", "data": guardrail_event}

        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics.observe("agent_run_ms", elapsed_ms)

        usage = gateway.budget.snapshot()
        prompt_tokens = sum(int(c.get("prompt_tokens", 0)) for c in usage["charges"])
        completion_tokens = sum(int(c.get("completion_tokens", 0)) for c in usage["charges"])
        usage["total_tokens"] = prompt_tokens + completion_tokens
        agent_span.set(
            **{
                "agent.status": status,
                "agent.iterations": values.get("iterations", 0),
                "gen_ai.usage.input_tokens": prompt_tokens,
                "gen_ai.usage.output_tokens": completion_tokens,
                "gen_ai.response.finish_reasons": status,
            }
        )

        yield {
            "event": "done",
            "data": {
                "thread_id": run.thread_id,
                "status": status,
                "answer": answer,
                "citations": values.get("citations") or [],
                "intent": values.get("intent"),
                "degraded": bool(values.get("degraded")) or gateway.degraded_to_cheap,
                "awaiting_approval": interrupted or snapshot.next == ("approval",),
                "next": list(snapshot.next),
                "iterations": values.get("iterations", 0),
                "usage": usage,
                "latency_ms": round(elapsed_ms, 2),
                "trace_id": trace_id,
                "guardrail": guardrail_event,
            },
        }

    async def get_state(self, thread_id: str, graph_key: str = "copilot_default") -> dict[str, Any]:
        graph = get_compiled(graph_key, self.checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        values = snapshot.values or {}
        pending = None
        for item in snapshot.interrupts or ():
            pending = getattr(item, "value", item)
        return {
            "thread_id": thread_id,
            # 对外只暴露稳定语义：有待审批就是 awaiting_approval，别把内部的 running 漏出去
            "status": "awaiting_approval" if snapshot.next else values.get("status"),
            "answer": values.get("answer") or _last_ai_text(values.get("messages") or []),
            "citations": values.get("citations") or [],
            "intent": values.get("intent"),
            "next": list(snapshot.next),
            "pending_approval": pending,
            "awaiting_approval": bool(snapshot.next),
            "message_count": len(values.get("messages") or []),
        }


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return str(message.content)
    return ""
