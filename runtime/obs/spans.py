"""轻量 span：字段对齐 OpenTelemetry GenAI 语义约定。

为什么不直接引 opentelemetry-sdk：
  - 运行时只需要"一条 trace 能串起 agent → llm → tool"，不需要完整的
    BatchSpanProcessor / 采样器 / 资源探测；
  - 企业接入时通常由 sidecar/collector 统一采集，应用侧输出结构化 span 即可。

这里输出的 JSON 字段可直接映射到 OTel：
  trace_id / span_id / parent_span_id / name / duration_ms / status / attributes
属性名采用 GenAI semconv（gen_ai.operation.name / gen_ai.request.model /
gen_ai.usage.input_tokens / gen_ai.tool.name / gen_ai.conversation.id …），
因此接入 Langfuse / Phoenix / Tempo 时不需要改业务代码。
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from common.logging import trace_id_ctx

logger = logging.getLogger("trace")

_span_id_ctx: Any = None


def _span_ctx():
    global _span_id_ctx
    if _span_id_ctx is None:
        import contextvars

        _span_id_ctx = contextvars.ContextVar("span_id", default=None)
    return _span_id_ctx


def new_trace_id() -> str:
    return uuid.uuid4().hex


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def parse_traceparent(header: str | None) -> str | None:
    """W3C traceparent: 00-<trace_id 32hex>-<span_id>-<flags>。"""
    if not header:
        return None
    parts = header.strip().split("-")
    if len(parts) == 4 and len(parts[1]) == 32:
        try:
            int(parts[1], 16)
        except ValueError:
            return None
        return parts[1].lower()
    return None


@dataclass(slots=True)
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    status: str = "ok"
    error: str | None = None

    def set(self, **attrs: Any) -> Span:
        self.attributes.update({k: v for k, v in attrs.items() if v is not None})
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "duration_ms": round((time.perf_counter() - self.started) * 1000, 3),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }

    def finish(self) -> None:
        logger.info("trace_span", extra={"span": self.as_dict()})


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    ctx = _span_ctx()
    trace_id = trace_id_ctx.get() or new_trace_id()
    if trace_id_ctx.get() is None:
        trace_id_ctx.set(trace_id)
    current = Span(
        name=name,
        trace_id=trace_id,
        span_id=new_span_id(),
        parent_span_id=ctx.get(),
        attributes={k: v for k, v in attributes.items() if v is not None},
    )
    token = ctx.set(current.span_id)
    try:
        yield current
    except Exception as exc:  # noqa: BLE001
        current.status = "error"
        current.error = type(exc).__name__
        raise
    finally:
        ctx.reset(token)
        current.finish()


def gen_ai_attrs(
    *,
    operation: str,
    model: str = "",
    agent: str = "",
    conversation_id: str = "",
    tool: str = "",
    tool_call_id: str = "",
) -> dict[str, Any]:
    """GenAI semconv 的公共属性，避免每个调用点各写一套键名。"""
    attrs: dict[str, Any] = {"gen_ai.operation.name": operation}
    if model:
        attrs["gen_ai.request.model"] = model
    if agent:
        attrs["gen_ai.agent.name"] = agent
    if conversation_id:
        attrs["gen_ai.conversation.id"] = conversation_id
    if tool:
        attrs["gen_ai.tool.name"] = tool
    if tool_call_id:
        attrs["gen_ai.tool.call.id"] = tool_call_id
    return attrs
