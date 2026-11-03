"""JSON 结构化日志 + 请求上下文。

为什么不用 print / 默认 logging：
日志是排障的第一入口，必须能被 ELK / Loki 直接索引。
约定字段：ts / level / logger / msg / service / request_id / tenant_id / user_id / thread_id。
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
from typing import Any

# 上下文变量：由中间件在每个请求进入时设置，日志格式化器自动带上
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
tenant_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("tenant_id", default=None)
user_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)
thread_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("thread_id", default=None)
# W3C trace_id：跨 runtime → MCP → 业务系统串一条链路（日志/追踪/spans 共用）
trace_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)

_RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
        "message", "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "msg": record.getMessage(),
        }
        for key, attr in (
            ("request_id", request_id_ctx),
            ("tenant_id", tenant_id_ctx),
            ("user_id", user_id_ctx),
            ("thread_id", thread_id_ctx),
            ("trace_id", trace_id_ctx),
        ):
            value = attr.get()
            if value:
                payload[key] = value

        # logger.info("...", extra={"tool": "query_ticket_status"}) 里的自定义字段
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(service: str, level: str | None = None, stream=None) -> None:
    """stream 默认 stdout；**stdio 传输的 MCP Server 必须传 stderr**，
    否则日志会混进 JSON-RPC 通道，把协议直接打坏。"""
    root = logging.getLogger()
    if getattr(root, "_copilot_configured", False):
        return
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root.handlers = [handler]
    root.setLevel(getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO))
    # 降噪：这些库的 INFO 日志在生产里纯属噪声
    for noisy in ("httpx", "httpcore", "uvicorn.access", "sqlalchemy.engine.Engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._copilot_configured = True  # type: ignore[attr-defined]


def bind_context(**kwargs: str | None) -> None:
    """在非 HTTP 场景（后台任务、图节点）绑定日志上下文。"""
    mapping = {
        "request_id": request_id_ctx,
        "tenant_id": tenant_id_ctx,
        "user_id": user_id_ctx,
        "thread_id": thread_id_ctx,
        "trace_id": trace_id_ctx,
    }
    for key, value in kwargs.items():
        if key in mapping and value is not None:
            mapping[key].set(value)
