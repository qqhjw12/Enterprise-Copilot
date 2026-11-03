"""公共中间件：request_id 注入 + 访问日志 + 耗时统计。

request_id 优先复用上游（网关）传入的 X-Request-Id，保证跨服务可追踪。
"""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from common.logging import request_id_ctx, trace_id_ctx
from common.security import Principal

access_logger = logging.getLogger("access")


def _trace_id_from(request: Request) -> str:
    """优先复用上游 traceparent，其次 x-trace-id，最后新建。

    复用而不是重新生成，才能让网关 / runtime / MCP 的日志落在同一条链路上。
    """
    parent = request.headers.get("traceparent", "")
    parts = parent.split("-")
    if len(parts) == 4 and len(parts[1]) == 32:
        try:
            int(parts[1], 16)
            return parts[1].lower()
        except ValueError:
            pass
    incoming = request.headers.get("x-trace-id", "").strip()
    if incoming and len(incoming) <= 64:
        return incoming
    return uuid.uuid4().hex


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service: str) -> None:
        super().__init__(app)
        self.service = service

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        trace_id = _trace_id_from(request)
        request.state.trace_id = trace_id
        rid_token = request_id_ctx.set(rid)
        trace_token = trace_id_ctx.set(trace_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(rid_token)
            trace_id_ctx.reset(trace_token)

        elapsed_ms = (time.perf_counter() - started) * 1000
        principal: Principal | None = getattr(request.state, "principal", None)
        response.headers["x-request-id"] = rid
        response.headers["x-trace-id"] = trace_id
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.2f}"
        access_logger.info(
            "http_request",
            extra={
                "service": self.service,
                "request_id": rid,
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "tenant_id": principal.tenant_id if principal else None,
                "user_id": principal.user_id if principal else None,
            },
        )
        # 未认证接口不要挂上用户维度字段，避免日志里出现 null 噪声
        return response
