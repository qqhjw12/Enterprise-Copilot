"""统一的错误信封。

对外只暴露可预期的错误码，内部异常一律折叠成 500 + request_id，
让用户能报障、让我们能定位，但堆栈绝不进响应体。
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """业务错误基类。status_code 决定 HTTP 语义，code 决定前端分支。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class NotFound(ApiError):
    def __init__(self, message: str = "资源不存在", **kw: Any) -> None:
        super().__init__("not_found", message, status_code=404, **kw)


class Unauthorized(ApiError):
    def __init__(self, message: str = "未认证或令牌已失效", **kw: Any) -> None:
        super().__init__("unauthorized", message, status_code=401, **kw)


class Forbidden(ApiError):
    def __init__(self, message: str = "无权访问该资源", **kw: Any) -> None:
        super().__init__("forbidden", message, status_code=403, **kw)


class Conflict(ApiError):
    def __init__(self, message: str = "资源冲突", **kw: Any) -> None:
        super().__init__("conflict", message, status_code=409, **kw)


class BudgetExceeded(ApiError):
    def __init__(self, message: str = "本次运行超出成本预算", **kw: Any) -> None:
        super().__init__("budget_exceeded", message, status_code=402, **kw)


class ServiceUnavailable(ApiError):
    """依赖未就绪（启动中 / 工具箱降级 / 下游不可用）。用 503 而不是 404，
    否则客户端会把"再等等"当成"接口不存在"。"""

    def __init__(self, message: str = "服务尚未就绪，请稍后重试", **kw: Any) -> None:
        super().__init__("service_unavailable", message, status_code=503, **kw)


def _envelope(code: str, message: str, request_id: str | None, details: dict[str, Any] | None = None) -> dict:
    body: dict[str, Any] = {"error": {"code": code, "message": message, "request_id": request_id}}
    if details:
        body["error"]["details"] = details
    return body


def install_error_handlers(app: FastAPI) -> None:
    from common.logging import request_id_ctx

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, request_id_ctx.get(), exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error",
                "请求参数校验失败",
                request_id_ctx.get(),
                {"errors": [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]},
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "服务内部错误，请携带 request_id 反馈", request_id_ctx.get()),
        )
