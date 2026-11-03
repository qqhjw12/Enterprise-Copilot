"""执行面健康检查 / 就绪探针 / 指标导出。"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import PlainTextResponse

from runtime.config import get_rag_settings, get_settings
from runtime.graph.checkpointer import is_production_safe
from runtime.obs import metrics

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])
VERSION = "0.1.0"


@router.get("/health")
async def health() -> dict[str, Any]:
    """存活探针：只要能响应就返回 200，不检查依赖。"""
    return {"status": "ok", "service": "runtime-service", "version": VERSION, "checks": {}}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, Any]:
    """就绪探针：依赖不可用时返回 503，让编排层把流量摘掉。"""
    checks: dict[str, str] = {}
    ok = True

    settings = get_settings()
    checks["checkpoint_backend"] = settings.checkpoint_backend
    if not is_production_safe(settings):
        checks["checkpoint_safety"] = "memory-in-prod"
        ok = False

    toolbox = getattr(request.app.state, "toolbox", None)
    if toolbox is None:
        checks["mcp_toolbox"] = "not_ready"
        ok = False
    else:
        checks["mcp_toolbox"] = f"{toolbox.transport}{'(degraded)' if toolbox.degraded else ''}"

    try:
        from runtime.rag.service import get_rag

        rag = await get_rag()
        checks["rag"] = f"{type(rag.store).__name__}/{type(rag.reranker).__name__}"
        checks["rag_chunker"] = rag.chunker.name
        _ = get_rag_settings()
    except Exception as exc:  # noqa: BLE001
        checks["rag"] = f"error: {type(exc).__name__}"
        ok = False
        logger.warning("readiness_rag_failed", extra={"error": str(exc)})

    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ok else "degraded",
        "service": "runtime-service",
        "version": VERSION,
        "checks": checks,
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    return metrics.render()
