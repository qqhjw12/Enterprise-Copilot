"""健康检查（liveness）与就绪探针（readiness）。

两者语义必须区分：
  /health  —— 进程活着就 200，供容器编排判断是否重启；
  /ready   —— 依赖可用才 200，供负载均衡判断是否摘流。
Docker Compose 的 depends_on: condition: service_healthy 依赖的就是这两个端点。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from platform_api.db import get_sessionmaker

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])
VERSION = "0.1.0"


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "platform-api", "version": VERSION, "checks": {}}


@router.get("/ready")
async def ready(response: Response) -> dict:
    checks: dict[str, str] = {}
    ok = True
    try:
        async with get_sessionmaker()() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 —— 就绪探针必须吞异常并降级为 503
        checks["database"] = f"error: {type(exc).__name__}"
        ok = False
        logger.warning("readiness_db_check_failed", extra={"error": str(exc)})

    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if ok else "degraded", "service": "platform-api", "version": VERSION, "checks": checks}
