"""Platform API —— 控制面。

边界（写死在这里，防止长期腐化）：
  ✅ 认证鉴权、租户/项目隔离、模型 Catalog、审计日志
  ❌ 任何 LangGraph 图逻辑、任何 Prompt、任何检索实现
执行面挂了，控制面依然要能登录、能查审计。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from common.errors import install_error_handlers
from common.logging import setup_logging
from common.middleware import RequestContextMiddleware
from platform_api.bootstrap import bootstrap
from platform_api.config import get_settings
from platform_api.db import dispose_engine, get_sessionmaker, init_db
from platform_api.routers import audit, auth, health, models_catalog, projects, users

logger = logging.getLogger(__name__)
setup_logging("platform-api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with get_sessionmaker()() as session:
        await bootstrap(session)
    logger.info("platform_api_started", extra={"env": settings.app_env})
    try:
        yield
    finally:
        await dispose_engine()
        logger.info("platform_api_stopped")


app = FastAPI(
    title="Enterprise Copilot · Platform API",
    version=health.VERSION,
    description="控制面：认证鉴权 / 租户隔离 / 模型 Catalog / 审计。不含任何 Agent 图逻辑。",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware, service="platform-api")
install_error_handlers(app)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(projects.router)
app.include_router(models_catalog.router)
app.include_router(audit.router)
