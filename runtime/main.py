"""Runtime Service —— 执行面。

启动顺序是有讲究的：
  1. checkpointer（没有持久化就不允许对外服务，否则 HITL 恢复必崩）
  2. MCP 工具箱（发现工具；失败则降级为本地直调并标记 degraded）
  3. RAG（预热单例，懒加载语料）
  4. 编译图并挂到 app.state
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from common.errors import install_error_handlers
from common.logging import setup_logging
from common.middleware import RequestContextMiddleware
from runtime.api import chat, health, tools
from runtime.config import get_settings
from runtime.graph.checkpointer import checkpointer_context, is_production_safe
from runtime.graph.registry import list_graphs
from runtime.graph.runner import AgentRunner
from runtime.mcp_layer.client import build_toolbox
from runtime.obs.tracing import configure_tracing

logger = logging.getLogger(__name__)
setup_logging("runtime-service")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_tracing(settings)
    if not is_production_safe(settings):
        logger.error("insecure_checkpoint_backend_in_prod", extra={"backend": settings.checkpoint_backend})

    toolbox = await build_toolbox()
    app.state.toolbox = toolbox

    async with checkpointer_context(settings) as saver:
        app.state.checkpointer = saver
        app.state.runner = AgentRunner(toolbox=toolbox, checkpointer=saver, settings=settings)
        logger.info(
            "runtime_started",
            extra={
                "graphs": list_graphs(),
                "checkpointer": settings.checkpoint_backend,
                "toolbox": toolbox.transport,
                "llm_offline": not settings.llm_configured,
            },
        )
        try:
            yield
        finally:
            await toolbox.close()
            logger.info("runtime_stopped")


app = FastAPI(
    title="Enterprise Copilot · Runtime Service",
    version=health.VERSION,
    description="执行面：LangGraph 编排 / MCP 工具装配 / 混合检索 / SSE 事件流。",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware, service="runtime-service")
install_error_handlers(app)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(tools.router)
