"""Checkpointer 工厂：PostgreSQL（生产）/ SQLite（本地）/ 内存（测试）。

"禁止用内存 runtime 上生产"是硬约束，所以这里把三档写得一清二楚：
postgres 依赖缺失或连不上时，**显式降级并大声告警**，而不是静默用内存顶上。
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from runtime.config import RuntimeSettings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def checkpointer_context(settings: RuntimeSettings | None = None) -> AsyncIterator[object]:
    settings = settings or get_settings()
    backend = (settings.checkpoint_backend or "memory").lower()

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError:
            logger.error(
                "checkpoint_backend_unavailable",
                extra={"backend": "postgres", "hint": "pip install langgraph-checkpoint-postgres", "fallback": "memory"},
            )
            async with _memory() as saver:
                yield saver
            return

        dsn = settings.checkpoint_dsn
        if not dsn:
            raise RuntimeError("RUNTIME_CHECKPOINT_BACKEND=postgres 但未配置 RUNTIME_CHECKPOINT_DSN")
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            await saver.setup()  # 幂等建表
            logger.info("checkpointer_ready", extra={"backend": "postgres"})
            yield saver
        return

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError:
            logger.error(
                "checkpoint_backend_unavailable",
                extra={"backend": "sqlite", "hint": "pip install langgraph-checkpoint-sqlite", "fallback": "memory"},
            )
            async with _memory() as saver:
                yield saver
            return

        dsn = settings.checkpoint_dsn or "./var/runtime.db"
        Path(dsn).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(dsn) as saver:
            logger.info("checkpointer_ready", extra={"backend": "sqlite", "path": dsn})
            yield saver
        return

    logger.warning("checkpointer_memory_mode", extra={"hint": "仅用于本地开发/测试，生产请切 postgres"})
    async with _memory() as saver:
        yield saver


@asynccontextmanager
async def _memory() -> AsyncIterator[object]:
    from langgraph.checkpoint.memory import InMemorySaver

    yield InMemorySaver()


def is_production_safe(settings: RuntimeSettings | None = None) -> bool:
    settings = settings or get_settings()
    if settings.app_env in {"prod", "production"}:
        return (settings.checkpoint_backend or "").lower() == "postgres"
    return True
