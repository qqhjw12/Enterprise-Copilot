"""RAG 门面：进程内单例 + 租户懒加载。

为什么需要"懒加载"：演示与单测都希望 `docker compose up` 之后直接能问，
而不是先手工跑一次灌库脚本。这里在首次为某个租户检索时，
自动把 data/knowledge 下属于该租户（或未标注租户）的文档索引一遍，幂等。
"""
from __future__ import annotations

import asyncio
import logging

from runtime.config import get_rag_settings
from runtime.rag.pipeline import HybridRAG
from runtime.rag.types import AccessContext, RetrievedContext

logger = logging.getLogger(__name__)

_rag: HybridRAG | None = None
_lock = asyncio.Lock()
_indexed_tenants: set[str] = set()


async def get_rag() -> HybridRAG:
    global _rag
    if _rag is not None:
        return _rag
    async with _lock:
        if _rag is None:
            rag = HybridRAG.from_settings()
            loaded = await rag.load_persisted()
            if loaded:
                logger.info("rag_loaded_from_disk", extra={"chunks": loaded})
            _rag = rag
    return _rag


async def ensure_tenant_indexed(tenant_id: str) -> int:
    """确保该租户至少有一份可检索语料；返回本次新索引的 chunk 数。"""
    if not tenant_id:
        raise ValueError("tenant_id 不能为空")

    rag = await get_rag()
    if tenant_id in _indexed_tenants:
        return 0

    existing = [c for c in await rag.store.all_chunks() if c.metadata.get("tenant_id") == tenant_id]
    added = 0
    if not existing:
        settings = get_rag_settings()
        from pathlib import Path

        if Path(settings.knowledge_dir).exists():
            stats = await rag.index_directory(settings.knowledge_dir, tenant_id=tenant_id)
            added = stats.chunks
            if stats.skipped:
                logger.warning("rag_seed_skipped", extra={"skipped": stats.skipped[:5]})
        if added:
            rag.persist()
    _indexed_tenants.add(tenant_id)
    return added


async def retrieve(query: str, ctx: AccessContext, *, top_k: int | None = None) -> RetrievedContext:
    await ensure_tenant_indexed(ctx.tenant_id)
    rag = await get_rag()
    return await rag.retrieve(query, ctx, top_k=top_k)


def reset() -> None:
    """测试用：清空进程内单例。"""
    global _rag
    _rag = None
    _indexed_tenants.clear()
