"""检索期权限过滤（fail-closed）。

设计要点：
1. 权限判断发生在**排序之前**（作为 predicate 传给召回器），而不是取回结果后再筛。
   否则 top_k=20 在"A 租户只能看 3 篇"时会被别人的文档挤满，正确结果根本进不了候选。
2. 元数据缺 tenant_id 的 chunk 一律视为不可见 —— 默认拒绝，而不是默认放行。
3. 密级用"可见等级上限"表达，避免每加一个密级就要改所有判断。
"""
from __future__ import annotations

from collections.abc import Callable

from common.scopes import SCOPE_RAG_CONFIDENTIAL
from runtime.rag.types import AccessContext, Chunk

CLASSIFICATION_LEVEL = {"public": 0, "internal": 1, "confidential": 2}


def _max_level(ctx: AccessContext) -> int:
    if ctx.is_privileged or SCOPE_RAG_CONFIDENTIAL in ctx.scopes:
        return CLASSIFICATION_LEVEL["confidential"]
    return CLASSIFICATION_LEVEL["internal"]


def _as_list(value) -> list[str]:
    """front matter 里 `allowed_roles: hr` 会被解析成字符串，这里统一成列表。
    直接对字符串做 set() 会拆成单个字符（{'h','r'}），是一类很难发现的权限绕过/误拒。"""
    if not value:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def chunk_visible(chunk: Chunk, ctx: AccessContext) -> bool:
    meta = chunk.metadata

    tenant_id = meta.get("tenant_id")
    if not tenant_id or tenant_id != ctx.tenant_id:
        return False

    classification = str(meta.get("classification", "internal"))
    if CLASSIFICATION_LEVEL.get(classification, 99) > _max_level(ctx):
        return False

    allowed_scopes = _as_list(meta.get("allowed_scopes"))
    if allowed_scopes and not ctx.has_any_scope(allowed_scopes):
        return False

    allowed_roles = _as_list(meta.get("allowed_roles"))
    if allowed_roles and not ctx.is_privileged and not set(allowed_roles) & set(ctx.roles):
        return False

    return True


def make_predicate(ctx: AccessContext) -> Callable[[Chunk], bool]:
    return lambda chunk: chunk_visible(chunk, ctx)
