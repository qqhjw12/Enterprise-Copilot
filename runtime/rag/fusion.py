"""RRF（Reciprocal Rank Fusion）融合。

标准公式：score(d) = Σ_i 1 / (k + rank_i(d))，rank 从 1 开始，k 默认 60。
只依赖排名、不依赖分数量纲 —— 这正是它能把 BM25（无界）和余弦（0~1）缝在一起的原因。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from runtime.rag.types import Chunk


@dataclass(slots=True)
class FusionResult:
    chunks: list[Chunk]
    # chunk_id -> 各召回源排名，排障时非常有用
    contributions: dict[str, dict[str, int]] = field(default_factory=dict)


def rrf_fuse(
    ranked_lists: dict[str, list[Chunk]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    limit: int = 20,
) -> FusionResult:
    """按 RRF 融合多路召回结果。

    ranked_lists 的 key 是召回源名称（如 "dense" / "sparse"），value 必须已按相关性降序排列。
    传 weights 时可对某一路加权（如稀疏召回更信任时 weights={"sparse": 1.5}）。
    """
    if k <= 0:
        raise ValueError("RRF 的 k 必须为正数")

    weights = weights or {}
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    contributions: dict[str, dict[str, int]] = {}

    for source, items in ranked_lists.items():
        weight = weights.get(source, 1.0)
        for rank, chunk in enumerate(items, start=1):
            # 同一 chunk 在多路出现：分数累加，但正文只保留首次出现的实例
            by_id.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight * (1.0 / (k + rank))
            contributions.setdefault(chunk.id, {})[source] = rank

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    fused: list[Chunk] = []
    for chunk_id, score in ordered:
        chunk = by_id[chunk_id]
        merged_scores = dict(chunk.scores)
        merged_scores["rrf"] = score
        fused.append(
            Chunk(id=chunk.id, text=chunk.text, metadata=chunk.metadata, score=score, scores=merged_scores)
        )
    return FusionResult(chunks=fused, contributions=contributions)
