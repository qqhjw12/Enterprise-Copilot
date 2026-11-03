"""重排序器。

三级实现，按成本递增：
  NoneReranker       —— 直接截断（Only 用于压测/对照）
  HeuristicReranker  —— 词面覆盖 + RRF 先验，零模型、毫秒级，离线默认
  CrossEncoderReranker —— BGE-Reranker，真正的语义重排，生产推荐

为什么要重排：RRF 只融合"排名"，无法判断"这段文字是否真的回答了问题"。
Cross-Encoder 逐对打分能修正 BM25 的词汇偶然命中与向量召回的语义漂移。
"""
from __future__ import annotations

import asyncio
import math
from typing import Protocol

from runtime.rag.bm25 import tokenize
from runtime.rag.types import Chunk


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]: ...


class NoneReranker:
    name = "none"

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        return chunks[:top_n]


class HeuristicReranker:
    """无模型重排：IDF 主导的"证据强度"打分。

    为什么不是"数命中几个词"：中文问句里的"公司/有没有/怎么"几乎每篇文档都有，
    等权计数会让"公司有没有太空电梯补贴"仅凭命中"公司"就拿到能过阈值的高分，
    把无关文档塞进上下文。

    为什么也不是"IDF 加权覆盖率"：覆盖率会被问句里的冗余 bigram（"准是""是多"）
    稀释——它们在语料里根本不存在，却占着分母，导致"出差住宿标准是多少"
    这种真问题反而被压到阈值以下。

    所以这里用两层：
      1. 证据强度 presence = 命中词里最高的 IDF ÷ 本候选集的理论最大 IDF
         —— 直接回答"有没有命中一个'有辨识度'的词"
      2. 覆盖率 coverage = 命中词的 IDF 之和 ÷ 语料内查询词的 IDF 之和
         —— 防止"只命中一个稀有词、其余全忽略"的片段拿满分
    score = presence × (0.65 + 0.35 × coverage) × 长度惩罚

    局限：IDF 来自"本次召回的候选集"（约 20 条），不是全库统计；候选集 <8 条时
    统计不可靠，退化为等权（此时更像纯 RRF）。生产换 CrossEncoderReranker 后
    这一层不再是瓶颈，但阈值需按新分数量纲重新标定。
    """

    name = "heuristic"

    def __init__(self, prior_weight: float = 0.0) -> None:
        self.prior_weight = prior_weight

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if not chunks:
            return []
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return chunks[:top_n]

        chunk_terms = [set(tokenize(c.text)) for c in chunks]
        n = len(chunks)
        max_rrf = max((c.scores.get("rrf", 0.0) for c in chunks), default=1.0) or 1.0

        if n >= 8:
            df = {term: sum(1 for terms in chunk_terms if term in terms) for term in query_terms}
        else:  # 候选太少时 DF 不可信，当等权处理
            df = {term: 1 for term in query_terms}
        idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in query_terms}
        # df=1（全语料独一份）时的 IDF 视为上限，用它把 presence 压到 0~1
        idf_ceiling = math.log((n + 1) / 2) + 1.0
        in_corpus = [t for t in query_terms if df[t] > 0]
        total_idf = sum(idf[t] for t in in_corpus) or 1.0

        scored: list[tuple[float, Chunk]] = []
        for chunk, terms in zip(chunks, chunk_terms, strict=True):
            matched = [t for t in in_corpus if t in terms]
            if not matched:
                continue  # 一个语料内的查询词都没命中：直接出局，不占 top_n
            presence = max(idf[t] for t in matched) / idf_ceiling
            coverage = sum(idf[t] for t in matched) / total_idf
            prior = chunk.scores.get("rrf", 0.0) / max_rrf
            length_penalty = 1.0 if len(chunk.text) <= 800 else 0.92
            score = presence * (0.65 + 0.35 * coverage) * length_penalty
            score *= (1 - self.prior_weight) + self.prior_weight * prior
            scores = dict(chunk.scores)
            scores["coverage"] = coverage
            scores["rerank"] = score
            scored.append((score, Chunk(chunk.id, chunk.text, chunk.metadata, score=score, scores=scores)))

        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [c for _, c in scored[:top_n]]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # 可选依赖

            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if not chunks:
            return []
        model = self._load()
        pairs = [(query, c.text) for c in chunks]

        def _run() -> list[float]:
            return [float(s) for s in model.predict(pairs)]

        raw = await asyncio.to_thread(_run)
        # 归一化到 0~1，便于前端统一展示"相似度"
        lo, hi = min(raw), max(raw)
        span = (hi - lo) or 1.0
        scored: list[tuple[float, Chunk]] = []
        for score, chunk in zip(raw, chunks, strict=True):
            norm = (score - lo) / span if hi != lo else 1.0
            scores = dict(chunk.scores)
            scores["rerank"] = norm
            scored.append((norm, Chunk(chunk.id, chunk.text, chunk.metadata, score=norm, scores=scores)))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [c for _, c in scored[:top_n]]


def build_reranker(name: str, *, model: str = "BAAI/bge-reranker-base") -> Reranker:
    name = (name or "none").lower()
    if name in {"none", "noop", "off"}:
        return NoneReranker()
    if name in {"heuristic", "lexical"}:
        return HeuristicReranker()
    if name in {"cross-encoder", "crossencoder", "bge"}:
        return CrossEncoderReranker(model_name=model)
    raise ValueError(f"未知重排器: {name}")
