"""BM25 稀疏检索。

不引入 rank-bm25 的理由：需要在打分时就应用权限过滤（先过滤再排序），
外部库只能"检索完再筛"，在强隔离场景里会泄漏候选集大小与文档存在性。

分词策略：中文字符 bigram + 英文/数字词元。
对短查询与专有名词（工单号、系统名）效果稳定，且不需要 jieba 这类词典依赖。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable

from runtime.rag.types import Chunk

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        span = match.group()
        if "\u4e00" <= span[0] <= "\u9fff":
            if len(span) == 1:
                tokens.append(span)
            else:
                tokens.extend(span[i : i + 2] for i in range(len(span) - 1))
        else:
            tokens.append(span)
    return tokens


class BM25Index:
    """内存版 BM25-Okapi。生产可替换为 OpenSearch（见 docs/architecture.md）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, Chunk] = {}
        self._term_freqs: dict[str, Counter[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._chunks[chunk.id] = chunk
            self._term_freqs[chunk.id] = Counter(tokens)
            self._doc_len[chunk.id] = len(tokens)
            for term in set(tokens):
                self._df[term] += 1
        self._recompute_avg()

    def remove_doc(self, doc_id: str) -> int:
        victims = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in victims:
            for term in set(self._term_freqs[cid]):
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]
            self._chunks.pop(cid, None)
            self._term_freqs.pop(cid, None)
            self._doc_len.pop(cid, None)
        self._recompute_avg()
        return len(victims)

    def _recompute_avg(self) -> None:
        n = len(self._doc_len)
        self._avg_len = (sum(self._doc_len.values()) / n) if n else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def idf(self, term: str) -> float:
        """公开的 IDF：查询改写（RM3）需要用它给反馈词加权。"""
        return self._idf(term)

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        predicate: Callable[[Chunk], bool] | None = None,
    ) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms or not self._chunks:
            return []
        scores: dict[str, float] = defaultdict(float)
        avg = self._avg_len or 1.0
        for term in terms:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for chunk_id, freqs in self._term_freqs.items():
                tf = freqs.get(term)
                if not tf:
                    continue
                dl = self._doc_len[chunk_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avg)
                scores[chunk_id] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        results: list[tuple[Chunk, float]] = []
        for chunk_id, score in ranked:
            chunk = self._chunks[chunk_id]
            if predicate is not None and not predicate(chunk):
                continue
            results.append((chunk, score))
            if len(results) >= top_k:
                break
        return results
