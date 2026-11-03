"""RAG 领域类型。

这里只放"跨模块传递"的数据结构：
  Document  —— 解析层产物
  Chunk     —— 切分层产物，带完整权限元数据
  AccessContext —— 检索时唯一的身份来源
  RetrievalTrace —— 一次检索的中间结果与耗时（用于评估与排障，不含正文）
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def stable_id(*parts: str) -> str:
    digest = hashlib.blake2b("\x1f".join(parts).encode(), digest_size=10).hexdigest()
    return f"chk_{digest}"


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    tenant_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", self.doc_id))

    def citation(self) -> dict[str, Any]:
        """给前端渲染"引用来源"的最小结构。"""
        return {
            "chunk_id": self.id,
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.metadata.get("source", ""),
            "section": self.metadata.get("section", ""),
            "snippet": self.text[:280],
            "score": round(self.scores.get("rerank", self.score), 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


@dataclass(slots=True)
class AccessContext:
    """一次检索的身份上下文。tenant_id 缺失视为非法调用。"""

    tenant_id: str
    user_id: str = ""
    roles: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)

    @property
    def is_privileged(self) -> bool:
        return "admin" in self.roles or "*" in self.scopes

    def has_any_scope(self, scopes: list[str]) -> bool:
        return self.is_privileged or any(s in self.scopes for s in scopes)


@dataclass(slots=True)
class RetrievalTrace:
    """检索链路快照：评估脚本与 /v1/debug/retrieval 都吃这个结构。"""

    query: str
    variants: list[str] = field(default_factory=list)
    dense_ids: list[str] = field(default_factory=list)
    sparse_ids: list[str] = field(default_factory=list)
    fused_ids: list[str] = field(default_factory=list)
    reranked_ids: list[str] = field(default_factory=list)
    filtered_out: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    backend: dict[str, str] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "variants": self.variants,
            "dense_ids": self.dense_ids,
            "sparse_ids": self.sparse_ids,
            "fused_ids": self.fused_ids,
            "reranked_ids": self.reranked_ids,
            "filtered_out": self.filtered_out,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "total_ms": self.total_ms,
            "backend": self.backend,
        }


@dataclass(slots=True)
class RetrievedContext:
    chunks: list[Chunk] = field(default_factory=list)
    trace: RetrievalTrace | None = None

    def as_prompt_block(self) -> str:
        """把检索结果拼成提示词片段。编号是强制的：模型必须能指向 [1][2]。"""
        if not self.chunks:
            return "（知识库未检索到相关内容）"
        blocks = []
        for idx, chunk in enumerate(self.chunks, start=1):
            blocks.append(f"[{idx}] 《{chunk.title}》 section={chunk.metadata.get('section', '-')}\n{chunk.text}")
        return "\n\n".join(blocks)

    def citations(self) -> list[dict[str, Any]]:
        return [c.citation() for c in self.chunks]
