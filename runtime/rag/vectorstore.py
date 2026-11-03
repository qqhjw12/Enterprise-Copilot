"""向量存储（内存 + Qdrant 两种实现）。

对外只暴露 upsert / search / delete_doc / all_chunks 四个动作，
pipeline 不关心底层是 Qdrant 还是内存数组 —— 这是能"零依赖跑单测"的前提。
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from runtime.rag.types import Chunk


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...

    async def search(
        self,
        vector: list[float],
        top_k: int = 20,
        *,
        predicate: Callable[[Chunk], bool] | None = None,
    ) -> list[tuple[Chunk, float]]: ...

    async def delete_doc(self, doc_id: str) -> int: ...

    async def all_chunks(self) -> list[Chunk]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """内存向量库。开发、单测、离线评估的默认后端。

    注意：多副本部署时必须换 Qdrant，否则每个副本看到的知识库不一样。
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vector
        return len(chunks)

    async def search(
        self,
        vector: list[float],
        top_k: int = 20,
        *,
        predicate: Callable[[Chunk], bool] | None = None,
    ) -> list[tuple[Chunk, float]]:
        scored: list[tuple[Chunk, float]] = []
        for chunk_id, chunk in self._chunks.items():
            if predicate is not None and not predicate(chunk):
                continue
            scored.append((chunk, _cosine(vector, self._vectors[chunk_id])))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    async def delete_doc(self, doc_id: str) -> int:
        victims = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in victims:
            self._chunks.pop(cid, None)
            self._vectors.pop(cid, None)
        return len(victims)

    async def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    # ---------------- 持久化：让离线评估与"重启不丢库"成为可能

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [
                {"id": c.id, "text": c.text, "metadata": c.metadata, "vector": self._vectors[c.id]}
                for c in self._chunks.values()
            ]
        }
        (path / "vector_store.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, directory: str | Path) -> int:
        file = Path(directory) / "vector_store.json"
        if not file.exists():
            return 0
        payload = json.loads(file.read_text(encoding="utf-8"))
        for item in payload.get("chunks", []):
            chunk = Chunk(id=item["id"], text=item["text"], metadata=item.get("metadata", {}))
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = item["vector"]
        return len(self._chunks)


class QdrantVectorStore:
    """Qdrant 后端。向量检索走服务端，权限过滤用 payload filter 下推。"""

    def __init__(
        self,
        url: str,
        collection: str,
        dim: int,
        *,
        api_key: str = "",
        tenant_field: str = "tenant_id",
        prefer_grpc: bool = False,
    ) -> None:
        from qdrant_client import AsyncQdrantClient  # 可选依赖

        self.collection = collection
        self.dim = dim
        self.tenant_field = tenant_field
        self._client = AsyncQdrantClient(url=url, api_key=api_key or None, prefer_grpc=prefer_grpc)
        self._ready = False

    async def _ensure_collection(self) -> None:
        if self._ready:
            return
        from qdrant_client.http import models as qm

        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self.collection not in names:
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )
        self._ready = True

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from qdrant_client.http import models as qm

        await self._ensure_collection()
        points = [
            qm.PointStruct(
                id=_to_point_id(chunk.id),
                vector=vector,
                payload={"chunk_id": chunk.id, "text": chunk.text, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    async def search(
        self,
        vector: list[float],
        top_k: int = 20,
        *,
        predicate: Callable[[Chunk], bool] | None = None,
    ) -> list[tuple[Chunk, float]]:
        from qdrant_client.http import models as qm

        await self._ensure_collection()
        # predicate 无法下推（Python 闭包），只能放大候选量后在本地二次过滤；
        # 生产环境应把租户/密级写成 payload filter 下推，见 docs
        fetch_k = top_k if predicate is None else max(top_k * 8, 64)
        hits = await self._client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=fetch_k,
            with_payload=True,
            query_filter=qm.Filter(must=[]),
        )
        out: list[tuple[Chunk, float]] = []
        for hit in hits:
            payload: dict[str, Any] = dict(hit.payload or {})
            text = payload.pop("text", "")
            chunk_id = payload.pop("chunk_id", str(hit.id))
            chunk = Chunk(id=chunk_id, text=text, metadata=payload)
            if predicate is not None and not predicate(chunk):
                continue
            out.append((chunk, float(hit.score)))
            if len(out) >= top_k:
                break
        return out

    async def delete_doc(self, doc_id: str) -> int:
        from qdrant_client.http import models as qm

        await self._ensure_collection()
        await self._client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
            ),
            wait=True,
        )
        return -1  # Qdrant 不回传删除数量，调用方不应依赖此值

    async def all_chunks(self) -> list[Chunk]:
        await self._ensure_collection()
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            for point in points:
                payload: dict[str, Any] = dict(point.payload or {})
                text = payload.pop("text", "")
                chunk_id = payload.pop("chunk_id", str(point.id))
                chunks.append(Chunk(id=chunk_id, text=text, metadata=payload))
            if offset is None:
                break
        return chunks


def _to_point_id(chunk_id: str) -> str:
    """Qdrant 的 point id 只接受 UUID/无符号整数；这里做稳定映射。"""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_vector_store(backend: str, *, collection: str, dim: int, url: str = "", api_key: str = "") -> VectorStore:
    backend = (backend or "memory").lower()
    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "qdrant":
        return QdrantVectorStore(url=url, collection=collection, dim=dim, api_key=api_key)
    raise ValueError(f"未知的向量库后端: {backend}")
