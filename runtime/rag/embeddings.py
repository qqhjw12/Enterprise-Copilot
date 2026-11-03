"""Embedding 提供方（可插拔）。

三个实现解决三种场景：
  HashingEmbedder        —— 零依赖、确定性，单测与离线评估用（不需要模型下载）
  SentenceTransformer    —— 生产语义检索主力（BGE 系列）
  OpenAICompatEmbedder   —— 走 OpenAI 兼容 /embeddings 的服务
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class HashingEmbedder:
    """基于字符 n-gram 的特征哈希。

    它不是"假 embedding"：同样的文本永远得到同样的向量，语义相近的句子
    （共享大量 n-gram）余弦相似度确实更高，因此 RRF/重排链路能真实工作。
    只是它捕捉不到同义改写 —— 所以 README 里的指标标注为 offline baseline。
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        grams: list[str] = []
        for i, token in enumerate(tokens):
            grams.append(token)
            if i + 1 < len(tokens):
                grams.append(token + tokens[i + 1])
        for gram in grams:
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


class SentenceTransformerEmbedder:
    """BGE / m3e 等本地模型。首次调用会下载权重，故懒加载。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.name = model_name
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # 可选依赖

            self._model = SentenceTransformer(self.name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim or 512

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()

        def _run() -> list[list[float]]:
            vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
            return [v.tolist() for v in vectors]

        return await asyncio.to_thread(_run)

    async def embed_query(self, text: str) -> list[float]:
        # BGE 系列检索任务建议加指令前缀
        return (await self.embed_documents([f"为这个句子生成表示以用于检索相关文章：{text}"]))[0]


class OpenAICompatEmbedder:
    def __init__(self, model: str, base_url: str, api_key: str, dim: int = 1024) -> None:
        self.name = model
        self.dim = dim
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    async def _call(self, texts: list[str]) -> list[list[float]]:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self.name, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            out.extend(await self._call(texts[i : i + 32]))
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self._call([text]))[0]


def build_embedder(provider: str, *, model: str, dim: int, base_url: str = "", api_key: str = "") -> Embedder:
    provider = (provider or "hashing").lower()
    if provider in {"hashing", "offline", "fake"}:
        return HashingEmbedder(dim=dim or 512)
    if provider in {"sentence-transformers", "st", "bge"}:
        return SentenceTransformerEmbedder(model_name=model)
    if provider in {"openai", "openai-compat"}:
        return OpenAICompatEmbedder(model=model, base_url=base_url or "https://api.openai.com/v1", api_key=api_key, dim=dim)
    raise ValueError(f"未知的 embedding provider: {provider}")
