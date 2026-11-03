"""切分策略：RecursiveChunker 与 SemanticChunker，可配置切换。

两种策略的取舍（写进文档，避免后人盲选）：
  Recursive —— 快、稳定、无模型依赖，是默认兜底；缺点是按字符边界切，可能切断语义。
  Semantic  —— 按相邻句向量距离找"话题断点"，长文档上召回更完整；缺点是索引期慢（要算 embedding）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from runtime.rag.embeddings import Embedder
from runtime.rag.types import Chunk, Document, stable_id

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


@dataclass(slots=True)
class Block:
    section: str
    text: str


def split_blocks(text: str) -> list[Block]:
    """把文档切成"带标题路径"的块，供切分器继续细切。"""
    blocks: list[Block] = []
    headings: dict[int, str] = {}
    buf: list[str] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        if content:
            path = " / ".join(headings[k] for k in sorted(headings) if headings.get(k))
            blocks.append(Block(section=path or "正文", text=content))
        buf.clear()

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            headings[level] = match.group(2).strip()
            for deeper in [k for k in headings if k > level]:
                headings.pop(deeper, None)
        else:
            buf.append(line)
    flush()
    return blocks or [Block(section="正文", text=text)]


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _recursive_split(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    sep = next((s for s in separators if s and s in text), None)
    if sep is None:
        return _hard_split(text, size)

    parts = text.split(sep)
    rest = separators[separators.index(sep) + 1 :]
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(part) > size:
            pieces.extend(_recursive_split(part, rest, size) if rest else _hard_split(part, size))
            current = ""
        else:
            current = part
    if current:
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


class Chunker(Protocol):
    name: str

    async def split(self, doc: Document) -> list[Chunk]: ...


class RecursiveChunker:
    name = "recursive"

    def __init__(self, size: int = 480, overlap: int = 80) -> None:
        self.size = size
        self.overlap = min(overlap, size // 2)
        # 中文优先按句号/分号断，其次逗号，最后空格
        self.separators = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]

    async def split(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for block_index, block in enumerate(split_blocks(doc.text)):
            pieces = _recursive_split(block.text, self.separators, self.size)
            for index, piece in enumerate(pieces):
                text = piece
                if index > 0 and self.overlap:  # 上下文重叠，缓解跨块断裂
                    tail = pieces[index - 1][-self.overlap :]
                    text = f"{tail}{piece}"
                chunks.append(
                    _make_chunk(doc, text, block.section, len(chunks), extra={"block": block_index})
                )
        return chunks


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        *,
        size: int = 480,
        break_percentile: float = 80.0,
        min_chars: int = 120,
    ) -> None:
        self.embedder = embedder
        self.size = size
        self.break_percentile = break_percentile
        self.min_chars = min_chars

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_RE.findall(text) if s and s.strip()]

    async def split(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for block_index, block in enumerate(split_blocks(doc.text)):
            sentences = self._sentences(block.text)
            if len(sentences) <= 2:
                chunks.append(_make_chunk(doc, block.text, block.section, len(chunks), extra={"block": block_index}))
                continue

            vectors = await self.embedder.embed_documents(sentences)
            distances = [
                1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
            ]
            threshold = _percentile(distances, self.break_percentile)

            # 断点后按 size 上限再合并，避免语义段落过长
            groups: list[list[str]] = [[]]
            for i, sentence in enumerate(sentences):
                current_len = sum(len(s) for s in groups[-1])
                if groups[-1] and (distances[i - 1] > threshold or current_len + len(sentence) > self.size):
                    groups.append([])
                groups[-1].append(sentence)

            for group in groups:
                text = "".join(group).strip()
                if not text:
                    continue
                # 太碎的段并回上一块，减少噪声片段
                if len(text) < self.min_chars and chunks and chunks[-1].metadata.get("block") == block_index:
                    chunks[-1] = Chunk(
                        id=chunks[-1].id,
                        text=f"{chunks[-1].text}\n{text}",
                        metadata=chunks[-1].metadata,
                    )
                    continue
                chunks.append(_make_chunk(doc, text, block.section, len(chunks), extra={"block": block_index}))
        return chunks


def _make_chunk(doc: Document, text: str, section: str, index: int, extra: dict | None = None) -> Chunk:
    metadata = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "source": doc.source,
        "tenant_id": doc.tenant_id,
        "section": section,
        "chunk_index": index,
        "classification": doc.metadata.get("classification", "internal"),
        "allowed_scopes": doc.metadata.get("allowed_scopes", []),
        "allowed_roles": doc.metadata.get("allowed_roles", []),
        **(extra or {}),
    }
    return Chunk(id=stable_id(doc.doc_id, str(index), text[:64]), text=text, metadata=metadata)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[k]


def build_chunker(name: str, *, embedder: Embedder, size: int, overlap: int, break_percentile: float) -> Chunker:
    if name == "semantic":
        return SemanticChunker(embedder, size=size, break_percentile=break_percentile)
    if name == "recursive":
        return RecursiveChunker(size=size, overlap=overlap)
    raise ValueError(f"未知切分策略: {name}")
