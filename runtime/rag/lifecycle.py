"""索引生命周期：内容指纹、变更判定、清单持久化。

为什么单独一个模块？因为增量索引翻车的地方从来不是"怎么比对"，
而是**比错了维度**。两个指纹是刻意分开的：

  content_hash  只算正文。变了才需要重新切分 + 重新嵌入（贵）。
  policy_hash   只算可见性字段（tenant_id / classification /
                allowed_scopes / allowed_roles）。变了不改向量，
                但**必须重新写入元数据**（便宜，可跳过就是事故）。

一个真实的反例：某份文档从 internal 上调为 confidential，正文一字未改。
若只对正文做哈希，它会被判为 unchanged 而整体跳过，库里留着旧的低密级
元数据 —— 结果是全员可见。这类"静默权限降级"不会报错、不会告警，
只会在某次合规检查里被发现。

title / source / section 刻意**不**进 policy_hash：它们只影响展示，
改了不该触发重新嵌入。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from runtime.rag.types import Document

MANIFEST_VERSION = 1

#: 影响"谁能看见这个 chunk"的字段。与 runtime/rag/access.py::chunk_visible 一一对应，
#: 改动任一侧都必须同步另一侧，否则增量判定会出现与权限判定不一致的盲区。
POLICY_FIELDS: tuple[str, ...] = ("classification", "allowed_scopes", "allowed_roles")

ChangeKind = Literal["added", "updated_content", "updated_policy", "unchanged"]

_DIGEST_LEN = 32


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_DIGEST_LEN]


def _as_sorted_list(value: Any) -> list[str]:
    """把 `allowed_roles: hr,finance` / `["hr","finance"]` 归一成有序列表。

    排序是必须的：`allowed_roles: hr,finance` 与 `finance,hr` 权限完全等价，
    若按原序遍历哈希，仅调整书写顺序就会触发一次无谓的全量重嵌入。
    拆分规则与 access._as_list 保持一致（字符串按逗号切）。
    """
    if not value:
        return []
    if isinstance(value, str):
        return sorted(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set)):
        return sorted(str(item) for item in value)
    return [str(value)]


def content_hash(doc: Document) -> str:
    """正文指纹。只取 text：同样的正文切成同样的 chunk，复用已有向量。"""
    return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()[:_DIGEST_LEN]


def policy_hash(doc: Document) -> str:
    """可见性指纹。字段集合见 POLICY_FIELDS 的说明。"""
    payload: dict[str, Any] = {"tenant_id": doc.tenant_id or ""}
    for name in POLICY_FIELDS:
        payload[name] = _as_sorted_list(doc.metadata.get(name))
    return _digest(payload)


@dataclass(slots=True)
class DocState:
    """清单里一条文档记录。不存正文，只存指纹与规模。"""

    doc_id: str
    content_hash: str
    policy_hash: str
    chunks: int
    updated_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content_hash": self.content_hash,
            "policy_hash": self.policy_hash,
            "chunks": self.chunks,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DocState:
        return cls(
            doc_id=str(raw["doc_id"]),
            content_hash=str(raw.get("content_hash", "")),
            policy_hash=str(raw.get("policy_hash", "")),
            chunks=int(raw.get("chunks", 0)),
            updated_at=float(raw.get("updated_at", 0.0)),
        )


@dataclass(slots=True)
class IndexManifest:
    """doc_id → 指纹的清单，落盘为 index_manifest.json。

    单写者假设：多个 ingest 进程同时写同一目录会互相覆盖。生产环境应把
    清单放到与向量库同一套事务边界里（PG 表或对象存储 + 条件写）。
    """

    docs: dict[str, DocState] = field(default_factory=dict)
    version: int = MANIFEST_VERSION

    def __len__(self) -> int:
        return len(self.docs)

    def get(self, doc_id: str) -> DocState | None:
        return self.docs.get(doc_id)

    def put(self, doc_id: str, *, content: str, policy: str, chunks: int) -> DocState:
        state = DocState(
            doc_id=doc_id,
            content_hash=content,
            policy_hash=policy,
            chunks=chunks,
            updated_at=time.time(),
        )
        self.docs[doc_id] = state
        return state

    def drop(self, doc_id: str) -> bool:
        return self.docs.pop(doc_id, None) is not None

    def total_chunks(self) -> int:
        return sum(s.chunks for s in self.docs.values())

    # ---------------- 持久化

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "index_manifest.json"
        payload = {
            "version": self.version,
            "docs": {doc_id: state.as_dict() for doc_id, state in self.docs.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> IndexManifest:
        path = Path(directory) / "index_manifest.json"
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 清单坏了就当作没有：宁可全量重建一次，也不要基于不可信的指纹跳过文档
            return cls()
        docs = {
            str(doc_id): DocState.from_dict(raw)
            for doc_id, raw in (payload.get("docs") or {}).items()
        }
        return cls(docs=docs, version=int(payload.get("version", MANIFEST_VERSION)))


def plan_changes(
    documents: list[Document],
    manifest: IndexManifest,
    *,
    force: bool = False,
) -> tuple[dict[str, ChangeKind], list[str]]:
    """纯函数：算出每份文档要做什么，以及哪些文档已从输入中消失。

    返回 (doc_id → 变更类型, 消失的 doc_id 列表)。

    force=True 时全部按需重建（已有文档记作 updated_content，便于报告如实反映）。
    无 tenant_id 的文档由调用方先行过滤 —— 它们不入库，也不能进清单，
    否则下一轮会被判成"消失"而触发一次无意义的删除。
    """
    kinds: dict[str, ChangeKind] = {}
    seen: set[str] = set()

    for doc in documents:
        seen.add(doc.doc_id)
        previous = manifest.get(doc.doc_id)
        if previous is None:
            kinds[doc.doc_id] = "added"
            continue
        if force or content_hash(doc) != previous.content_hash:
            kinds[doc.doc_id] = "updated_content"
        elif policy_hash(doc) != previous.policy_hash:
            # 必须落到 updated_policy：正文与向量都能复用，但元数据要重写
            kinds[doc.doc_id] = "updated_policy"
        else:
            kinds[doc.doc_id] = "unchanged"

    removed = sorted(doc_id for doc_id in manifest.docs if doc_id not in seen)
    return kinds, removed


@dataclass(slots=True)
class SyncReport:
    """一次同步的结果。字段是按"运维真正会问的问题"排的：

    哪些该重建、跳过了多少、删了没有、为什么退化成全量。
    """

    added: list[str] = field(default_factory=list)
    updated_content: list[str] = field(default_factory=list)
    updated_policy: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    pending_removal: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    documents: int = 0
    chunks: int = 0
    chunks_deleted: int = 0
    split_seconds: float = 0.0
    embed_seconds: float = 0.0
    forced: bool = False
    force_reason: str = ""

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated_content) + len(self.updated_policy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "added": len(self.added),
            "updated_content": len(self.updated_content),
            "updated_policy": len(self.updated_policy),
            "unchanged": len(self.unchanged),
            "removed": len(self.removed),
            "pending_removal": len(self.pending_removal),
            "documents": self.documents,
            "chunks": self.chunks,
            "chunks_deleted": self.chunks_deleted,
            "split_seconds": round(self.split_seconds, 3),
            "embed_seconds": round(self.embed_seconds, 3),
            "forced": self.forced,
            "force_reason": self.force_reason,
        }
