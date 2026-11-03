"""混合检索管线：Dense + BM25 → RRF → Rerank → 组装上下文。

固定顺序，不提供"跳过某一步"的开关（除 reranker=none 的压测模式）：
顺序一旦可配置，线上就会出现"某个项目偷偷关掉权限过滤"这种事。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.config import RagSettings, get_rag_settings
from runtime.rag import access
from runtime.rag.bm25 import BM25Index
from runtime.rag.chunkers import build_chunker
from runtime.rag.embeddings import Embedder, build_embedder
from runtime.rag.fusion import rrf_fuse
from runtime.rag.lifecycle import IndexManifest, SyncReport, content_hash, plan_changes, policy_hash
from runtime.rag.parsers import load_corpus
from runtime.rag.query_transform import expand_queries
from runtime.rag.rerank import Reranker, build_reranker
from runtime.rag.types import AccessContext, Chunk, Document, RetrievalTrace, RetrievedContext
from runtime.rag.vectorstore import InMemoryVectorStore, VectorStore, build_vector_store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexStats:
    documents: int = 0
    chunks: int = 0
    embed_seconds: float = 0.0
    split_seconds: float = 0.0
    skipped: list[str] = field(default_factory=list)


class HybridRAG:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker,
        chunker,
        settings: RagSettings,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.chunker = chunker
        self.settings = settings

    # ------------------------------------------------------------ 构建

    @classmethod
    def from_settings(cls, settings: RagSettings | None = None, *, embedder: Embedder | None = None) -> HybridRAG:
        settings = settings or get_rag_settings()
        embedder = embedder or build_embedder(
            settings.embedding_provider,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
        store = build_vector_store(
            settings.vector_backend,
            collection=settings.collection,
            dim=getattr(embedder, "dim", settings.embedding_dim),
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        return cls(
            embedder=embedder,
            store=store,
            bm25=BM25Index(),
            reranker=build_reranker(settings.reranker, model=settings.reranker_model),
            chunker=build_chunker(
                settings.chunker,
                embedder=embedder,
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                break_percentile=settings.semantic_break_percentile,
            ),
            settings=settings,
        )

    # ------------------------------------------------------------ 索引

    async def _split(self, documents: list[Document], *, skipped: list[str]) -> tuple[list[Chunk], int, float]:
        """逐篇切分，单篇失败不中断整批。返回 (chunks, 成功篇数, 耗时)。"""
        chunks: list[Chunk] = []
        ok = 0
        seconds = 0.0
        for doc in documents:
            started = time.perf_counter()
            try:
                doc_chunks = await self.chunker.split(doc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("chunk_failed", extra={"doc_id": doc.doc_id, "error": str(exc)})
                skipped.append(f"{doc.doc_id}: {exc}")
                continue
            seconds += time.perf_counter() - started
            ok += 1
            chunks.extend(doc_chunks)
        return chunks, ok, seconds

    async def _embed(self, chunks: list[Chunk]) -> tuple[list[list[float]], float]:
        vectors: list[list[float]] = []
        started = time.perf_counter()
        batch_size = max(1, self.settings.embedding_batch_size)
        for i in range(0, len(chunks), batch_size):
            vectors.extend(await self.embedder.embed_documents([c.text for c in chunks[i : i + batch_size]]))
        return vectors, time.perf_counter() - started

    async def index_documents(self, documents: list[Document]) -> IndexStats:
        """全量写入：不比对指纹，已有 chunk 直接覆盖。

        保留这个方法是因为它语义简单（"把这些文档写进去"），测试与评估脚本都依赖。
        日常灌库请用 sync_documents —— 它会跳过没变的文档。
        """
        stats = IndexStats()
        valid: list[Document] = []
        for doc in documents:
            if not doc.tenant_id:
                # 没有租户归属的文档不入库：宁可少索引，也不能让它在检索期变成"公共资产"
                stats.skipped.append(f"{doc.doc_id}: missing tenant_id")
                continue
            valid.append(doc)

        chunk_batch, stats.documents, stats.split_seconds = await self._split(valid, skipped=stats.skipped)
        if not chunk_batch:
            return stats

        vectors, stats.embed_seconds = await self._embed(chunk_batch)
        await self.store.upsert(chunk_batch, vectors)
        self.bm25.add(chunk_batch)
        stats.chunks = len(chunk_batch)
        logger.info(
            "index_built",
            extra={
                "documents": stats.documents,
                "chunks": stats.chunks,
                "embed_seconds": round(stats.embed_seconds, 3),
            },
        )
        return stats

    async def sync_documents(
        self,
        documents: list[Document],
        *,
        prune: bool = False,
        force: bool = False,
        dry_run: bool = False,
    ) -> SyncReport:
        """增量同步：只重建真正变了的文档。

        prune=False 时，已从输入中消失的文档只在报告里记一笔（pending_removal），
        不删 —— 因为 index_documents 单文件追加是合法用法，不能顺手清空别人。
        目录级同步请显式传 prune=True。

        关于状态丢失：manifest 说"这些文档没变"而向量库是空的（换了机器、
        没带 --persist、内存后端重启），那增量就是自杀 —— 会得到一个空索引
        却以为同步成功。所以这里先校验库是否为空，不一致就退化成全量重建。
        """
        report = SyncReport()
        manifest = IndexManifest.load(self.settings.persist_dir)

        valid: list[Document] = []
        for doc in documents:
            if not doc.tenant_id:
                report.skipped.append(f"{doc.doc_id}: missing tenant_id")
                continue
            valid.append(doc)

        indexed = await self.store.all_chunks()
        forced = force
        reason = "caller_forced" if force else ""
        if not forced and manifest.docs and not indexed:
            forced, reason = True, "manifest_present_but_store_empty"
        elif not forced and indexed and not manifest.docs:
            # 有数据却没有清单（旧版本升级上来）：无法证明哪些 chunk 属于哪些文档，
            # 只能全量重建，否则内容变更过的文档会残留旧 chunk。
            forced, reason = True, "store_present_but_manifest_missing"
        if forced and reason:
            logger.warning("index_sync_forced", extra={"reason": reason, "indexed_chunks": len(indexed)})

        report.forced = forced
        report.force_reason = reason

        kinds, gone = plan_changes(valid, manifest, force=forced)

        if dry_run:
            # 只汇报计划，不碰库。误把子目录当全量目录传进来时，这是唯一的后悔药。
            for doc_id, kind in kinds.items():
                if kind == "added":
                    report.added.append(doc_id)
                elif kind == "updated_policy":
                    report.updated_policy.append(doc_id)
                elif kind == "updated_content":
                    report.updated_content.append(doc_id)
                else:
                    report.unchanged.append(doc_id)
            report.removed, report.pending_removal = (gone, []) if prune else ([], gone)
            report.documents = len(valid)
            return report

        # 1) 下架文档：旧内容继续被检索到，比"少收录"严重得多
        for doc_id in gone:
            if not prune:
                report.pending_removal.append(doc_id)
                continue
            deleted = await self.store.delete_doc(doc_id)
            self.bm25.remove_doc(doc_id)
            manifest.drop(doc_id)
            report.removed.append(doc_id)
            report.chunks_deleted += max(0, deleted)

        # 2) 需要重建的文档：先清旧 chunk 再写新的。
        # 内容变更时 chunk id 会变（stable_id 含正文），不清就会新旧并存，
        # 检索同时命中"现行版"和"废止版"。
        rebuild = [d for d in valid if kinds.get(d.doc_id) != "unchanged"]
        for doc in rebuild:
            if manifest.get(doc.doc_id) is not None:
                deleted = await self.store.delete_doc(doc.doc_id)
                if deleted > 0:
                    report.chunks_deleted += deleted
                self.bm25.remove_doc(doc.doc_id)

        chunk_batch, report.documents, report.split_seconds = await self._split(rebuild, skipped=report.skipped)

        if chunk_batch:
            vectors, report.embed_seconds = await self._embed(chunk_batch)
            await self.store.upsert(chunk_batch, vectors)
            self.bm25.add(chunk_batch)
            report.chunks = len(chunk_batch)

        # 3) 回写清单。只有真正落到库里的文档才进清单 —— 切分失败的文档不算"已同步"，
        # 否则下一轮会被判成 unchanged，错误被永久固化。
        per_doc: dict[str, int] = {}
        for chunk in chunk_batch:
            per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
        for doc in rebuild:
            if doc.doc_id not in per_doc:
                continue
            kind = kinds.get(doc.doc_id)
            manifest.put(
                doc.doc_id,
                content=content_hash(doc),
                policy=policy_hash(doc),
                chunks=per_doc[doc.doc_id],
            )
            if kind == "added":
                report.added.append(doc.doc_id)
            elif kind == "updated_policy":
                # 单独计数：这是"权限口径变了"，审计上应能看到，不该混进内容更新
                report.updated_policy.append(doc.doc_id)
            else:
                report.updated_content.append(doc.doc_id)
        report.unchanged = [d.doc_id for d in valid if kinds.get(d.doc_id) == "unchanged"]

        self._persist_manifest(manifest)
        logger.info("index_synced", extra=report.as_dict())
        return report

    def _persist_manifest(self, manifest: IndexManifest) -> None:
        """清单描述的是"库里真实存在什么"。库里一条都没有时不落盘，
        否则下次启动会读到一份指向空库的清单，把全部文档误判为 unchanged。"""
        if not manifest.docs and self.bm25.chunks:
            return
        try:
            manifest.save(self.settings.persist_dir)
        except OSError as exc:
            logger.warning("manifest_save_failed", extra={"error": str(exc)})


    async def index_directory(self, root: str | Path, *, tenant_id: str, metadata: dict[str, Any] | None = None) -> IndexStats:
        documents = load_corpus(root, tenant_id=tenant_id, metadata=metadata)
        return await self.index_documents(documents)

    async def sync_directory(
        self,
        root: str | Path,
        *,
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
        prune: bool = True,
        force: bool = False,
        dry_run: bool = False,
    ) -> SyncReport:
        """目录级增量同步。目录是"完整真相"，所以 prune 默认为 True：
        源文件删了，索引里也必须消失。"""
        documents = load_corpus(root, tenant_id=tenant_id, metadata=metadata)
        return await self.sync_documents(documents, prune=prune, force=force, dry_run=dry_run)

    async def rebuild_sparse_index(self) -> int:
        """把向量库里的全量 chunk 重建成 BM25 索引（进程重启后调用）。"""
        chunks = await self.store.all_chunks()
        self.bm25 = BM25Index()
        if chunks:
            self.bm25.add(chunks)
        return len(chunks)

    # ------------------------------------------------------------ 检索

    async def retrieve(
        self,
        query: str,
        ctx: AccessContext,
        *,
        top_k: int | None = None,
        trace: bool = True,
    ) -> RetrievedContext:
        if not ctx.tenant_id:
            raise ValueError("AccessContext.tenant_id 不能为空")

        settings = self.settings
        top_k = top_k or settings.rerank_top_n
        predicate = access.make_predicate(ctx)
        trace_data = RetrievalTrace(
            query=query,
            backend={"vector": settings.vector_backend, "embedding": getattr(self.embedder, "name", "?")},
        )

        # 1) Dense：始终用原始问题（改写的目的是补词面证据，不是改变语义意图）
        started = time.perf_counter()
        query_vector = await self.embedder.embed_query(query)
        dense_hits = await self.store.search(query_vector, top_k=settings.dense_top_k, predicate=predicate)
        trace_data.timings_ms["dense"] = (time.perf_counter() - started) * 1000
        dense_chunks = [c for c, _ in dense_hits]
        for chunk, score in dense_hits:
            chunk.scores["dense"] = score

        # 2) Sparse：可选多查询（RM3-lite），每个变体单独召回，最后一起进 RRF
        variants = [query]
        if settings.query_transform == "multi_query":
            started = time.perf_counter()
            variants = expand_queries(query, max_variants=settings.query_variants)
            trace_data.timings_ms["query_transform"] = (time.perf_counter() - started) * 1000
        trace_data.variants = variants

        started = time.perf_counter()
        rankings: dict[str, list[Chunk]] = {"dense": dense_chunks}
        sparse_chunks: list[Chunk] = []
        for idx, variant in enumerate(variants):
            hits = self.bm25.search(variant, top_k=settings.sparse_top_k, predicate=predicate)
            chunk_list = [c for c, _ in hits]
            rankings[f"sparse_{idx}"] = chunk_list
            if idx == 0:
                sparse_chunks = chunk_list
                for chunk, score in hits:
                    chunk.scores["sparse"] = score
        trace_data.timings_ms["sparse"] = (time.perf_counter() - started) * 1000

        # 3) RRF
        started = time.perf_counter()
        fused = rrf_fuse(
            rankings,
            k=settings.rrf_k,
            limit=max(settings.rerank_top_n * 4, 20),
        ).chunks
        trace_data.timings_ms["fusion"] = (time.perf_counter() - started) * 1000

        # 4) Rerank：只有"受控词表命中"时才用扩展词重排。
        # 关键在"受控"：语料外问题没有词表命中（variants 只有原问题），
        # 重排就回到原问题，不会被反馈词自我强化。曾经用伪相关反馈生成扩展词，
        # 结果噪声文档贡献的扩展词把覆盖率刷到 1.0，语料外问题直接被答成"年假"。
        started = time.perf_counter()
        rerank_query = " ".join(variants) if len(variants) > 1 else query
        final = await self.reranker.rerank(rerank_query, fused, top_k)
        trace_data.timings_ms["rerank"] = (time.perf_counter() - started) * 1000

        if settings.score_threshold > 0:
            final = [c for c in final if c.scores.get("rerank", c.score) >= settings.score_threshold]

        if trace:
            trace_data.dense_ids = [c.id for c in dense_chunks]
            trace_data.sparse_ids = [c.id for c in sparse_chunks]
            trace_data.fused_ids = [c.id for c in fused]
            trace_data.reranked_ids = [c.id for c in final]
            all_chunks = self.bm25.chunks
            visible = sum(1 for c in all_chunks if predicate(c))
            trace_data.filtered_out = max(0, len(all_chunks) - visible)

        return RetrievedContext(chunks=final, trace=trace_data)

    # ------------------------------------------------------------ 生成

    @staticmethod
    def build_answer_messages(question: str, context: RetrievedContext) -> list[dict[str, str]]:
        system = (
            "你是企业内部知识助手。只能依据【检索上下文】回答，不得编造。\n"
            "规则：\n"
            "1. 每个结论后用 [编号] 标注来源，例如：年假为 10 天 [2]。\n"
            "2. 上下文中没有的信息，直接回答“知识库中未收录该信息”，并建议联系对应部门。\n"
            "3. 用简体中文，先给结论再给依据，不要复述整个文档。"
        )
        user = f"【检索上下文】\n{context.as_prompt_block()}\n\n【问题】\n{question}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ------------------------------------------------------------ 持久化

    def persist(self, directory: str | Path | None = None) -> bool:
        """仅内存后端支持落盘；Qdrant 自带持久化。"""
        if not isinstance(self.store, InMemoryVectorStore):
            return False
        target = Path(directory or self.settings.persist_dir)
        self.store.save(target)
        return True

    async def load_persisted(self, directory: str | Path | None = None) -> int:
        if not isinstance(self.store, InMemoryVectorStore):
            return 0
        count = self.store.load(directory or self.settings.persist_dir)
        if count:
            await self.rebuild_sparse_index()
        return count
