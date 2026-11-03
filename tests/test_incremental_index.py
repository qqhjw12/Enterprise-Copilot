"""增量索引：指纹判定、下架清理、以及几个"静默出错"的场景。

这批用例的价值不在"happy path 能跑通"（那只证明代码没语法错），
而在固定住三类不会报错、只会悄悄错下去的失效：

  1. 权限上调被判成 unchanged —— 答案继续以旧的低密级元数据外流；
  2. 内容变更后旧 chunk 残留 —— 同一份制度的新旧两版同时被引用；
  3. 清单在、向量库空 —— 增量全部跳过，得到一个空索引却报"同步成功"。
"""
from __future__ import annotations

from runtime.config import RagSettings
from runtime.rag.lifecycle import (
    IndexManifest,
    content_hash,
    plan_changes,
    policy_hash,
)
from runtime.rag.pipeline import HybridRAG
from runtime.rag.types import AccessContext, Document

TENANT = "ten_a"


class CountingEmbedder:
    """包一层计数：增量到底省没省，只能看它调了几次。"""

    def __init__(self) -> None:
        from runtime.rag.embeddings import HashingEmbedder

        self._inner = HashingEmbedder(dim=512)
        self.name = "counting"
        self.dim = 512
        self.document_calls = 0
        self.texts_embedded = 0

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        self.texts_embedded += len(texts)
        return await self._inner.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self._inner.embed_query(text)


def build_rag(tmp_path, *, embedder=None) -> HybridRAG:
    settings = RagSettings(
        embedding_provider="hashing",
        embedding_dim=512,
        chunker="recursive",
        chunk_size=200,
        chunk_overlap=20,
        reranker="heuristic",
        vector_backend="memory",
        persist_dir=str(tmp_path),
        # 阈值与 reranker 量纲绑定；测试关心的是"谁可见"，不是分数高低
        score_threshold=0.0,
    )
    return HybridRAG.from_settings(settings, embedder=embedder)


def doc(doc_id: str, text: str, **metadata) -> Document:
    return Document(
        doc_id=doc_id,
        title=metadata.pop("title", doc_id),
        text=text,
        source=f"{doc_id}.md",
        tenant_id=TENANT,
        metadata=metadata,
    )


EMPLOYEE = AccessContext(tenant_id=TENANT, user_id="E1", roles=["employee"], scopes=["rag:read"])


# --------------------------------------------------------------- 指纹语义


def test_content_hash_ignores_policy_and_policy_hash_ignores_text():
    """两个指纹必须正交：否则一个变了会误判另一个，增量要么漏要么贵。"""
    base = doc("d1", "年假为 10 天。", classification="internal")
    same_text_higher = doc("d1", "年假为 10 天。", classification="confidential")
    other_text_same_policy = doc("d1", "年假为 15 天。", classification="internal")

    assert content_hash(base) == content_hash(same_text_higher)
    assert policy_hash(base) != policy_hash(same_text_higher)

    assert content_hash(base) != content_hash(other_text_same_policy)
    assert policy_hash(base) == policy_hash(other_text_same_policy)


def test_policy_hash_is_order_insensitive_for_role_lists():
    """`hr,finance` 与 `finance,hr` 权限等价。按原序哈希会让它白跑一次全量重建。"""
    a = doc("d1", "同一段正文。", allowed_roles="hr,finance")
    b = doc("d1", "同一段正文。", allowed_roles=["finance", "hr"])
    assert policy_hash(a) == policy_hash(b)


def test_plan_marks_policy_change_as_its_own_kind():
    """权限变更不能折叠进 unchanged，也不该混进内容更新 —— 审计要能区分。"""
    original = doc("d1", "同一段正文。", classification="internal")
    manifest = IndexManifest()
    manifest.put("d1", content=content_hash(original), policy=policy_hash(original), chunks=3)

    kinds, removed = plan_changes([doc("d1", "同一段正文。", classification="confidential")], manifest)

    assert kinds["d1"] == "updated_policy"
    assert removed == []


def test_plan_detects_vanished_documents():
    manifest = IndexManifest()
    manifest.put("gone", content="c", policy="p", chunks=1)
    manifest.put("kept", content="c", policy="p", chunks=1)
    kept = doc("kept", "正文。")

    kinds, removed = plan_changes([kept], manifest, force=False)

    assert removed == ["gone"]
    # 指纹对不上（这里用假指纹），所以 kept 会被判为内容更新
    assert kinds["kept"] == "updated_content"


def test_manifest_roundtrip(tmp_path):
    manifest = IndexManifest()
    manifest.put("d1", content="abc", policy="def", chunks=7)
    manifest.save(tmp_path)

    reloaded = IndexManifest.load(tmp_path)

    assert reloaded.docs["d1"].chunks == 7
    assert reloaded.docs["d1"].content_hash == "abc"
    assert reloaded.docs["d1"].policy_hash == "def"


def test_corrupt_manifest_is_treated_as_absent(tmp_path):
    """清单损坏时宁可全量重建，也不要基于一份读不懂的指纹去跳过文档。"""
    (tmp_path / "index_manifest.json").write_text("{ not json", encoding="utf-8")
    assert len(IndexManifest.load(tmp_path)) == 0


# --------------------------------------------------------------- 同步行为


async def test_second_sync_embeds_nothing(tmp_path):
    """增量最核心的收益：没改就不该花一次嵌入。"""
    embedder = CountingEmbedder()
    rag = build_rag(tmp_path, embedder=embedder)
    docs = [doc("d1", "年假为 10 天，需提前三个工作日申请。"), doc("d2", "报销需在 30 天内提交。")]

    first = await rag.sync_documents(docs, prune=True)
    calls_after_first = embedder.document_calls
    texts_after_first = embedder.texts_embedded

    second = await rag.sync_documents(docs, prune=True)

    assert first.added == ["d1", "d2"]
    assert calls_after_first >= 1
    assert second.unchanged == ["d1", "d2"]
    assert second.chunks == 0
    # 关键断言：一次嵌入都没发生
    assert embedder.document_calls == calls_after_first
    assert embedder.texts_embedded == texts_after_first


async def test_content_change_replaces_old_chunks(tmp_path):
    """改一版制度，旧版必须从索引里消失，不能和新版并列被引用。"""
    rag = build_rag(tmp_path)
    await rag.sync_documents([doc("holiday", "年假为 10 天。")], prune=True)

    report = await rag.sync_documents([doc("holiday", "年假为 15 天。")], prune=True)

    assert report.updated_content == ["holiday"]
    stored = await rag.store.all_chunks()
    texts = " ".join(c.text for c in stored)
    assert "15 天" in texts
    assert "10 天" not in texts
    assert len([c for c in stored if c.doc_id == "holiday"]) == 1


async def test_policy_change_is_not_skipped_and_reaches_retrieval(tmp_path):
    """安全用例：只改密级、正文一字不动，也必须重写元数据。

    如果增量只按正文哈希判断，这里会被判 unchanged 而跳过，
    库里留着 internal，普通员工照样能检索到 —— 静默的权限降级。
    """
    rag = build_rag(tmp_path)
    await rag.sync_documents([doc("salary", "调薪窗口在每年 4 月。", classification="internal")], prune=True)

    before = await rag.retrieve("调薪窗口", EMPLOYEE, top_k=5)
    assert before.chunks, "前置条件：internal 密级下员工本应能检索到"

    report = await rag.sync_documents(
        [doc("salary", "调薪窗口在每年 4 月。", classification="confidential")],
        prune=True,
    )

    assert report.updated_policy == ["salary"]
    assert report.unchanged == []
    assert report.chunks > 0  # 确实重写了

    after = await rag.retrieve("调薪窗口", EMPLOYEE, top_k=5)
    assert after.chunks == [], "密级已上调为 confidential，普通员工不应再检索到"

    hr = AccessContext(
        tenant_id=TENANT,
        user_id="E2",
        roles=["hrbp"],
        scopes=["rag:read", "rag:confidential"],
    )
    assert (await rag.retrieve("调薪窗口", hr, top_k=5)).chunks, "有权限的人仍应能检索到"


async def test_prune_removes_vanished_documents(tmp_path):
    """下架的制度如果留在索引里，模型会拿废止条款当依据 —— 比少收录更糟。"""
    rag = build_rag(tmp_path)
    await rag.sync_documents([doc("old", "旧版差旅标准：住宿 400 元。"), doc("keep", "年假 10 天。")], prune=True)

    report = await rag.sync_documents([doc("keep", "年假 10 天。")], prune=True)

    assert report.removed == ["old"]
    stored = await rag.store.all_chunks()
    assert {c.doc_id for c in stored} == {"keep"}
    assert rag.bm25.remove_doc("old") == 0  # 已清干净，再删一次没有受害者


async def test_without_prune_vanished_documents_are_reported_not_deleted(tmp_path):
    """单文件追加是合法用法，不能因为没传 --prune 就顺手清空知识库。"""
    rag = build_rag(tmp_path)
    await rag.sync_documents([doc("a", "甲制度。"), doc("b", "乙制度。")], prune=True)

    report = await rag.sync_documents([doc("a", "甲制度。")], prune=True, dry_run=True)
    assert report.removed == ["b"]  # dry-run 下的含义是"将会删除"
    assert report.pending_removal == []

    real = await rag.sync_documents([doc("a", "甲制度。")], prune=False)
    assert real.removed == []
    assert real.pending_removal == ["b"]
    assert len({c.doc_id for c in await rag.store.all_chunks()}) == 2


async def test_empty_store_with_manifest_forces_full_rebuild(tmp_path):
    """清单说"都没变"、库里却空空如也（重启没落盘 / 换了机器）。

    此时若相信清单，同步会全部跳过，得到一个空索引却报成功 ——
    线上表现为"知识库突然查不到任何东西"，而且日志一片绿。
    """
    embedder = CountingEmbedder()
    rag = build_rag(tmp_path, embedder=embedder)
    docs = [doc("d1", "年假为 10 天。")]
    await rag.sync_documents(docs, prune=True)
    assert len(IndexManifest.load(tmp_path)) == 1

    # 模拟：进程重启，清单还在磁盘，向量库是新的空的
    fresh = build_rag(tmp_path, embedder=CountingEmbedder())
    assert await fresh.store.all_chunks() == []

    report = await fresh.sync_documents(docs, prune=True)

    assert report.forced is True
    assert report.force_reason == "manifest_present_but_store_empty"
    assert report.updated_content == ["d1"]
    assert len(await fresh.store.all_chunks()) == 1


async def test_force_rebuilds_even_when_unchanged(tmp_path):
    rag = build_rag(tmp_path)
    docs = [doc("d1", "年假为 10 天。")]
    await rag.sync_documents(docs, prune=True)

    report = await rag.sync_documents(docs, prune=True, force=True)

    assert report.forced is True
    assert report.force_reason == "caller_forced"
    assert report.updated_content == ["d1"]


async def test_dry_run_leaves_store_untouched(tmp_path):
    rag = build_rag(tmp_path)
    await rag.sync_documents([doc("d1", "年假为 10 天。")], prune=True)
    before = len(await rag.store.all_chunks())

    report = await rag.sync_documents([doc("d1", "年假改为 20 天。"), doc("d2", "新制度。")], prune=True, dry_run=True)

    assert report.added == ["d2"]
    assert report.updated_content == ["d1"]
    assert len(await rag.store.all_chunks()) == before


async def test_documents_without_tenant_never_enter_manifest(tmp_path):
    """没有租户归属的文档不入库，也不能进清单 —— 否则下一轮会被判"消失"而误删同名文档。"""
    rag = build_rag(tmp_path)
    orphan = Document(doc_id="x", title="x", text="无主正文。", source="x.md", tenant_id="")

    report = await rag.sync_documents([orphan], prune=True)

    assert report.skipped == ["x: missing tenant_id"]
    assert len(IndexManifest.load(tmp_path)) == 0


async def test_bm25_and_vector_store_stay_in_sync(tmp_path):
    """两条召回通道各自维护一份 chunk，增量时必须同时增删，否则一边召回废止内容。"""
    rag = build_rag(tmp_path)
    await rag.sync_documents([doc("a", "差旅住宿标准 400 元。")], prune=True)
    await rag.sync_documents([doc("a", "差旅住宿标准 600 元。")], prune=True)

    store_ids = {c.id for c in await rag.store.all_chunks()}
    bm25_ids = {c.id for c in rag.bm25.chunks}

    assert store_ids == bm25_ids
    assert bm25_ids, "不能出现两边都空的假一致"
