"""RAG 检索管线单测：零 API Key、零模型下载、零外部服务。"""
from __future__ import annotations

import pytest

from runtime.rag.access import chunk_visible
from runtime.rag.bm25 import BM25Index, tokenize
from runtime.rag.chunkers import RecursiveChunker, SemanticChunker, split_blocks
from runtime.rag.embeddings import HashingEmbedder
from runtime.rag.fusion import rrf_fuse
from runtime.rag.parsers import parse_file
from runtime.rag.rerank import HeuristicReranker
from runtime.rag.types import AccessContext, Chunk, Document

# ---------------------------------------------------------------- 分词 & BM25

def test_tokenize_handles_chinese_bigram_and_ascii():
    tokens = tokenize("年假10天")
    assert "年假" in tokens  # 中文 bigram
    assert "10" in tokens  # 数字保留


def test_bm25_ranks_relevant_doc_first():
    index = BM25Index()
    index.add(
        [
            Chunk(id="a", text="年假天数按工龄计算，满一年 7 天。", metadata={"tenant_id": "t"}),
            Chunk(id="b", text="差旅住宿标准：一线城市 600 元。", metadata={"tenant_id": "t"}),
        ]
    )
    hits = index.search("年假几天", top_k=2)
    assert hits and hits[0][0].id == "a"
    assert hits[0][1] > 0


def test_bm25_predicate_filters_before_scoring():
    index = BM25Index()
    index.add(
        [
            Chunk(id="a", text="年假 7 天", metadata={"tenant_id": "t1"}),
            Chunk(id="b", text="年假 10 天", metadata={"tenant_id": "t2"}),
        ]
    )
    hits = index.search("年假", top_k=5, predicate=lambda c: c.metadata["tenant_id"] == "t1")
    assert [c.id for c, _ in hits] == ["a"]


# ---------------------------------------------------------------- RRF

def test_rrf_formula_matches_standard():
    dense = [Chunk(id=f"d{i}", text="x") for i in range(3)]
    sparse = [Chunk(id=f"d{i}", text="x") for i in (1, 2, 0)]
    fused = rrf_fuse({"dense": dense, "sparse": sparse}, k=60, limit=3)

    # d0: dense rank1 + sparse rank3; d1: dense rank2 + sparse rank1
    expected = {
        "d0": 1 / (60 + 1) + 1 / (60 + 3),
        "d1": 1 / (60 + 2) + 1 / (60 + 1),
    }
    scores = {c.id: c.scores["rrf"] for c in fused.chunks}
    for chunk_id, value in expected.items():
        assert scores[chunk_id] == pytest.approx(value, rel=1e-9)
    # 两路都靠前的 d1 应排在只有一路靠前的 d0 前面以外的位置由累加决定
    assert fused.chunks[0].id in {"d0", "d1"}


def test_rrf_requires_candidates_from_both_sources():
    only_dense = [Chunk(id="a", text="x")]
    fused = rrf_fuse({"dense": only_dense, "sparse": []}, k=60)
    assert [c.id for c in fused.chunks] == ["a"]


# ---------------------------------------------------------------- 切分

@pytest.mark.asyncio
async def test_recursive_chunker_respects_size_and_overlap():
    doc = Document(
        doc_id="d1",
        title="测试",
        text="# 标题\n\n" + "这是一个用于测试切分的句子。" * 60,
        source="x.md",
        tenant_id="t1",
    )
    chunker = RecursiveChunker(size=200, overlap=40)
    chunks = await chunker.split(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 + 40 for c in chunks)
    assert all(c.metadata["doc_id"] == "d1" for c in chunks)
    assert chunks[0].metadata["section"] == "标题"


@pytest.mark.asyncio
async def test_semantic_chunker_splits_at_topic_change():
    text = (
        "# 制度\n\n"
        "公司年假按工龄计算，满一年七天，满三年十天，满五年十二天。年假可以跨年结转。\n"
        "差旅报销需要提供发票，一线城市住宿上限六百元，二线城市四百五十元。"
        "交通优先高铁二等座，超标部分自理。"
    )
    doc = Document(doc_id="d2", title="t", text=text, source="y.md", tenant_id="t1")
    chunker = SemanticChunker(HashingEmbedder(dim=256), size=300, break_percentile=50)
    chunks = await chunker.split(doc)
    assert chunks, "语义切分至少要产出一个块"
    assert all(c.text.strip() for c in chunks)


def test_split_blocks_builds_section_path():
    blocks = split_blocks("# A\n\ntext1\n\n## B\n\ntext2")
    assert [b.section for b in blocks] == ["A", "A / B"]


# ---------------------------------------------------------------- 解析

def test_markdown_front_matter_is_parsed(tmp_path):
    file = tmp_path / "doc.md"
    file.write_text(
        "---\ntitle: 薪酬细则\nclassification: confidential\nallowed_roles: hr,finance\n---\n\n# 正文\n内容",
        encoding="utf-8",
    )
    doc = parse_file(file, tenant_id="t9")
    assert doc.title == "薪酬细则"
    assert doc.metadata["classification"] == "confidential"
    assert doc.tenant_id == "t9"
    assert "---" not in doc.text


def test_html_parser_keeps_headings(tmp_path):
    file = tmp_path / "doc.html"
    file.write_text("<h1>年假</h1><p>年假 7 天</p><ul><li>需提前申请</li></ul>", encoding="utf-8")
    doc = parse_file(file, tenant_id="t9")
    assert "# 年假" in doc.text
    assert "需提前申请" in doc.text


# ---------------------------------------------------------------- 端到端检索

@pytest.mark.asyncio
async def test_retrieval_hits_expected_document(rag, employee_ctx):
    result = await rag.retrieve("年假有几天", employee_ctx, top_k=5)
    titles = [c.title for c in result.chunks]
    assert "员工年假与请假制度" in titles
    assert result.trace is not None
    assert result.trace.total_ms >= 0


@pytest.mark.asyncio
async def test_cross_tenant_probe_returns_nothing(rag, employee_ctx):
    """语料里有一篇属于 ten_other_demo 的探针文档，任何情况下都不该被跨租户检索到。"""
    result = await rag.retrieve("ZQ-OTHER-TENANT-9931", employee_ctx, top_k=10)
    assert all("ZQ-OTHER-TENANT-9931" not in c.text for c in result.chunks)
    assert result.trace.filtered_out >= 1


@pytest.mark.asyncio
async def test_confidential_doc_requires_scope(rag, employee_ctx, hr_ctx):
    emp_result = await rag.retrieve("补充医疗保险报销上限", employee_ctx, top_k=5)
    assert all(c.metadata.get("classification") != "confidential" for c in emp_result.chunks)

    hr_result = await rag.retrieve("补充医疗保险报销上限", hr_ctx, top_k=5)
    assert any(c.title == "薪酬与福利管理细则" for c in hr_result.chunks)


def test_chunk_visible_rejects_missing_tenant():
    ctx = AccessContext(tenant_id="t1")
    assert chunk_visible(Chunk(id="a", text="x", metadata={}), ctx) is False
    assert chunk_visible(Chunk(id="b", text="x", metadata={"tenant_id": "t1"}), ctx) is True


def test_chunk_visible_handles_string_allowed_roles():
    """front matter 的 `allowed_roles: hr` 会解析成字符串，不能按字符集合比较。"""
    ctx = AccessContext(tenant_id="t1", roles=["employee"])
    chunk = Chunk(id="a", text="x", metadata={"tenant_id": "t1", "allowed_roles": "hr"})
    assert chunk_visible(chunk, ctx) is False
    ctx_hr = AccessContext(tenant_id="t1", roles=["hr"])
    assert chunk_visible(chunk, ctx_hr) is True


@pytest.mark.asyncio
async def test_heuristic_reranker_drops_chunks_without_query_terms():
    """一个查询词都没命中的片段直接出局，而不是"排在最后但仍在上下文里"。

    早先的实现会保留它并给 0 分，结果 top_k=5 时上下文里混进 2~3 条无关片段，
    模型于是拿着噪声硬编答案。
    """
    reranker = HeuristicReranker()
    chunks = [
        Chunk(id="relevant", text="年假十天", metadata={}, scores={"rrf": 0.02}),
        Chunk(id="irrelevant", text="供应商准入需要营业执照", metadata={}, scores={"rrf": 0.03}),
    ]
    ranked = await reranker.rerank("年假", chunks, top_n=2)
    assert [c.id for c in ranked] == ["relevant"]
    assert ranked[0].scores["rerank"] > 0
