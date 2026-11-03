"""RAG 检索评估：Recall@5 / MRR / nDCG@5 / MAP@5 / P@5 + 95% Bootstrap CI。

刻意做成"五路对照"，因为只报一个总分没有信息量：
  dense        —— 纯向量
  sparse       —— 纯 BM25
  hybrid       —— RRF 融合（不重排）
  full         —— 完整管线（RRF + Rerank）
  full_mq      —— 完整管线 + RM3-lite 多查询（BM25 查询扩展）

用法：
    python -m eval.run_eval                     # 默认离线（hashing embedding + heuristic rerank）
    RAG_EMBEDDING_PROVIDER=sentence-transformers RAG_RERANKER=cross-encoder python -m eval.run_eval
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from eval.metrics import first_hit_rank, summarize
from runtime.config import RagSettings, get_rag_settings
from runtime.rag import access
from runtime.rag.fusion import rrf_fuse
from runtime.rag.pipeline import HybridRAG
from runtime.rag.types import AccessContext, Chunk

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "dataset.jsonl"
OUTPUT = ROOT / "var" / "eval_report.json"


@dataclass(slots=True)
class QueryCase:
    id: str
    query: str
    expected: str
    roles: str = "employee"
    style: str = "keyword"  # keyword | paraphrase


@dataclass(slots=True)
class ModeResult:
    name: str
    recall_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    map_at_5: float = 0.0
    precision_at_5: float = 0.0
    recall_ci: tuple[float, float] = (0.0, 0.0)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    subset_recall: dict[str, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)


def load_dataset(path: Path = DATASET) -> list[QueryCase]:
    cases: list[QueryCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        cases.append(QueryCase(**raw))
    return cases


def doc_key(chunk: Chunk) -> str:
    return Path(str(chunk.metadata.get("source", ""))).stem


async def run_mode(rag: HybridRAG, cases: list[QueryCase], mode: str) -> ModeResult:
    original_settings = rag.settings
    settings = original_settings
    if mode == "full_mq":
        rag.settings = original_settings.model_copy(update={"query_transform": "multi_query"})

    latencies: list[float] = []
    ranks: list[int | None] = []
    subset_ranks: dict[str, list[int | None]] = {"keyword": [], "paraphrase": []}
    per_query: list[dict] = []

    try:
        for case in cases:
            ctx = AccessContext(
                tenant_id="ten_eval",
                user_id="E1024",
                roles=[case.roles],
                scopes=["rag:read"] + (["rag:confidential"] if case.roles == "hr" else []),
            )
            predicate = access.make_predicate(ctx)
            started = time.perf_counter()

            if mode == "dense":
                vector = await rag.embedder.embed_query(case.query)
                hits_raw = await rag.store.search(vector, top_k=5, predicate=predicate)
                chunks = [c for c, _ in hits_raw]
            elif mode == "sparse":
                hits_raw = rag.bm25.search(case.query, top_k=5, predicate=predicate)
                chunks = [c for c, _ in hits_raw]
            elif mode == "hybrid":
                vector = await rag.embedder.embed_query(case.query)
                dense = [c for c, _ in await rag.store.search(vector, top_k=settings.dense_top_k, predicate=predicate)]
                sparse = [c for c, _ in rag.bm25.search(case.query, top_k=settings.sparse_top_k, predicate=predicate)]
                chunks = rrf_fuse({"dense": dense, "sparse": sparse}, k=settings.rrf_k, limit=5).chunks
            else:  # full / full_mq
                result = await rag.retrieve(case.query, ctx, top_k=5, trace=False)
                chunks = result.chunks

            latencies.append((time.perf_counter() - started) * 1000)
            rank = first_hit_rank([doc_key(c) for c in chunks], case.expected)
            ranks.append(rank)
            subset_ranks.setdefault(case.style, []).append(rank)
            per_query.append(
                {
                    "id": case.id,
                    "query": case.query,
                    "expected": case.expected,
                    "hit": rank is not None,
                    "rank": rank,
                    "top1": doc_key(chunks[0]) if chunks else None,
                }
            )
    finally:
        rag.settings = original_settings

    stats = summarize(ranks, k=5)
    latencies.sort()
    return ModeResult(
        name=mode,
        recall_at_5=stats["recall"],
        mrr=stats["mrr"],
        ndcg_at_5=stats["ndcg"],
        map_at_5=stats["map"],
        precision_at_5=stats["precision"],
        recall_ci=(stats["recall_ci_low"], stats["recall_ci_high"]),
        p50_ms=statistics.median(latencies),
        p95_ms=latencies[max(0, int(0.95 * len(latencies)) - 1)],
        subset_recall={k: (sum(1 for r in v if r) / len(v) if v else 0.0) for k, v in subset_ranks.items()},
        per_query=per_query,
    )


def _settings_for_eval(chunker: str) -> RagSettings:
    base = get_rag_settings()
    return base.model_copy(update={"chunker": chunker, "vector_backend": "memory"})


async def main_async(args: argparse.Namespace) -> int:
    cases = load_dataset()
    settings = _settings_for_eval(args.chunker)
    rag = HybridRAG.from_settings(settings)
    started = time.perf_counter()
    stats = await rag.index_directory(ROOT / "data" / "knowledge", tenant_id="ten_eval")
    index_ms = (time.perf_counter() - started) * 1000

    print(
        f"语料：{stats.documents} 篇 / {stats.chunks} chunks"
        f"（chunker={rag.chunker.name}, embedding={getattr(rag.embedder, 'name', '?')}, "
        f"reranker={rag.reranker.name}，索引耗时 {index_ms:.0f} ms）"
    )
    print(f"评估集：{len(cases)} 条\n")

    results: list[ModeResult] = []
    for mode in ("dense", "sparse", "hybrid", "full", "full_mq"):
        results.append(await run_mode(rag, cases, mode))

    print(f"{'管线':<16}{'Recall@5':>9}{'95% CI':>14}{'MRR':>8}{'nDCG@5':>9}{'P@5':>8}{'关键词':>8}{'改写':>8}{'p50(ms)':>9}")
    print("-" * 97)
    label = {"dense": "Dense", "sparse": "BM25", "hybrid": "RRF", "full": "RRF+Rerank", "full_mq": "RRF+Rerank+MQ"}
    for result in results:
        sub = result.subset_recall
        ci = f"[{result.recall_ci[0]:.3f},{result.recall_ci[1]:.3f}]"
        print(
            f"{label[result.name]:<18}{result.recall_at_5:>7.3f}{ci:>15}{result.mrr:>8.3f}"
            f"{result.ndcg_at_5:>9.3f}{result.precision_at_5:>8.3f}"
            f"{sub.get('keyword', 0):>8.3f}{sub.get('paraphrase', 0):>8.3f}{result.p50_ms:>9.2f}"
        )

    misses = [q for q in results[-1].per_query if not q["hit"]]
    if misses:
        print("\n未命中：")
        for item in misses:
            print(f"  {item['id']} {item['query']} → 期望 {item['expected']}，实际 top1 {item['top1']}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "corpus": {
                    "documents": stats.documents,
                    "chunks": stats.chunks,
                    "index_ms": round(index_ms, 2),
                    "chunker": rag.chunker.name,
                    "embedding": getattr(rag.embedder, "name", "?"),
                    "reranker": rag.reranker.name,
                },
                "dataset_size": len(cases),
                "k": 5,
                "modes": {
                    r.name: {
                        "recall_at_5": round(r.recall_at_5, 4),
                        "recall_ci95": [round(r.recall_ci[0], 4), round(r.recall_ci[1], 4)],
                        "mrr": round(r.mrr, 4),
                        "ndcg_at_5": round(r.ndcg_at_5, 4),
                        "map_at_5": round(r.map_at_5, 4),
                        "precision_at_5": round(r.precision_at_5, 4),
                        "subset_recall": {k: round(v, 4) for k, v in r.subset_recall.items()},
                        "p50_ms": round(r.p50_ms, 2),
                        "p95_ms": round(r.p95_ms, 2),
                    }
                    for r in results
                },
                "per_query": results[-1].per_query,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n明细已写入 {OUTPUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评估")
    parser.add_argument("--chunker", default="recursive", choices=["recursive", "semantic"])
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
