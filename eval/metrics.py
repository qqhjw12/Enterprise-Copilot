"""检索评估指标：Recall@k / MRR / nDCG@k / MAP@k / P@k + Bootstrap 置信区间。

为什么不用 ragas 现成指标：
  ragas 的 faithfulness / answer_relevancy 需要 LLM 裁判，离线跑既不稳定也不可复现；
  而检索层的评估**本来就应该确定性**。这里实现的是信息检索领域的标准指标，
  ragas 的 context_precision/context_recall 在"单篇标注"场景下退化为 P@k 与 Recall@k，
  两者可以对齐比较，见 docs/github-benchmark.md。
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence


def first_hit_rank(ranked_docs: Sequence[str], expected: str, k: int = 5) -> int | None:
    for rank, doc in enumerate(ranked_docs[:k], start=1):
        if doc == expected:
            return rank
    return None


def metrics_from_ranks(ranks: Sequence[int | None], k: int = 5) -> dict[str, float]:
    """ranks: 每条查询的首个相关结果排名（1-based），未命中为 None。"""
    total = len(ranks)
    if total == 0:
        return {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "map": 0.0, "precision": 0.0}

    hits = [1.0 if rank and rank <= k else 0.0 for rank in ranks]
    reciprocal = [1.0 / rank if rank and rank <= k else 0.0 for rank in ranks]
    # 单相关文档时 DCG 的理想值就是 1（rank=1 时 1/log2(2)=1）
    ndcg = [1.0 / math.log2(rank + 1) if rank and rank <= k else 0.0 for rank in ranks]
    # 单相关文档时 AP@k = 1/rank；多相关文档时需扩展为逐点平均
    average_precision = reciprocal
    precision = [1.0 / k if rank and rank <= k else 0.0 for rank in ranks]

    return {
        "recall": statistics.fmean(hits),
        "mrr": statistics.fmean(reciprocal),
        "ndcg": statistics.fmean(ndcg),
        "map": statistics.fmean(average_precision),
        "precision": statistics.fmean(precision),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float]:
    """百分位 Bootstrap。固定 seed，保证 CI 在 CI（持续集成）里可复现。"""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistic(draw))
    estimates.sort()
    lo = estimates[int((alpha / 2) * len(estimates))]
    hi = estimates[min(len(estimates) - 1, int((1 - alpha / 2) * len(estimates)))]
    return (lo, hi)


def summarize(ranks: Sequence[int | None], k: int = 5) -> dict[str, float]:
    result = metrics_from_ranks(ranks, k=k)
    hits = [1.0 if rank and rank <= k else 0.0 for rank in ranks]
    lo, hi = bootstrap_ci(hits)
    result["recall_ci_low"] = lo
    result["recall_ci_high"] = hi
    return result
