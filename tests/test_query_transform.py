"""查询改写与评估指标测试（纯离线，不依赖模型）。"""
from __future__ import annotations

import math

import pytest

from eval.metrics import bootstrap_ci, first_hit_rank, metrics_from_ranks
from runtime.rag.query_transform import expand_queries, lexicon_terms


def test_lexicon_expands_colloquial_to_policy_terms():
    variants = expand_queries("用自己的笔记本连公司内网行不行")

    assert len(variants) == 2
    assert variants[0] == "用自己的笔记本连公司内网行不行"
    assert "VPN" in variants[1]


def test_lexicon_covers_compensation_language():
    terms = lexicon_terms("公司一般什么时候统一涨工资")

    assert "调薪" in terms and "普调" in terms


def test_expansion_is_deduplicated():
    variants = expand_queries("年假有几天")

    assert len(variants) == len(set(variants))


def test_out_of_corpus_query_is_not_expanded():
    """语料外问题绝不能靠反馈词"造"出证据——这是查询漂移的真实事故。"""
    variants = expand_queries("公司有没有太空电梯补贴")

    assert variants == ["公司有没有太空电梯补贴"]


def test_metrics_are_rank_aware():
    stats = metrics_from_ranks([1, 2, 5, None])

    assert stats["recall"] == 0.75
    assert stats["mrr"] == pytest.approx((1 + 0.5 + 0.2) / 4)
    assert stats["ndcg"] == pytest.approx((1 + 1 / math.log2(3) + 1 / math.log2(6)) / 4)
    assert stats["precision"] == pytest.approx(0.15)


def test_bootstrap_ci_is_deterministic_and_bounded():
    values = [1.0] * 40 + [0.0] * 10

    low, high = bootstrap_ci(values, samples=300)

    assert 0.7 <= low <= 1.0
    assert 0.8 <= high <= 1.0
    assert (low, high) == bootstrap_ci(values, samples=300)


def test_first_hit_rank_respects_k():
    docs = ["a", "b", "target"]

    assert first_hit_rank(docs, "target", k=3) == 3
    assert first_hit_rank(docs, "target", k=2) is None
