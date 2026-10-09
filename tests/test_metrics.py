"""recall@k 与 MRR 的手算样例单测。"""

import pytest

from zh_embedding_bench.metrics import evaluate_rankings, mean_reciprocal_rank, recall_at_k


class TestRecallAtK:
    def test_gold_in_top_k(self) -> None:
        # gold=3 排在首位之后第 2 位：k=1 未命中，k=2 命中
        assert recall_at_k([5, 3, 1, 0], gold_index=3, k=1) == 0.0
        assert recall_at_k([5, 3, 1, 0], gold_index=3, k=2) == 1.0
        assert recall_at_k([5, 3, 1, 0], gold_index=3, k=10) == 1.0

    def test_gold_at_first_position(self) -> None:
        assert recall_at_k([2, 0, 1], gold_index=2, k=1) == 1.0

    def test_gold_not_in_ranking(self) -> None:
        assert recall_at_k([0, 1, 2], gold_index=2, k=1) == 0.0

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError, match="k 必须"):
            recall_at_k([0, 1], gold_index=0, k=0)

    def test_invalid_gold_index(self) -> None:
        with pytest.raises(ValueError, match="超出候选池范围"):
            recall_at_k([0, 1], gold_index=5, k=1)


class TestMRR:
    def test_rank_one(self) -> None:
        assert mean_reciprocal_rank([2, 0, 1], gold_index=2) == 1.0

    def test_rank_three(self) -> None:
        assert mean_reciprocal_rank([2, 1, 4, 0, 3], gold_index=4) == pytest.approx(1 / 3)

    def test_gold_missing(self) -> None:
        # gold 不在排序序列中得 0 分
        assert mean_reciprocal_rank([0, 1], gold_index=0) == 1.0


class TestEvaluateRankings:
    def test_aggregation(self) -> None:
        # 三条 query：gold 分别排第 1、2、5 位
        rankings = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [3, 2, 1, 4, 0]]
        golds = [0, 0, 0]
        metrics = evaluate_rankings(rankings, golds, ks=(1, 3, 5))
        assert metrics["recall@1"] == pytest.approx(1 / 3)
        assert metrics["recall@3"] == pytest.approx(2 / 3)
        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 0.2) / 3)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="长度不一致"):
            evaluate_rankings([[0, 1]], [0, 1])

    def test_empty_dataset(self) -> None:
        with pytest.raises(ValueError, match="测试集为空"):
            evaluate_rankings([], [])
