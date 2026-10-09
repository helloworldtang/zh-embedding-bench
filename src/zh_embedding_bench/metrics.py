"""检索评价指标：recall@k 与 MRR。

纯 numpy 实现，不依赖模型，可离线单测。
"""

from collections.abc import Sequence

import numpy as np

from zh_embedding_bench import DEFAULT_KS


def recall_at_k(ranked_indices: Sequence[int], gold_index: int, k: int) -> float:
    """计算单条 query 的 recall@k。

    Args:
        ranked_indices: 候选池按相似度从高到低排序后的下标序列。
        gold_index: 正确答案在候选池中的下标。
        k: 截断位置，前 k 个结果。

    Returns:
        前 k 个结果包含正确答案返回 1.0，否则 0.0。

    Raises:
        ValueError: 如果 k 小于 1 或 gold_index 不在下标序列覆盖范围内。
    """
    if k < 1:
        raise ValueError(f"k 必须大于等于 1，收到 {k}")
    if gold_index < 0 or gold_index >= len(ranked_indices):
        raise ValueError(f"gold_index {gold_index} 超出候选池范围 [0, {len(ranked_indices)})")
    return 1.0 if gold_index in set(ranked_indices[:k]) else 0.0


def mean_reciprocal_rank(ranked_indices: Sequence[int], gold_index: int) -> float:
    """计算单条 query 的 MRR（倒数排名）。

    Args:
        ranked_indices: 候选池按相似度从高到低排序后的下标序列。
        gold_index: 正确答案在候选池中的下标。

    Returns:
        正确答案排在第 r 位则返回 1/r；不在序列中返回 0.0。

    Raises:
        ValueError: 如果 gold_index 不合法。
    """
    if gold_index < 0 or gold_index >= len(ranked_indices):
        raise ValueError(f"gold_index {gold_index} 超出候选池范围 [0, {len(ranked_indices)})")
    for rank, index in enumerate(ranked_indices, start=1):
        if index == gold_index:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(
    rankings: Sequence[Sequence[int]],
    gold_indices: Sequence[int],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """对整个测试集汇总 recall@k 与 MRR。

    Args:
        rankings: 每条 query 的候选下标排序结果。
        gold_indices: 每条 query 的正确答案下标，与 rankings 一一对应。
        ks: recall@k 的 k 列表。

    Returns:
        形如 {"recall@1": 0.72, "recall@3": 0.83, ..., "mrr": 0.78} 的指标字典。

    Raises:
        ValueError: 如果 rankings 与 gold_indices 长度不一致，或 ks 含非法值。
    """
    if len(rankings) != len(gold_indices):
        raise ValueError(f"rankings({len(rankings)}) 与 gold_indices({len(gold_indices)}) 长度不一致")
    if not rankings:
        raise ValueError("测试集为空，无法计算指标")
    invalid_ks = [k for k in ks if k < 1]
    if invalid_ks:
        raise ValueError(f"ks 含非法值: {invalid_ks}")

    rankings_array = [np.asarray(r, dtype=np.int64) for r in rankings]
    metrics: dict[str, float] = {}
    for k in ks:
        recalls = [recall_at_k(list(r), g, k) for r, g in zip(rankings_array, gold_indices, strict=True)]
        metrics[f"recall@{k}"] = float(np.mean(recalls))
    mrrs = [mean_reciprocal_rank(list(r), g) for r, g in zip(rankings_array, gold_indices, strict=True)]
    metrics["mrr"] = float(np.mean(mrrs))
    return metrics
