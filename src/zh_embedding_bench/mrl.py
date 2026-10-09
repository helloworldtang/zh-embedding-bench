"""MRL（Matryoshka Representation Learning）截断评测。

MRL 训练的 embedding（如 Qwen3-Embedding 全系、OpenAI text-embedding-3）支持
只取向量前 N 维作为低维 embedding——已存向量截断即可降维，无需全库重新编码。
本模块评测截断到各维度后的检索指标损失。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zh_embedding_bench.encoder import rank_queries
from zh_embedding_bench.metrics import evaluate_rankings

if TYPE_CHECKING:
    from collections.abc import Sequence


def truncate_and_normalize(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """截取向量前 dim 维并重新归一化到单位长度。

    截断后向量范数变小，必须重归一化，否则点积不再是 cosine 相似度。

    Args:
        embeddings: 形状 (n, full_dim) 的单位向量矩阵。
        dim: 目标维度，1 <= dim <= full_dim。

    Returns:
        形状 (n, dim) 的单位向量矩阵。

    Raises:
        ValueError: 如果 dim 超出范围。
    """
    full_dim = embeddings.shape[1]
    if not 1 <= dim <= full_dim:
        raise ValueError(f"dim 须在 [1, {full_dim}]，收到 {dim}")
    truncated = embeddings[:, :dim].astype(np.float64)
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / np.maximum(norms, 1e-12)


def evaluate_mrl_dims(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    dims: Sequence[int],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[int, dict[str, float]]:
    """对已编码的向量评测各截断维度的检索指标。

    Args:
        query_embeddings: 形状 (n_queries, full_dim) 的单位向量矩阵。
        corpus_embeddings: 形状 (n_corpus, full_dim) 的单位向量矩阵。
        dims: 要评测的维度列表，通常含 full_dim 作为基线。
        ks: recall@k 的 k 列表。

    Returns:
        维度到指标字典的映射，如 {4096: {"recall@1": ...}, 1024: {...}}。

    Raises:
        ValueError: 如果两个矩阵维度不一致。
    """
    if query_embeddings.shape[1] != corpus_embeddings.shape[1]:
        raise ValueError(f"向量维度不一致: query {query_embeddings.shape[1]} vs corpus {corpus_embeddings.shape[1]}")
    gold_indices = list(range(corpus_embeddings.shape[0]))
    results: dict[int, dict[str, float]] = {}
    for dim in dims:
        truncated_queries = truncate_and_normalize(query_embeddings, dim)
        truncated_corpus = truncate_and_normalize(corpus_embeddings, dim)
        rankings = rank_queries(truncated_queries, truncated_corpus)
        results[dim] = evaluate_rankings(rankings, gold_indices, ks=ks)
    return results
