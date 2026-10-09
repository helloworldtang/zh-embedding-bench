"""sentence-transformers 编码封装。

两个关键点：
1. normalize_embeddings=True——向量归一化到单位长度后，点积即 cosine 相似度；
2. 手动设置 max_seq_length——sentence-transformers 默认 512 会截断长文档（Jina 支持 8192）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from zh_embedding_bench.models import ModelSpec, build_query_text

if TYPE_CHECKING:
    from collections.abc import Sequence

TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(spec: ModelSpec, device: str | None = None) -> SentenceTransformer:
    """按模型描述加载 SentenceTransformer 并设置 max_seq_length。

    Args:
        spec: 模型描述。
        device: 推理设备，如 "mps"/"cpu"；None 交给库自动选择。

    Returns:
        已就绪的模型实例。

    Raises:
        ValueError: 如果 spec.torch_dtype 不是受支持的精度名。
    """
    if spec.torch_dtype and spec.torch_dtype not in TORCH_DTYPES:
        raise ValueError(f"不支持的精度 '{spec.torch_dtype}'，可选: {sorted(TORCH_DTYPES)}")
    model_kwargs: dict[str, object] = {}
    if spec.attn_implementation:
        model_kwargs["attn_implementation"] = spec.attn_implementation
    if spec.torch_dtype:
        model_kwargs["dtype"] = TORCH_DTYPES[spec.torch_dtype]
    # transformers 锁 4.x 后 sentence-transformers 构造函数缺类型标注，须显式标注变量类型
    model: SentenceTransformer = SentenceTransformer(
        spec.hf_id,
        device=device,
        model_kwargs=model_kwargs or None,
        trust_remote_code=spec.trust_remote_code or None,
    )
    model.max_seq_length = spec.max_seq_length
    return model


def encode_corpus(model: SentenceTransformer, corpus: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """编码候选文档，返回归一化后的向量矩阵。

    Args:
        model: 已加载的模型。
        corpus: 候选文档列表。
        batch_size: 编码 batch；max_seq_length 大的模型须调小（显存 ∝ batch × max_seq_length）。

    Returns:
        形状 (len(corpus), dim) 的单位向量矩阵。
    """
    return np.asarray(
        model.encode(list(corpus), normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
    )


def encode_queries(
    model: SentenceTransformer,
    spec: ModelSpec,
    queries: Sequence[str],
    batch_size: int = 32,
) -> np.ndarray:
    """编码 query，自动拼接模型要求的指令前缀（BGE 系列需要）。

    Args:
        model: 已加载的模型。
        spec: 模型描述，用于判断是否需要 query 前缀。
        queries: 原始 query 列表。
        batch_size: 编码 batch。

    Returns:
        形状 (len(queries), dim) 的单位向量矩阵。
    """
    prefixed = [build_query_text(spec, q) for q in queries]
    return np.asarray(model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size))


def rank_queries(query_embeddings: np.ndarray, corpus_embeddings: np.ndarray) -> list[list[int]]:
    """对每条 query 输出候选池按相似度降序的下标排序。

    向量已归一化，点积即 cosine 相似度。

    Args:
        query_embeddings: 形状 (n_queries, dim) 的单位向量矩阵。
        corpus_embeddings: 形状 (n_corpus, dim) 的单位向量矩阵。

    Returns:
        长度 n_queries 的列表，每项是候选池下标按得分从高到低的排列。

    Raises:
        ValueError: 如果两个矩阵维度不一致。
    """
    if query_embeddings.shape[1] != corpus_embeddings.shape[1]:
        raise ValueError(f"向量维度不一致: query {query_embeddings.shape[1]} vs corpus {corpus_embeddings.shape[1]}")
    scores = query_embeddings @ corpus_embeddings.T
    # argsort 降序；用负分升序排保证稳定性
    rankings = np.argsort(-scores, axis=1)
    return [list(row) for row in rankings]
