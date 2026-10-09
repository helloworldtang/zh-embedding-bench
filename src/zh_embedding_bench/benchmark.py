"""跑分主流程：逐模型编码、排序、计算指标、汇总输出。

用法见 scripts/run_benchmark.py；也可作为库调用 run_benchmark()。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import sentence_transformers
import torch
import transformers

from zh_embedding_bench import DEFAULT_KS
from zh_embedding_bench.encoder import encode_corpus, encode_queries, load_model, rank_queries
from zh_embedding_bench.metrics import evaluate_rankings
from zh_embedding_bench.models import ModelSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zh_embedding_bench.data import QAPair


def collect_env_info() -> dict[str, str]:
    """采集当前跑分环境的库版本。

    跑分有效性不要求各模型环境完全一致（同硬件/同数据/同指标实现即可），
    但库版本必须随结果落盘，跨环境对比耗时与指标时才有据可查。

    Returns:
        形如 {"python": "3.14", "torch": "...", ...} 的版本字典。
    """
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__,
    }


@dataclass
class ModelResult:
    """单个模型的跑分结果。

    Attributes:
        spec: 被测模型描述。
        metrics: 指标字典，如 {"recall@1": 0.72, ..., "mrr": 0.78}。
        encode_seconds: 编码全部 query + corpus 的耗时（秒）。
        extra: 附加信息（如失败原因）。
    """

    spec: ModelSpec
    metrics: dict[str, float] = field(default_factory=dict)
    encode_seconds: float = 0.0
    extra: str = ""


def benchmark_model(
    spec: ModelSpec,
    pairs: Sequence[QAPair],
    ks: Sequence[int] = DEFAULT_KS,
    device: str | None = None,
) -> ModelResult:
    """跑单个模型：加载、编码、排序、算指标。

    Args:
        spec: 模型描述。
        pairs: 问答测试集。
        ks: recall@k 的 k 列表。
        device: 推理设备；None 自动选择。

    Returns:
        含指标与耗时的 ModelResult。
    """
    model = load_model(spec, device=device)
    corpus = [pair.answer for pair in pairs]
    queries = [pair.query for pair in pairs]

    start = time.perf_counter()
    corpus_embeddings = encode_corpus(model, corpus, batch_size=spec.encode_batch_size)
    query_embeddings = encode_queries(model, spec, queries, batch_size=spec.encode_batch_size)
    encode_seconds = time.perf_counter() - start

    rankings = rank_queries(query_embeddings, corpus_embeddings)
    gold_indices = list(range(len(pairs)))
    metrics = evaluate_rankings(rankings, gold_indices, ks=ks)
    return ModelResult(spec=spec, metrics=metrics, encode_seconds=encode_seconds)


def run_benchmark(
    specs: Sequence[ModelSpec],
    pairs: Sequence[QAPair],
    ks: Sequence[int] = DEFAULT_KS,
    device: str | None = None,
) -> list[ModelResult]:
    """顺序跑多个模型并汇总。

    单个模型加载或编码失败不中断整体，失败以 extra 字段记录。

    Args:
        specs: 模型描述列表。
        pairs: 问答测试集。
        ks: recall@k 的 k 列表。
        device: 推理设备；None 自动选择。

    Returns:
        与 specs 顺序一致的 ModelResult 列表。
    """
    results: list[ModelResult] = []
    for spec in specs:
        print(f"\n>>> 跑分 {spec.name} ({spec.hf_id}) ...")
        try:
            results.append(benchmark_model(spec, pairs, ks=ks, device=device))
        except Exception as exc:  # 跑分允许多模型部分失败，不中断整体
            print(f"!!! {spec.name} 跑分失败: {exc}")
            results.append(ModelResult(spec=spec, extra=f"failed: {exc}"))
    return results


def format_results_table(results: Sequence[ModelResult]) -> str:
    """把跑分结果渲染成 Markdown 表格。

    Args:
        results: ModelResult 列表。

    Returns:
        Markdown 表格字符串，列：模型 / recall@1,3,5,10 / MRR / 编码耗时。
    """
    metric_keys = [f"recall@{k}" for k in (1, 3, 5, 10)] + ["mrr"]
    header = "| 模型 | " + " | ".join(metric_keys) + " | 编码耗时(s) |"
    separator = "|" + "---|" * (len(metric_keys) + 2)
    lines = [header, separator]
    for result in results:
        if result.extra:
            cells = [result.spec.name, result.extra] + ["-"] * len(metric_keys)
            lines.append("| " + " | ".join(cells) + " | - |")
            continue
        metric_cells = [
            f"{result.metrics[key]:.2%}" if key != "mrr" else f"{result.metrics[key]:.4f}" for key in metric_keys
        ]
        lines.append(f"| {result.spec.name} | " + " | ".join(metric_cells) + f" | {result.encode_seconds:.1f} |")
    return "\n".join(lines)
