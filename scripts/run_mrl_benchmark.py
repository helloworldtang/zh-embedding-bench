"""MRL 截断跑分 CLI：同一模型一次编码，评测各截断维度的指标损失。

用法：
    uv run python scripts/run_mrl_benchmark.py --models qwen3-8b --dims 4096,1024,512,256,64
    uv run python scripts/run_mrl_benchmark.py --data data/t2retrieval_500.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zh_embedding_bench.encoder import encode_corpus, encode_queries, load_model
from zh_embedding_bench.models import MODEL_REGISTRY, get_model_spec
from zh_embedding_bench.mrl import evaluate_mrl_dims

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "qa_pairs.jsonl"
DEFAULT_DIMS = "4096,1024,512,256,64"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="MRL 截断维度评测")
    parser.add_argument(
        "--models",
        default="qwen3-8b",
        help=f"逗号分隔的模型短名（须为 MRL 训练模型才有意义），可选: {', '.join(MODEL_REGISTRY)}",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="测试集 JSONL 路径")
    parser.add_argument("--dims", default=DEFAULT_DIMS, help="逗号分隔的截断维度列表")
    parser.add_argument("--device", default=None, help="推理设备；缺省自动选择")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """MRL 评测主入口。"""
    args = parse_args(argv)
    from zh_embedding_bench.data import load_qa_pairs

    pairs = load_qa_pairs(args.data)
    dims = sorted({int(d) for d in args.dims.split(",") if d.strip()}, reverse=True)
    print(f"测试集: {args.data}（{len(pairs)} 条），截断维度: {dims}")

    for name in [n.strip() for n in args.models.split(",") if n.strip()]:
        spec = get_model_spec(name)
        print(f"\n>>> {spec.name} ({spec.hf_id}) 编码中 ...")
        model = load_model(spec, device=args.device)
        corpus_embeddings = encode_corpus(model, [p.answer for p in pairs], batch_size=spec.encode_batch_size)
        query_embeddings = encode_queries(model, spec, [p.query for p in pairs], batch_size=spec.encode_batch_size)

        results = evaluate_mrl_dims(query_embeddings, corpus_embeddings, dims)
        full_dim = corpus_embeddings.shape[1]
        baseline = results.get(full_dim, {}).get("recall@1")
        print(f"\n| 维度 | recall@1 | recall@5 | mrr | 相对全维{'@1' if baseline else ''} |")
        print("|---|---|---|---|---|")
        for dim in sorted(results, reverse=True):
            m = results[dim]
            delta = f"{(m['recall@1'] - baseline):+.2%}" if baseline is not None and dim != full_dim else "-"
            print(f"| {dim} | {m['recall@1']:.2%} | {m['recall@5']:.2%} | {m['mrr']:.4f} | {delta} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
