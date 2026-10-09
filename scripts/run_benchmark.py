"""跑分 CLI 入口。

用法：
    uv run python scripts/run_benchmark.py                          # 全部四个模型
    uv run python scripts/run_benchmark.py --models bge-large,gte-base
    uv run python scripts/run_benchmark.py --data data/qa_pairs.jsonl --device cpu

结果打印到控制台，并写入 results/ 目录（Markdown + JSON）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from zh_embedding_bench import DEFAULT_KS
from zh_embedding_bench.benchmark import collect_env_info, format_results_table, run_benchmark
from zh_embedding_bench.data import load_qa_pairs
from zh_embedding_bench.models import MODEL_REGISTRY, get_model_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "qa_pairs.jsonl"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="中文 embedding 模型检索跑分")
    parser.add_argument(
        "--models",
        default=",".join(MODEL_REGISTRY),
        help=f"逗号分隔的模型短名，可选: {', '.join(MODEL_REGISTRY)}",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="测试集 JSONL 路径")
    parser.add_argument("--device", default=None, help="推理设备，如 mps/cpu；缺省自动选择")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """跑分主入口。

    Returns:
        进程退出码：全部成功 0；存在失败模型 1。
    """
    args = parse_args(argv)
    specs = [get_model_spec(name.strip()) for name in args.models.split(",") if name.strip()]

    pairs = load_qa_pairs(args.data)
    print(f"测试集: {args.data}（{len(pairs)} 条）")
    print(f"待测模型: {[spec.name for spec in specs]}")

    results = run_benchmark(specs, pairs, ks=DEFAULT_KS, device=args.device)

    table = format_results_table(results)
    print("\n" + table + "\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_info = collect_env_info()
    env_line = " / ".join(f"{k} {v}" for k, v in env_info.items())
    md_path = RESULTS_DIR / f"benchmark_{stamp}.md"
    md_path.write_text(
        f"# 跑分结果 {stamp}\n\n测试集: {args.data}（{len(pairs)} 条）\n\n{table}\n\n环境: {env_line}\n",
        encoding="utf-8",
    )
    json_path = RESULTS_DIR / f"benchmark_{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "env": env_info,
                "models": [
                    {"model": r.spec.name, "hf_id": r.spec.hf_id, "metrics": r.metrics, "extra": r.extra}
                    for r in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果已写入: {md_path.name}, {json_path.name}")

    return 1 if any(r.extra for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
