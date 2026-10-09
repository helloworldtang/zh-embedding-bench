"""构建 T2Retrieval 公开参照测试集。

用法（需先 uv sync --extra data）：
    uv run --extra data python scripts/build_t2retrieval_set.py [--n 500] [--seed 42]

产物：data/t2retrieval_500.jsonl（仓库标准格式，可直接 --data 喂给跑分 CLI）。
原始 parquet 缓存到 data/raw/（约 150MB，已 gitignore）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zh_embedding_bench.t2retrieval import download_raw, select_pairs, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "t2retrieval_500.jsonl"
RAW_CACHE = REPO_ROOT / "data" / "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="从 mteb/T2Retrieval 采样生成公开参照测试集")
    parser.add_argument("--n", type=int, default=500, help="采样 query 条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 JSONL 路径")
    parser.add_argument("--endpoint", default="https://hf-mirror.com", help="HF 端点")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """构建入口。"""
    args = parse_args(argv)
    print(f"下载/加载 T2Retrieval dev split 到 {RAW_CACHE} ...")
    raw = download_raw(RAW_CACHE, endpoint=args.endpoint)
    print(f"queries: {len(raw.queries)}, qrels: {len(raw.qrels)}, corpus: {len(raw.corpus)}")

    pairs = select_pairs(raw, args.n, seed=args.seed)
    write_jsonl(pairs, args.out)
    print(f"已写出 {len(pairs)} 条到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
