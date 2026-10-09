"""从 mteb/T2Retrieval (C-MTEB 检索子任务) 采样生成公开参照测试集。

任务映射：query 原样保留；gold 取该 query 的第一个正例段落；候选池 = 采样后
各 query 的 gold 段落（与自建集同构：gold i 配 query i，其余段落天然构成干扰）。
每 query 有多个正例时只取一个，指标口径偏保守，已在 README 注明。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from zh_embedding_bench.data import QAPair

if TYPE_CHECKING:
    from collections.abc import Sequence

HF_FILES = {
    "queries": "queries/dev-00000-of-00001.parquet",
    "qrels": "data/dev-00000-of-00001.parquet",
    "corpus": "corpus/dev-00000-of-00001.parquet",
}

T2_CATEGORY = "web"

# 段落里的 HTML 噪声标签（<br>/<img> 等）做轻量清理，保留正文标点
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class T2Raw:
    """T2Retrieval dev split 的三张表。

    Attributes:
        queries: query_id 到 query 文本的映射。
        qrels: query_id 到其正例段落 id 列表的映射（本数据集二值标注）。
        corpus: 段落 id 到段落文本的映射。
    """

    queries: dict[int, str]
    qrels: dict[int, list[int]]
    corpus: dict[int, str]


def clean_passage(text: str) -> str:
    """轻量清理段落文本：去 HTML 标签、压空白。

    Args:
        text: 原始段落文本，含 <br>/<img> 等标记。

    Returns:
        清理后的文本。
    """
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()


def select_pairs(raw: T2Raw, n: int, seed: int = 42) -> list[QAPair]:
    """按固定随机种子采样 n 条 query 并配对 gold 段落。

    只保留 corpus 中真实存在正例文本的 query；同一 query 多个正例取第一个。

    Args:
        raw: T2Retrieval dev split 原始数据。
        n: 采样条数；不足时取全部可用 query。
        seed: 随机种子，保证可复现。

    Returns:
        采样后的 QAPair 列表，category 统一为 "web"。

    Raises:
        ValueError: 如果 n 小于 1 或可用 query 为空。
    """
    if n < 1:
        raise ValueError(f"n 必须大于等于 1，收到 {n}")
    usable = sorted(qid for qid, pids in raw.qrels.items() if qid in raw.queries and pids and pids[0] in raw.corpus)
    if not usable:
        raise ValueError("没有可用 query（qrels 与 queries/corpus 对不上）")
    sampled = usable if len(usable) <= n else random.Random(seed).sample(usable, n)
    return [
        QAPair(
            query=raw.queries[qid].strip(),
            answer=clean_passage(raw.corpus[raw.qrels[qid][0]]),
            category=T2_CATEGORY,
        )
        for qid in sampled
    ]


def write_jsonl(pairs: Sequence[QAPair], path: str | Path) -> None:
    """把 QAPair 列表写成仓库标准 JSONL 格式。

    Args:
        pairs: 问答对列表。
        path: 输出文件路径。
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def download_raw(cache_dir: str | Path, endpoint: str = "https://hf-mirror.com") -> T2Raw:
    """下载 T2Retrieval dev split 三张表并装配为 T2Raw。

    用 requests 直连 resolve URL 而非 huggingface_hub：hf-mirror 对单文件
    resolve 请求 308 回源，hub>=0.36 的元数据校验会失败（见 README 第六坑）。

    Args:
        cache_dir: 三张 parquet 的缓存目录（约 150MB，勿入 git）。
        endpoint: HF 端点，默认国内镜像。

    Returns:
        装配好的 T2Raw。

    Raises:
        RuntimeError: 如果下载失败或 parquet 解析失败。
    """
    import pyarrow.parquet as pq  # 延迟导入：仅本函数需要

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames: dict[str, list[dict[str, Any]]] = {}
    for key, rel_path in HF_FILES.items():
        # 三张表 basename 相同（dev-00000-of-00001.parquet），须按 key 区分缓存文件名
        local = cache / f"{key}.parquet"
        if not local.exists():
            url = f"{endpoint}/datasets/mteb/T2Retrieval/resolve/main/{rel_path}"
            response = requests.get(url, timeout=300)
            if response.status_code != 200:
                raise RuntimeError(f"下载失败 {url}: HTTP {response.status_code}")
            local.write_bytes(response.content)
        try:
            frames[key] = pq.read_table(local).to_pylist()
        except Exception as exc:
            raise RuntimeError(f"解析 {local} 失败: {exc}") from exc

    queries = {row["_id"]: row["text"] for row in frames["queries"]}
    corpus = {row["_id"]: row["text"] for row in frames["corpus"]}
    qrels: dict[int, list[int]] = {}
    for row in frames["qrels"]:
        if row["score"] > 0:
            qrels.setdefault(row["query-id"], []).append(row["corpus-id"])
    return T2Raw(queries=queries, qrels=qrels, corpus=corpus)
