"""T2Retrieval 采样逻辑的离线单测（不下载）。"""

from pathlib import Path

import pytest

from zh_embedding_bench.data import load_qa_pairs
from zh_embedding_bench.t2retrieval import T2Raw, clean_passage, select_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_raw() -> T2Raw:
    queries = {10: "怎么重置密码", 11: "发票怎么开", 12: "孤儿query没有正例", 13: "悬空正例"}
    qrels = {
        10: [100, 101],  # 多正例取第一个
        12: [],  # 无正例，应被过滤
        13: [999],  # 正例不在 corpus，应被过滤
    }
    corpus = {
        100: "<br>进入设置<br>点击<b>重置密码</b>",
        101: "另一种重置方式",
        200: "发票相关段落",
    }
    return T2Raw(queries=queries, qrels=qrels, corpus=corpus)


class TestCleanPassage:
    def test_strips_html_tags_and_whitespace(self) -> None:
        assert clean_passage("<br><img><br>1、点击屏幕") == "1、点击屏幕"

    def test_keeps_plain_text(self) -> None:
        assert clean_passage("普通文本，含标点。") == "普通文本，含标点。"


class TestSelectPairs:
    def test_filters_unusable_queries_and_takes_first_positive(self) -> None:
        pairs = select_pairs(make_raw(), n=10)
        assert len(pairs) == 1
        assert pairs[0].query == "怎么重置密码"
        assert pairs[0].answer == "进入设置 点击 重置密码"
        assert pairs[0].category == "web"

    def test_sampling_reproducible(self) -> None:
        # 构造足量可用 query 验证固定种子可复现
        raw = make_raw()
        raw.queries.update({i: f"问题{i}" for i in range(100)})
        raw.qrels.update({i: [200] for i in range(100)})
        first = select_pairs(raw, n=5, seed=42)
        second = select_pairs(raw, n=5, seed=42)
        assert first == second

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n 必须"):
            select_pairs(make_raw(), n=0)

    def test_empty_dataset_raises(self) -> None:
        empty = T2Raw(queries={}, qrels={}, corpus={})
        with pytest.raises(ValueError, match="没有可用 query"):
            select_pairs(empty, n=5)


class TestShippedT2Set:
    def test_t2_set_loads_if_present(self) -> None:
        path = REPO_ROOT / "data" / "t2retrieval_500.jsonl"
        if not path.exists():
            pytest.skip("t2retrieval_500.jsonl 尚未生成（运行 scripts/build_t2retrieval_set.py）")
        pairs = load_qa_pairs(path)
        assert len(pairs) == 500
        assert all(pair.category == "web" for pair in pairs)
        assert all("<br>" not in pair.answer for pair in pairs)
