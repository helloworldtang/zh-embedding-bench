"""测试集加载与候选池构造的单测。"""

import json
from pathlib import Path

import pytest

from zh_embedding_bench.data import QAPair, build_corpus, load_qa_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "qa.jsonl"
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadQaPairs:
    def test_loads_all_fields_in_order(self, tmp_path) -> None:
        path = write_jsonl(
            tmp_path,
            [
                {"query": "q1", "answer": "a1", "category": "api"},
                {"query": "q2", "answer": "a2", "category": "faq"},
            ],
        )
        pairs = load_qa_pairs(path)
        assert pairs == [
            QAPair(query="q1", answer="a1", category="api"),
            QAPair(query="q2", answer="a2", category="faq"),
        ]

    def test_skips_blank_lines(self, tmp_path) -> None:
        path = write_jsonl(tmp_path, [{"query": "q", "answer": "a", "category": "c"}])
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n\n")
        assert len(load_qa_pairs(path)) == 1

    def test_missing_field_raises(self, tmp_path) -> None:
        path = write_jsonl(tmp_path, [{"query": "q", "answer": "a"}])
        with pytest.raises(ValueError, match="缺少必填字段.*category"):
            load_qa_pairs(path)

    def test_bad_json_raises(self, tmp_path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="第 1 行 JSON 解析失败"):
            load_qa_pairs(bad)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_qa_pairs(tmp_path / "nope.jsonl")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="测试集为空"):
            load_qa_pairs(empty)


class TestBuildCorpus:
    def test_corpus_aligned_with_pairs(self) -> None:
        pairs = [
            QAPair(query="q1", answer="a1", category="c"),
            QAPair(query="q2", answer="a2", category="c"),
        ]
        assert build_corpus(pairs) == ["a1", "a2"]


class TestShippedDataset:
    def test_repo_dataset_loads(self) -> None:
        pairs = load_qa_pairs(REPO_ROOT / "data" / "qa_pairs.jsonl")
        assert len(pairs) == 60
        assert {pair.category for pair in pairs} == {"api", "config", "faq", "doc"}
