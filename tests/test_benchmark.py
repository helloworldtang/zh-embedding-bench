"""环境信息采集与结果表渲染的离线单测（不下载模型）。"""

from zh_embedding_bench.benchmark import ModelResult, collect_env_info, format_results_table
from zh_embedding_bench.models import get_model_spec


class TestCollectEnvInfo:
    def test_contains_all_lib_versions(self) -> None:
        env = collect_env_info()
        assert set(env) == {"python", "torch", "transformers", "sentence_transformers", "numpy"}
        for key, version in env.items():
            assert version, f"{key} 版本为空"


class TestFormatResultsTable:
    def test_success_row_uses_percent_and_mrr(self) -> None:
        result = ModelResult(
            spec=get_model_spec("bge-small"),
            metrics={"recall@1": 0.5, "recall@3": 1.0, "recall@5": 1.0, "recall@10": 1.0, "mrr": 0.75},
            encode_seconds=1.2,
        )
        table = format_results_table([result])
        lines = table.splitlines()
        assert lines[0].startswith("| 模型 | recall@1 |")
        assert "50.00%" in lines[2]
        assert "0.7500" in lines[2]
        assert "1.2" in lines[2]

    def test_failed_row_fills_dashes(self) -> None:
        result = ModelResult(spec=get_model_spec("gte-base"), extra="failed: boom")
        row = format_results_table([result]).splitlines()[2]
        assert row.startswith("| gte-base | failed: boom |")
        assert row.count("-") >= 6
