"""MRL 截断逻辑的离线单测。"""

import numpy as np
import pytest

from zh_embedding_bench.mrl import evaluate_mrl_dims, truncate_and_normalize


class TestTruncateAndNormalize:
    def test_output_shape_and_unit_norm(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.normal(size=(5, 64))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        truncated = truncate_and_normalize(emb, dim=16)
        assert truncated.shape == (5, 16)
        np.testing.assert_allclose(np.linalg.norm(truncated, axis=1), 1.0, atol=1e-9)

    def test_truncation_takes_prefix(self) -> None:
        emb = np.array([[1.0, 2.0, 3.0, 4.0]])
        emb /= np.linalg.norm(emb)
        truncated = truncate_and_normalize(emb, dim=2)
        expected = np.array([[1.0, 2.0]])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(truncated, expected)

    def test_invalid_dim(self) -> None:
        emb = np.ones((2, 8))
        with pytest.raises(ValueError, match="dim 须在"):
            truncate_and_normalize(emb, dim=0)
        with pytest.raises(ValueError, match="dim 须在"):
            truncate_and_normalize(emb, dim=9)


class TestEvaluateMrlDims:
    def test_returns_metrics_per_dim(self) -> None:
        # 构造两个"簇"：query0 与 corpus0 相近，query1 与 corpus1 相近
        rng = np.random.default_rng(7)
        corpus = rng.normal(size=(2, 32))
        queries = corpus + rng.normal(scale=0.05, size=(2, 32))
        for emb in (queries, corpus):
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        results = evaluate_mrl_dims(queries, corpus, dims=[32, 8])
        assert set(results) == {32, 8}
        assert results[32]["recall@1"] == 1.0
        assert results[8]["recall@1"] == 1.0  # 信号足够强，截断后仍应全对

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="向量维度不一致"):
            evaluate_mrl_dims(np.ones((2, 4)), np.ones((3, 8)), dims=[8])
