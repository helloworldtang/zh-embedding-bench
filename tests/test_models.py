"""模型注册表与 BGE 查询前缀逻辑的单测（不下载模型）。"""

import pytest

from zh_embedding_bench.models import (
    BGE_QUERY_PREFIX,
    MODEL_REGISTRY,
    build_query_text,
    get_model_spec,
)


class TestRegistry:
    def test_contains_four_models(self) -> None:
        assert set(MODEL_REGISTRY) == {"bge-large", "bge-small", "gte-base", "jina-base"}

    def test_dimensions(self) -> None:
        assert get_model_spec("bge-large").dim == 1024
        assert get_model_spec("bge-small").dim == 512
        assert get_model_spec("gte-base").dim == 768
        assert get_model_spec("jina-base").dim == 768

    def test_unknown_model_lists_alternatives(self) -> None:
        with pytest.raises(KeyError, match="bge-large"):
            get_model_spec("nope")


class TestQueryPrefix:
    def test_bge_models_have_prefix(self) -> None:
        for name in ("bge-large", "bge-small"):
            assert get_model_spec(name).query_prefix == BGE_QUERY_PREFIX

    def test_non_bge_models_have_no_prefix(self) -> None:
        for name in ("gte-base", "jina-base"):
            assert get_model_spec(name).query_prefix == ""

    def test_prefix_applied_to_query_only_semantics(self) -> None:
        spec = get_model_spec("bge-large")
        assert build_query_text(spec, "rate limit") == BGE_QUERY_PREFIX + "rate limit"

    def test_no_prefix_passthrough(self) -> None:
        spec = get_model_spec("gte-base")
        assert build_query_text(spec, "rate limit") == "rate limit"


class TestMaxSeqLength:
    def test_jina_supports_long_context(self) -> None:
        assert get_model_spec("jina-base").max_seq_length == 8192

    def test_others_default_512(self) -> None:
        for name in ("bge-large", "bge-small", "gte-base"):
            assert get_model_spec(name).max_seq_length == 512


class TestJinaCompatFlags:
    def test_jina_requires_eager_attention_and_remote_code(self) -> None:
        spec = get_model_spec("jina-base")
        assert spec.attn_implementation == "eager"
        assert spec.trust_remote_code is True

    def test_others_need_no_compat_flags(self) -> None:
        for name in ("bge-large", "bge-small", "gte-base"):
            spec = get_model_spec(name)
            assert spec.attn_implementation == ""
            assert spec.trust_remote_code is False
