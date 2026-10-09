"""embedding 模型注册表与查询前缀处理。

覆盖文章实测的四个模型：BGE-large-zh-v1.5 / BGE-small-zh-v1.5 / gte-base-zh / jina-embeddings-v2-base-zh。
"""

from dataclasses import dataclass

BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

QWEN_QUERY_PREFIX = "Instruct: 给定一个中文技术文档问题，检索出最相关的答案段落\nQuery: "


@dataclass(frozen=True)
class ModelSpec:
    """单个 embedding 模型的静态描述。

    Attributes:
        name: 命令行与结果表使用的短名。
        hf_id: Hugging Face 模型 ID。
        dim: 输出向量维度。不同模型维度不同，切换模型时向量库必须重建。
        query_prefix: 检索场景下 query 侧要加的指令前缀，仅 BGE 系列需要；文档侧不加。
        max_seq_length: 手动设置的最大序列长度，避免 sentence-transformers 默认 512 截断（Jina 支持 8192）。
        attn_implementation: 传给 transformers 的注意力实现。Jina v2 自定义架构不支持
            新版 transformers 的默认值 "torch"，须显式指定 "eager"；空串表示不指定。
        trust_remote_code: 是否允许加载模型仓库自带的自定义代码。Jina v2 的 JinaBertModel
            需要 True，否则会退化成普通 BertModel、rotary 权重随机初始化，向量完全不可用。
        torch_dtype: 权重精度（"bfloat16"/"float16"/"float32"）。Qwen3-Embedding-8B 有 8B 参数，
            fp32 要占约 32GB 内存，消费级设备必须用 bfloat16（约 17GB）；空串用库默认。
        encode_batch_size: 编码 batch 大小。max_seq_length 大时显存占用 ∝ batch × max_seq_length，
            Jina 设 2048/8 是 32GB MPS 上的实测安全值；其余模型默认 32。
    """

    name: str
    hf_id: str
    dim: int
    query_prefix: str = ""
    max_seq_length: int = 512
    attn_implementation: str = ""
    trust_remote_code: bool = False
    torch_dtype: str = ""
    encode_batch_size: int = 32


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec(name="bge-large", hf_id="BAAI/bge-large-zh-v1.5", dim=1024, query_prefix=BGE_QUERY_PREFIX),
        ModelSpec(name="bge-small", hf_id="BAAI/bge-small-zh-v1.5", dim=512, query_prefix=BGE_QUERY_PREFIX),
        ModelSpec(name="gte-base", hf_id="thenlper/gte-base-zh", dim=768),
        ModelSpec(
            name="jina-base",
            hf_id="jinaai/jina-embeddings-v2-base-zh",
            dim=768,
            # 模型能力上 8192，但 sentence-transformers 按 max_seq_length 满长度 padding，
            # 显存 ∝ batch × max_seq_length：32GB MPS 上 8192 直接炸（52GiB buffer / OOM）。
            # 2048 覆盖参照集 95% 段落（超 2048 的 24/500 条被截断），配 batch 8 为实测安全值。
            max_seq_length=2048,
            encode_batch_size=8,
            attn_implementation="eager",
            trust_remote_code=True,
        ),
        ModelSpec(
            name="qwen3-8b",
            hf_id="Qwen/Qwen3-Embedding-8B",
            dim=4096,
            query_prefix=QWEN_QUERY_PREFIX,
            torch_dtype="bfloat16",
        ),
    ]
}


def get_model_spec(name: str) -> ModelSpec:
    """按短名查找模型描述。

    Args:
        name: 注册表中的模型短名，如 "bge-large"。

    Returns:
        对应的 ModelSpec。

    Raises:
        KeyError: 如果短名未注册，异常信息中列出全部可选短名。
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"未知模型 '{name}'，可选: {available}") from None


def build_query_text(spec: ModelSpec, query: str) -> str:
    """为 query 拼接模型要求的指令前缀（文档侧不拼）。

    BGE 官方要求检索场景 query 加"为这个句子生成表示以用于检索相关文章："前缀，
    漏加会直接掉召回（文章实测掉了几个点）。

    Args:
        spec: 模型描述。
        query: 原始 query 文本。

    Returns:
        拼接前缀后的文本；无需前缀的模型原样返回。
    """
    return f"{spec.query_prefix}{query}" if spec.query_prefix else query
