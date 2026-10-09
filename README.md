# zh-embedding-bench

中文 Embedding 模型检索实测：**BGE / GTE / Jina**，用自建中文问答测试集跑 recall@k 与 MRR。

配套文章：《Embedding模型怎么选——BGE、GTE、Jina我跑了中文检索实测，差距比想象大》。姊妹项目：[wiki-rag](https://github.com/helloworldtang/wiki-rag)。

## 为什么有这个仓

榜单分数（C-MTEB）好看，不代表你的场景能用。这个仓把文章里的实测方法落成可复现的代码：自建中文问答测试集，用问题做 query、答案段落做候选，逐模型跑 recall@1/3/5/10 与 MRR，让选型决策基于你自己的数据。

## 安装

```bash
git clone https://github.com/helloworldtang/embedding.git
cd embedding
uv sync
```

首次跑分会从 Hugging Face 下载模型（bge-small 约 100MB，bge-large 约 1.3GB，jina-base 约 650MB），缓存在本机，之后不再重复下载。

## 快速开始

```bash
# 跑全部四个模型
uv run python scripts/run_benchmark.py

# 只跑指定模型（逗号分隔）
uv run python scripts/run_benchmark.py --models bge-large,gte-base

# 指定测试集与设备
uv run python scripts/run_benchmark.py --data data/qa_pairs.jsonl --device mps
```

结果打印到控制台，并写入 `results/benchmark_<时间戳>.md` 与 `.json`。

## 模型清单

| 短名 | Hugging Face ID | 维度 | 特点 |
| --- | --- | --- | --- |
| `bge-large` | `BAAI/bge-large-zh-v1.5` | 1024 | 中文检索第一梯队，资源占用高 |
| `bge-small` | `BAAI/bge-small-zh-v1.5` | 512 | bge-large 的轻量版，速度快 |
| `gte-base` | `thenlper/gte-base-zh` | 768 | 均衡选手，速度/效果/尺寸都不差 |
| `jina-base` | `jinaai/jina-embeddings-v2-base-zh` | 768 | 多语言与长文本（8192 token）强项 |

## 跑分结果

示例测试集（60 条，api/config/faq/doc 四类），macOS(M1/MPS) + transformers 4.x 环境：

| 模型 | recall@1 | recall@3 | recall@5 | recall@10 | mrr | 编码耗时(s) |
| --- | --- | --- | --- | --- | --- | --- |
| bge-large | 93.33% | 100.00% | 100.00% | 100.00% | 0.9667 | 1.8 |
| bge-small | 93.33% | 98.33% | 98.33% | 98.33% | 0.9598 | 0.1 |
| gte-base | 88.33% | 96.67% | 98.33% | 100.00% | 0.9279 | 0.6 |
| jina-base | 91.67% | 100.00% | 100.00% | 100.00% | 0.9583 | 0.6 |

编码耗时含 query + corpus 共 120 条文本，仅供量级参考（受设备与缓存状态影响）。

> 数据集只有 60 条且答案段落较短，绝对分数与文章中 500 条规模的自建集不同，关注模型间的相对差距。换成你自己的数据重跑：替换 `data/qa_pairs.jsonl` 即可，格式见 [data/README.md](data/README.md)。

## 实测踩过的五个坑（本仓已内置处理）

1. **维度不一致**：四个模型输出维度不同（1024/512/768），切换模型时向量库必须重建，否则维度对不上直接报错。
2. **token 截断**：sentence-transformers 默认 `max_seq_length=512` 会静默截断长文档；Jina 支持 8192，需手动设置（见 `models.py` 的 `max_seq_length` 字段）。
3. **归一化**：编码时统一 `normalize_embeddings=True`，归一化后点积即 cosine 相似度。切换向量库时务必确认它算的是内积还是余弦，结果会差很多。
4. **BGE 查询指令前缀**：BGE 官方要求检索场景 query 侧加前缀「为这个句子生成表示以用于检索相关文章：」，文档侧不加。漏加召回率会掉几个点（`build_query_text()` 已内置，见 `models.py`）。
5. **Jina v2 与新版 transformers 不兼容**：两层坑——一是 Jina v2 是自定义架构（`JinaBertModel` + rotary 位置编码），必须 `trust_remote_code=True`，否则静默退化成普通 BertModel、rotary 权重随机初始化，召回率直接归零（本仓实测 0%）；二是其自定义代码 import 了 transformers 5.x 已删除的 `transformers.onnx`，因此依赖锁定 `transformers<5`。两个开关都已写入 `ModelSpec`。另外国内网络下 HF 直连未认证下载会被限速到几乎不动，建议 `export HF_ENDPOINT=https://hf-mirror.com`。

## 作为库使用

```python
from zh_embedding_bench.benchmark import benchmark_model
from zh_embedding_bench.data import load_qa_pairs
from zh_embedding_bench.models import get_model_spec

pairs = load_qa_pairs("data/qa_pairs.jsonl")
result = benchmark_model(get_model_spec("bge-large"), pairs)
print(result.metrics)  # {'recall@1': ..., 'mrr': ...}
```

## 开发

```bash
uv sync --group dev   # 或 uv sync（dev 组默认安装）
uv run pytest          # 离线单测：指标手算样例 / 数据加载 / 模型注册表
uv run ruff check .
uv run mypy src scripts
```

模型下载相关的端到端跑分不进单测（太重），由 `scripts/run_benchmark.py` 手动执行。

## License

[MIT](LICENSE)
