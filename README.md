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
| `jina-base` | `jinaai/jina-embeddings-v2-base-zh` | 768 | 多语言与长文本（模型能力 8192，本仓按显存实测锁 2048，见坑 2） |
| `qwen3-8b` | `Qwen/Qwen3-Embedding-8B` | 4096 | LLM 级 embedding，效果顶级但重：bf16 加载约 17GB 内存，建议 32GB 内存起 |

## 跑分结果

示例测试集（60 条，api/config/faq/doc 四类），macOS(M1/MPS) + transformers 4.x 环境：

| 模型 | recall@1 | recall@3 | recall@5 | recall@10 | mrr | 编码耗时(s) |
| --- | --- | --- | --- | --- | --- | --- |
| bge-large | 93.33% | 100.00% | 100.00% | 100.00% | 0.9667 | 1.8 |
| bge-small | 93.33% | 98.33% | 98.33% | 98.33% | 0.9598 | 0.1 |
| gte-base | 88.33% | 96.67% | 98.33% | 100.00% | 0.9279 | 0.6 |
| jina-base | 91.67% | 100.00% | 100.00% | 100.00% | 0.9583 | 0.6 |
| qwen3-8b | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 20.5 |

编码耗时含 query + corpus 共 120 条文本，仅供量级参考（受设备与缓存状态影响）。

### 公开参照集：T2Retrieval 500 条（真实网页搜索场景）

`data/t2retrieval_500.jsonl`，来自 [mteb/T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval)（C-MTEB 检索子任务）固定种子采样，构建方式见 [data/README.md](data/README.md)。同环境跑分：

| 模型 | recall@1 | recall@3 | recall@5 | recall@10 | mrr | 编码耗时(s) |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-8b | 96.20% | 98.80% | 99.60% | 99.80% | 0.9758 | 2066.7 |
| gte-base | 94.40% | 97.60% | 98.00% | 98.80% | 0.9618 | 11.6 |
| bge-large | 93.40% | 97.40% | 98.20% | 98.80% | 0.9560 | 40.0 |
| jina-base | 93.00% | 97.00% | 97.60% | 98.40% | 0.9518 | 36.8 |
| bge-small | 91.40% | 94.80% | 96.20% | 97.20% | 0.9359 | 2.2 |

注意两点：一，这是候选池 500 的同构任务，绝对分数不可与 C-MTEB 榜单的 118k 全库 ndcg 对表；二，**换数据，结论真的会变**——自建技术文档集上 bge-large 压过 gte-base（93.33% vs 88.33%），在这套真实网页搜索数据上 gte-base 反超（94.40% vs 93.40%）。这正是本仓存在的原因：选型决策要基于你自己的数据重跑，而不是任何单一榜单或本文的数字。qwen3-8b 两套数据都第一，但编码耗时是 bge-large 的 50 倍（长段落集上 34 分钟 vs 40 秒），是否值得取决于你的延迟预算。

Qwen3-Embedding-8B 钱上零成本（Apache 2.0 开源权重、本地推理），成本在算力侧：约 16GB 下载、bf16 加载占约 17GB 内存、编码耗时是 bge-large 的 10 倍量级。本测试集 60 条偏简单（满分不奇怪），但在 harder 数据集上 Qwen3-Embedding 系列在 C-MTEB 榜单也是第一梯队，属于「效果天花板、资源换来的」。query 侧同样要加 Instruct 前缀（本仓 `QWEN_QUERY_PREFIX` 已内置）。

> 数据集只有 60 条且答案段落较短，绝对分数与文章中 500 条规模的自建集不同，关注模型间的相对差距。换成你自己的数据重跑：替换 `data/qa_pairs.jsonl` 即可，格式见 [data/README.md](data/README.md)。

## 实测踩过的六个坑（本仓已内置处理）

1. **维度不一致**：四个模型输出维度不同（1024/512/768），切换模型时向量库必须重建，否则维度对不上直接报错。
2. **token 截断与显存旋钮**：sentence-transformers 默认 `max_seq_length=512` 会静默截断长文档；但调大时注意显存占用 ∝ `batch × max_seq_length`——Jina 在 8192 满长度 padding 下 32GB MPS 直接炸（52GiB tokenizer buffer / MPS OOM），实测安全值为 2048 + batch 8（覆盖公开参照集 95% 段落）。**max_seq_length 不只是截断旋钮，更是显存旋钮**。
3. **归一化**：编码时统一 `normalize_embeddings=True`，归一化后点积即 cosine 相似度。切换向量库时务必确认它算的是内积还是余弦，结果会差很多。
4. **BGE 查询指令前缀**：BGE 官方要求检索场景 query 侧加前缀「为这个句子生成表示以用于检索相关文章：」，文档侧不加。漏加召回率会掉几个点（`build_query_text()` 已内置，见 `models.py`）。
5. **Jina v2 与新版 transformers 不兼容**：两层坑——一是 Jina v2 是自定义架构（`JinaBertModel` + rotary 位置编码），必须 `trust_remote_code=True`，否则静默退化成普通 BertModel、rotary 权重随机初始化，召回率直接归零（本仓实测 0%）；二是其自定义代码 import 了 transformers 5.x 已删除的 `transformers.onnx`，因此依赖锁定 `transformers<5`。两个开关都已写入 `ModelSpec`。另外国内网络下 HF 直连未认证下载会被限速到几乎不动，建议 `export HF_ENDPOINT=https://hf-mirror.com`。
6. **Qwen3-Embedding 的加载链路三连坑**：一是 tokenizer 转换需要 `protobuf`（缺失时报的是误导性的 "Unrecognized processing class"——AutoProcessor 把真实异常吞了）；二是 hf-mirror 对单文件 `resolve/` 请求会 308 回源 huggingface.co，`huggingface_hub>=0.36` 的元数据校验直接失败，tokenizer 与 `1_Pooling/` 等子目录配置可能拉不全，需手动补进快照目录；三是 sentence-transformers 5.7 的新 Pooling API 要求 `embedding_dimension` 字段，而 Qwen3 仓库的 `1_Pooling/config.json` 是旧版字段，需手动补 `"embedding_dimension": 4096`。本仓已声明 protobuf 依赖；缓存修复属一次性操作，不影响代码。

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
