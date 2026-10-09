# 测试集说明

本仓支持两类测试集，格式相同（每行一个 JSON 对象，`query` / `answer` / `category` 三字段）：

| 文件 | 性质 | 规模 |
| --- | --- | --- |
| `qa_pairs.jsonl` | 自建示例集（技术文档场景） | 60 条 |
| `t2retrieval_500.jsonl` | 公开参照集（T2Retrieval 采样） | 500 条 |

## 检索任务定义（两类通用）

- 候选池 = 文件内全部 `answer`；
- 每条 query 的正确答案（gold）就是它自己配对的 `answer`；
- 其余段落天然构成干扰项。

## 自建集 `qa_pairs.jsonl`

```json
{"query": "这个API的rate limit是多少", "answer": "开放接口的速率限制为……", "category": "api"}
```

category 含义：`api`（API 使用）/ `config`（配置说明）/ `faq`（常见问题）/ `doc`（技术文档）。当前分布 15/15/14/16，共 60 条。同 category 的其他答案构成困难负例。

### 换成你自己的数据

直接替换或追加 JSONL 行即可，字段不变。文章中提到的真实测试集约 500 条，规模越大区分度越高；注意 query 与 answer 的措辞要有自然差异（用户怎么问 vs 文档怎么写），这是评测区分度的来源。

## 公开参照集 `t2retrieval_500.jsonl`

来源 [mteb/T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval)（C-MTEB 检索子任务，搜狗真实搜索日志，Apache 2.0）。构建方式：

```bash
uv sync --extra data
uv run --extra data python scripts/build_t2retrieval_set.py   # --n 500 --seed 42 可调
```

- dev split（22,812 query）中固定种子采样 500 条，query 原样保留，gold 取该 query 的第一个正例段落（平均每 query 约 5 个正例，只取一个，口径偏保守）；
- 段落做了轻量清理（去 `<br>`/`<img>` 等 HTML 标签、压空白），category 统一为 `web`；
- 候选池 = 500 条 gold 段落，与自建集任务同构（不是官方 118k 全库检索），**绝对分数不可与 C-MTEB 榜单 ndcg 对表**，价值在于公开、可复现、五模型同口径横比；
- 原始 parquet 缓存在 `data/raw/`（约 150MB，已 gitignore），`t2retrieval_500.jsonl` 本体入库。

| 字段 | 含义 |
| --- | --- |
| `query` | 问题文本，作为检索 query |
| `answer` | 答案段落，作为候选池中的一条文档 |
| `category` | 类别：`api`（API 使用）/ `config`（配置说明）/ `faq`（常见问题）/ `doc`（技术文档） |

## 检索任务定义

- 候选池 = 全部 `answer`（60 条）；
- 每条 query 的正确答案（gold）就是它自己配对的 `answer`；
- 同 category 的其他答案天然构成困难负例（比如问 rate limit，候选池里还有限流处理、并发限制等高度相似的话题）。

## 换成你自己的数据

直接替换或追加 JSONL 行即可，字段不变。文章中提到的真实测试集约 500 条，规模越大区分度越高；注意 query 与 answer 的措辞要有自然差异（用户怎么问 vs 文档怎么写），这是评测区分度的来源。

当前分布：api 15 / config 15 / faq 14 / doc 16，共 60 条。
