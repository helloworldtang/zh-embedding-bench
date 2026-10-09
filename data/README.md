# 测试集说明

`qa_pairs.jsonl` 是自建中文问答测试集（示例版 60 条），每行一个 JSON 对象：

```json
{"query": "这个API的rate limit是多少", "answer": "开放接口的速率限制为……", "category": "api"}
```

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
