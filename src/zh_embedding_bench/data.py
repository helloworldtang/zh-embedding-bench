"""测试集加载与检索候选池构造。

数据格式为 JSONL，每行一个 JSON 对象：{"query": ..., "answer": ..., "category": ...}。
检索任务：用 query 在全部 answer 构成的候选池中找对应的那条 answer；
同 category 的其他 answer 天然构成困难负例。
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QAPair:
    """一条中文问答对。

    Attributes:
        query: 问题文本，作为检索 query。
        answer: 答案段落，作为候选池中的一条文档。
        category: 问题类别（如 api/config/faq/doc），用于分桶分析。
    """

    query: str
    answer: str
    category: str


def load_qa_pairs(path: str | Path) -> list[QAPair]:
    """从 JSONL 文件加载问答测试集。

    Args:
        path: JSONL 文件路径。

    Returns:
        按文件行序排列的 QAPair 列表。

    Raises:
        FileNotFoundError: 如果文件不存在。
        ValueError: 如果某行缺少必填字段或 JSON 解析失败。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"测试集文件不存在: {file_path}")

    pairs: list[QAPair] = []
    with file_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc
            missing = [field for field in ("query", "answer", "category") if field not in obj]
            if missing:
                raise ValueError(f"第 {line_no} 行缺少必填字段: {missing}")
            pairs.append(QAPair(query=obj["query"], answer=obj["answer"], category=obj["category"]))
    if not pairs:
        raise ValueError(f"测试集为空: {file_path}")
    return pairs


def build_corpus(pairs: list[QAPair]) -> list[str]:
    """构建检索候选池。

    候选池就是全部 answer；pair i 的正确答案下标即 i。

    Args:
        pairs: 问答对列表。

    Returns:
        候选文档列表，下标与 pairs 一一对应。
    """
    return [pair.answer for pair in pairs]
