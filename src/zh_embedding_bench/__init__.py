"""中文 Embedding 模型检索实测工具包。

配套文章《Embedding模型怎么选——BGE、GTE、Jina我跑了中文检索实测》，
以 recall@k 与 MRR 为指标对比主流中文 embedding 模型的检索效果。
"""

DEFAULT_KS = (1, 3, 5, 10)

__version__ = "0.1.0"
