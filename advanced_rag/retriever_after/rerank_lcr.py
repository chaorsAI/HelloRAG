# rerank_lcr.py
# LongContextReorder：关注的是上下文本身，本质上他并不进行重新排序(使用模型或者其他维度)，旨在解决**注意力偏差**。
"""
研究发现，当 LLM 处理长序列输入（如多个检索到的文档）时，其对信息的关注度并非均匀分布。模型关注倾向于呈现显著的U型曲线：
-   **首因效应 (Primacy Bias)** ：模型对输入序列开头的 token 关注度最高。

-   **近因效应 (Recency Bias)** ：模型对输入序列末尾的 token 关注度次高。

-   **“迷失在中间” (Lost in the Middle)** ：位于序列中间部分的信息，无论其本身多重要，都**容易被忽略**，导致检索和推理性能大幅下降。

- 人类相关性：[5(最相关), 4(次相关)，3, 2, 1]
    - LongContextReorder 摸透了 LLM 的心思，预先做了处理
- LLM 接收序：[5, 3, 1, 2, 4](相当于桥接层)
- 由于 LLM 的**注意力偏差**问题，导致：
    - 将开头的 5 认为最相关
    - 将结尾的 4 认为次相关
    - 其实自然语言的相关性顺序**并未改变**
"""


from langchain_community.document_transformers import LongContextReorder
# 5,4,3,2,1
# 倒排：1,2,3,4,5
# 前一个，后一个：5,3,1,2,4

# 按相关性排序5，4，3，2，1，5是最相关的，相关性依次递减
documents = [
    "相关性:5",
    "相关性:4",
    "相关性:3",
    "相关性:2",
    "相关性:1",
]

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(documents)

print(reordered_docs)