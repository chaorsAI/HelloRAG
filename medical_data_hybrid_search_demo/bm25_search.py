
# 使用 Python 的rank_bm25库实现 BM25算法 的示例
# 从 rank_bm25 库中导入 BM25Okapi 类，用于计算 BM25 相似度得分
from rank_bm25 import BM25Okapi
# 导入 jieba 库，用于中文分词
import jieba

# 定义一个包含多个文档的语料库，每个文档是一个字符串
corpus = [
    "这是第一个文档",
    "这是第二个文档",
    "这是第三个文档"
]

# 对语料库中的每个文档进行分词操作，使用 jieba.lcut() 函数将文档分割成词语列表
# 最终得到一个包含多个词语列表的列表，每个子列表对应一个文档的分词结果
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
print("tokenized_corpus:", tokenized_corpus)
# 使用分词后的语料库初始化 BM25Okapi 对象，后续将使用该对象进行相似度计算
bm25 = BM25Okapi(tokenized_corpus)

# 定义一个查询语句，即要查找相关文档的关键词
query = "第一个文档"
# 对查询语句进行分词操作，将其转换为词语列表
tokenized_query = jieba.lcut(query)
print("tokenized_query:", tokenized_query)
print('-' * 100)

# 调用 BM25Okapi 对象的 get_scores 方法，计算查询语句与语料库中每个文档的相似度得分
# 得到一个包含多个得分的列表，每个得分对应语料库中的一个文档
scores = bm25.get_scores(tokenized_query)
# 打印计算得到的相似度得分列表
print("scores:", scores)

# 调用 BM25Okapi 对象的 get_top_n 方法，根据查询语句的相似度得分从语料库中选取前 n 个最相关的文档
# 这里 n 设置为 1，表示只选取最相关的一个文档
top_n = bm25.get_top_n(tokenized_query, corpus, n=3)
# 打印选取的最相关文档列表
print("top_n:", top_n)  # 输出：['这是第一个文档']
