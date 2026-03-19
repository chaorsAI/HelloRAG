# rerank_rag_fusion.py
# 检索后优化-重排序-RAG-Fusion。
# RAG-Fusion：通过使用多 重查询生成和互惠排名融合（Reciprocal Rank Fusion）对搜索结果进行重新排序。在Multi Query的基础上，对其检索结果进行重新排序(即reranking)后输出Top K个最相关文档，最后将这top k个文档喂给LLM并生成最终的答案


from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads
from langchain_core.runnables import chain

# 旧的 hub prompt 导入方式
# 旧的 hub prompt 导入方式：client.pull_prompt

from advanced_rag.models import get_ali_clients, get_langsimth_client


#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

texts=[
    "人工智能在医疗诊断中的应用。",
    "人工智能如何提升供应链效率。",
    "NBA季后赛最新赛况分析。",
    "传统法式烘焙的五大技巧。",
    "红楼梦人物关系图谱分析。",
    "人工智能在金融风险管理中的应用。",
    "人工智能如何影响未来就业市场。",
    "人工智能在制造业的应用。",
    "今天天气怎么样",
    "人工智能伦理：公平性与透明度。"
]

# 创建向量数据库对象
vectorstore = Chroma.from_texts(
    texts=texts, embedding= embeddings_model
)   

retriever = vectorstore.as_retriever()

#从langchain官网拉取预先定义好的prompt
# prompt = hub.pull("langchain-ai/rag-fusion-query-generation")
# print("-" * 15 + "hub.pull" + "-" * 15)
# print(prompt)
langsimth_client = get_langsimth_client()
prompt = langsimth_client.pull_prompt("langchain-ai/rag-fusion-query-generation")
print("-" * 15 + "langsimth_client" + "-" * 15)
print(prompt)
# exit()

#也可以手工定义prompt如下：
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
#     ("user", "Generate multiple search queries related to: {original_query}"),
#     ("user", "OUTPUT (4 queries):")
# ])

# 创建多重查询chain
generate_queries = (
    prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

original_query = "人工智能的应用"
queries = generate_queries.invoke({"original_query": original_query})
print(f"原始查询：{original_query},生成的查询：{queries}")

@chain
def reciprocal_rank_fusion(results: list[list], k=60):
    """互逆排序融合算法，用于合并多个排序文档列表
    Args:
        results: 包含多个排序文档列表的二维列表
        k: 融合公式中的平滑参数（默认60），值越小排名影响越大
    Returns:
        按融合分数降序排列的文档列表，每个元素为(文档对象, 分数)元组
    """
    # 初始化融合分数字典（key=序列化文档，value=累计分数）
    fused_scores = {}

    # 遍历每个检索结果列表（每个查询对应的结果）
    for docs in results:
        # 对当前结果列表中的文档进行遍历（rank从0开始计算）
        for rank, doc in enumerate(docs):
            # 序列化文档对象为字符串（用于唯一标识）
            doc_str = dumps(doc)
            # 初始化文档得分（如果是首次出现）
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 计算并累加RRF分数：1 / (当前排名 + k)
            # 排名越靠前（rank值小）的文档获得的分数越高
            fused_scores[doc_str] += 1 / (rank + k)

    # 按融合分数降序排序（分数越高排名越前）
    reranked_results = [
        (loads(doc), score)  # 反序列化还原文档对象
        for doc, score in sorted(fused_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
    ]

    return reranked_results

original_query = "人工智能的应用"
'''
generate_queries会生成4个多角度的query,
retriever.map()的作用是根据generate_queries的结果映射出4个retriever(可以理解为同时复制出4个retriever)
与generate_queries生成的4个query对应，
并为每个query检索出来的一组相关文档集(默认为4个相关文档)，
那么4个query总共可以生成16个相关文档。
最后会经过RRF算法重新排序后输出最相关的文档
'''
chain = generate_queries | retriever.map() | reciprocal_rank_fusion

# 输入结果列表
result_list = chain.invoke({"original_query": original_query})
# 提取文档内容和对应分数
contents = [doc[0].page_content for doc in result_list]
scores = [doc[1] for doc in result_list]

combined_tuples = list(zip(contents, scores))
print("--"*15,"最相关的文档及其得分：")
for item in combined_tuples:
    print(item)


print("--"*15,"分析一下这些分数是如何统计出来的：")
#分析一下这些分数是如何统计出来的
chain1 = generate_queries | retriever.map() 
chain1_result = chain1.invoke({"original_query": original_query})

#原始输出是一个二维列表，每个元素是由4个query生成的Document列表
print(chain1_result)

# 处理输出格式
contents = [doc.page_content 
            for group in chain1_result  # 遍历外层列表
            for doc in group]        # 遍历内层文档列表
print(contents)

