# hybrid_search_EnsembleRetriever.py
# 检索优化-混合检索。通过组合两种根本不同的检索方式，来克服单一方法的固有缺陷，从而在“**精确匹配**”与“**语义理解**”这两个常常矛盾的目标上取得最佳平衡。
"""
- 检索方式：
    -  **稀疏检索**：代表算法是**BM25**。它将文档和查询表示为庞大的、稀疏的向量（维度对应所有词表里的词，大部分值为0），通过统计关键词出现的频率、位置等信息计算相关性。它**不关心词的语义**，只关心“词是否匹配”。

    -   **稠密检索**：基于**嵌入模型**（如BGE、text-embedding-v3）。它将文本映射到数百或数千维的稠密向量空间，语义相似的文本其向量距离也近。它**深度理解语义**，能处理同义词、转述和上下文推理。
- 核心流程：
    -   **并行检索**：对于同一个用户查询，系统同时向稀疏检索索引（如倒排索引）和稠密检索索引（如向量数据库）发起搜索。

    -   **结果融合**：两路检索各自返回一个排序列表。由于它们的评分体系完全不同（一个是统计分数，一个是余弦相似度），无法直接比较。因此，需要做**归一化处理**。
"""


from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers.bm25 import BM25Retriever

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

from langchain_classic.retrievers import EnsembleRetriever

from langchain_text_splitters import RecursiveCharacterTextSplitter

from advanced_rag.models import get_ali_clients


#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# 加载文档
loader = TextLoader("../data/deepseek百度百科.txt",encoding="utf-8")
docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
split_docs = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=split_docs, embedding=embeddings_model
)

question = "相关评价"

print("=============== 【检索模块】===============")
# 向量检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
doc_vector_retriever = vector_retriever.invoke(question)
print("-------------------向量检索-------------------------")
pretty_print_docs(doc_vector_retriever)

# 关键词检索
BM25_retriever = BM25Retriever.from_documents(split_docs)
BM25Retriever.k = 3
doc_BM25Retriever = BM25_retriever.invoke(question)
print("-------------------BM25检索-------------------------")
pretty_print_docs(doc_BM25Retriever)

# 混合检索
#EnsembleRetriever是Langchain集合多个检索器的检索器。
ensembleRetriever = EnsembleRetriever(retrievers=[BM25_retriever, vector_retriever], weights=[0.5, 0.5])
retriever_doc = ensembleRetriever.invoke(question)
print("-------------------混合检索-------------------------")
print(retriever_doc)

# 创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 创建chain
ensemble_chain = RunnableMap({
    "context": lambda x: ensembleRetriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()
vector_chain = RunnableMap({
    "context": lambda x: vector_retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()


print("=============== 【模型回复】===============")
print("------------ 混合检索[0.5, 0.5] ------------------------")
print(ensemble_chain.invoke({"question":question}))
print("------------ 纯向量检索 ------------------------")
print(vector_chain.invoke({"question":question}))