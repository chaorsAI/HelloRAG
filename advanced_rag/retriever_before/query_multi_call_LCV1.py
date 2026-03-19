# query_multi_call_LCV1.py
# 预检索-查询-问题丰富优化示例。在 Advanced RAG 中主要解决 **“词汇鸿沟”** 和 **“语义不完整性”** 的核心痛点
# query_multi_call_MultiQueryRetriever 在 Langchain 1.0时代的自定义实现(主要针对MultiQueryRetriever)


from typing import List
import asyncio

from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from advanced_rag.models import get_ali_clients


#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

# 加载文档
loader = TextLoader("../data/deepseek百度百科.txt",encoding="utf-8")
docs = loader.load()

# 创建文档分割器，并分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 创建向量数据库
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=embeddings_model)
# 创建检索器
retriever = vectorstore.as_retriever()

# 自定义实现召回函数
async def advanced_multi_query_retrieve(question: str,
                                        llm,
                                        base_retriever,
                                        num_queries: int = 3) -> List[Document]:
    """
    多路召回核心函数
    参数完全可控，流程清晰可见
    """
    # 1. 动态生成多个查询
    generation_prompt = PromptTemplate.from_template("""
    你是一个AI语言模型助手。你的任务是根据以下问题生成正好{num}个不同角度的搜索查询。
    要求通过对用户问题产生多种观点，你的目标是提供帮助用户克服基于距离的相似性搜索的一些限制。
    问题：{question}
    只返回{num}个查询，每行一个：
    """)

    chain = generation_prompt | llm
    result = await chain.ainvoke({"question": question, "num": num_queries})

    # 解析生成的查询列表
    generated_queries = [q.strip() for q in result.content.split('\n') if q.strip()]
    print(f"生成的查询: {generated_queries}")  # 完全可观测

    # 2. 并行执行所有查询（提升效率）
    tasks = [base_retriever.ainvoke(q) for q in generated_queries[:num_queries]]
    all_results = await asyncio.gather(*tasks)

    # 3. 结果去重
    seen_ids = set()
    unique_docs = []
    for doc_list in all_results:
        for doc in doc_list:
            # 基于内容哈希或ID去重
            doc_id = hash(doc.page_content[:200])  # 简单示例
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
    return unique_docs[:20]  # 限制返回总数


# 同步调用封装
def multi_query_retrieve(question: str, num_queries: int = 3):
    import asyncio
    return asyncio.run(advanced_multi_query_retrieve(
        question, llm, retriever, num_queries
    ))


# 使用：数量完全动态控制
unique_docs = multi_query_retrieve("deepseek的应用场景", num_queries=5)
print("------------ unique_docs  ------------")
print(len(unique_docs))
print(unique_docs)