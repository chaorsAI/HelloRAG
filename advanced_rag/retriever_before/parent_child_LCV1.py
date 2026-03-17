# parent_child_LCV1.py
# 预检索优化-父子索引在Langchain V1.0架构之后的示例
# ParentDocumentRetriever：LangChain中用于解决长文档检索困境的一种经典检索器。
# 在Langchain1.0之后推荐使用LCEL范式自定义父子索引！！！


from typing import List, Dict, Any
from uuid import uuid4

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.stores import InMemoryStore
from  langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.tools import tool
from langchain.agents import create_agent

from advanced_rag.models.models import get_ali_clients

#获得访问大模型和嵌入模型客户端
client,embeddings_model = get_ali_clients()

# 加载数据
loader = TextLoader("../data/deepseek百度百科.txt",encoding="utf-8")
docs = loader.load()

# 创建向量数据库对象
vector_store = Chroma(
    collection_name="split_parents", embedding_function = embeddings_model
)
# 键值存储，用于通过ID快速查找父文档
doc_store = InMemoryStore()

# 查看长度
print(f"文章的长度：{len(docs[0].page_content)}")

# 子块是父块内容的子集
#创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)
#创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)


# 处理文档并建立父子关联
def process_documents_for_parent_child_retrieval(docs: List[Document]) -> None:
    """核心处理函数：创建父子文档并存储"""
    parent_docs = parent_splitter.split_documents(docs)

    all_child_docs = []
    parent_id_to_doc = {}

    for parent_doc in parent_docs:
        # 为每个父文档生成唯一ID
        parent_id = str(uuid4())
        parent_doc_with_id = Document(
            page_content=parent_doc.page_content,
            metadata={**parent_doc.metadata, "doc_id": parent_id, "type": "parent"}
        )
        parent_id_to_doc[parent_id] = parent_doc_with_id

        # 为父文档生成子文档
        children = child_splitter.split_documents([parent_doc])
        for child in children:
            child.metadata.update({
                "parent_doc_id": parent_id,
                "type": "child"
            })
            all_child_docs.append(child)

    # 存储
    if all_child_docs:
        vector_store.add_documents(all_child_docs)

    # 存储父文档
    for pid, pdoc in parent_id_to_doc.items():
        doc_store.mset([(pid, pdoc)])


# 核心检索函数
def retrieve_parent_docs_core(query: str, k: int = 3) -> List[Document]:
    """检索逻辑：检索子文档 -> 去重父ID -> 返回父文档"""
    # 第一步：基于向量相似性检索子文档
    retrieved_children = vector_store.similarity_search(query, k=k)

    if not retrieved_children:
        return []

    # 第二步：去重并获取父文档ID
    parent_ids = list(set([
        child.metadata.get("parent_doc_id")
        for child in retrieved_children
        if child.metadata.get("parent_doc_id")
    ]))

    # 第三步：从文档存储中获取完整的父文档
    if parent_ids:
        parent_docs = doc_store.mget(parent_ids)
        # 过滤掉可能的None值
        return [doc for doc in parent_docs if doc is not None]

    return []


# 包装成LangChain Agent可用的Tool
@tool
def parent_child_retriever_tool(query: str) -> str:
    """
    一个专用于父子索引检索的工具。
    输入一个查询，它将检索最相关的子块，然后返回对应的完整父文档以提供更丰富的上下文。

    Args:
        query: 用户的查询问题

    Returns:
        检索到的父文档内容拼接成的字符串
    """
    parent_docs = retrieve_parent_docs_core(query, k=3)
    if not parent_docs:
        return "未找到相关文档。"

    # 将多个父文档的内容合并返回
    contexts = [f"[文档片段 {i + 1}]:\n{doc.page_content}"
                for i, doc in enumerate(parent_docs)]
    return "\n\n".join(contexts)


process_documents_for_parent_child_retrieval(docs)

query = "Deepseek面临的挑战有哪些？"

# # ============ 测试代码 - 相似性搜索 start ============
# print("============ 测试代码 - 相似性搜索 ============")
# result = retrieve_parent_docs_core(query)
# print(f"查询: {query}")
# print(f"检索到 {len(result)} 个父文档")
# for i, doc in enumerate(result):
#     print(f"文档 {i+1}: {doc.page_content[:130]}...")
# exit()
# # ============ 测试代码 - 相似性搜索 end ============


# ============ 测试代码 - start ============
print("============ 测试代码 - 问答 ============")
# 定义工具列表
tools = [parent_child_retriever_tool]
agent = create_agent(
    client,
    tools=tools,
)
result = agent.invoke({"messages":[{"role":"user","content":"Deepseek面临的挑战有哪些？"}]})
print(result["messages"][-1].content)
# ============ 测试代码 - start ============