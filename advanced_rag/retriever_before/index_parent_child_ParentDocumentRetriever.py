# index_parent_child_ParentDocumentRetriever.py
# 预检索-索引优化-父子索引示例
# ParentDocumentRetriever：LangChain中用于解决长文档检索困境的一种经典检索器。
# 特别是在Langchain1.0之后推荐使用LCEL范式！！！

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from  langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from advanced_rag.models.models import get_ali_clients

#获得访问大模型和嵌入模型客户端
client,embeddings_model = get_ali_clients()

# 加载数据
loader = TextLoader("../data/deepseek百度百科.txt",encoding="utf-8")
docs = loader.load()

# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="split_parents", embedding_function = embeddings_model
)
# 创建内存存储对象
store = InMemoryStore()

# 查看长度
print(f"文章的长度：{len(docs[0].page_content)}")

# 子块是父块内容的子集
#创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)
#创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)

#创建父子文档检索器，帮我们通过检索子块，返回父文档块
# topK = 2,相似度最高的子文档块（A,B） A,B属于同一个父， 父文档块被查询两次，不会去重
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store, # 文档存储对象
    child_splitter=child_splitter, # 子文档分割器，子文档存储到向量数据库
    parent_splitter=parent_splitter,# 主文档分割器，主文档存储到内存中
    search_kwargs={"k": 1}  # topK = 1,相似度最高的子文档块
)

#添加文档集
retriever.add_documents(docs)

print(f"主文块的数量：{len(list(store.yield_keys()))}")


# # ============ 测试代码 - 相似性搜索 start ============
# print("============ 测试代码 - 相似性搜索 ============")
#
# # 这里我们通过向量数据库的similarity_search方法搜索出来的是与用户问题相关的子文档块的内容，
# # 下面我们使用检索器的get_relevant_documents的方法来对这个问题进行检索，
# # 它会返回该子文档块所属的主文档块的全部内容：
#
# print("------------similarity_search------------------------")
# sub_docs = vectorstore.similarity_search("deepseek的应用场景")
# print(sub_docs[0].page_content)
#
# print("------------get_relevant_documents-----------通过子找父-------------")
# retrieved_docs = retriever.invoke("deepseek的应用场景")
# print(retrieved_docs[0].page_content)
# exit()
# # ============ 测试代码 - 相似性搜索 end ============

#创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

#由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

#创建chain
chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | client | StrOutputParser()

print("------------模型回复------------------------")

response = chain.invoke({"question": "deepseek面临的技术竞争"})
print(response)
