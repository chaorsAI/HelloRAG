# query_multi_call_MultiQueryRetriever.py
# 预检索-查询-问题丰富优化示例。在 Advanced RAG 中主要解决 **“词汇鸿沟”** 和 **“语义不完整性”** 的核心痛点
# MultiQueryRetriever：Langchain中一个关键的**查询优化组件**。
# 本质逻辑： **“不要把所有鸡蛋放在一个篮子里”** 。
"""
- 核心流程：
    1.   **生成机制**：通过特定的 Prompt 指令（如“请生成 3 个不同视角的查询”），LLM 会分析原始 Query 的语义，生成一系列相关的衍生问题。这些衍生问题可能包括：
    -   **同义词替换**（如“LLM” -> “大语言模型”）
    -   **视角转换**（如“怎么用” -> “使用教程和步骤”）
    -   **细节补充**（如“学习 Python” -> “Python 语法、学习路线、实战项目”）

    2.   **检索与融合**：每个生成的 Query 都会独立进行向量检索，召回一批文档。最后，系统会对所有召回的文档进行 **去重** 和 **融合**，形成一个更全面的候选文档池
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_chroma.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate

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

#检索测试
relevant_docs= retriever.invoke('deepseek的应用场景')
print(relevant_docs)
#查看一下检索到的相关文档的数量：
print("检索器检索的文档数量为：",len(relevant_docs))
#检索测试-完成


#创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

#由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

chain = RunnableMap({
    "context": lambda x: relevant_docs,
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

# print("--------------优化前-------------------")
# response = chain.invoke({"question": "deepseek的应用场景"})
# print("大模型生成的不同视角问题为：",response)


print("--------------开始优化-------------------")
# 自定义提示词模板，用于改变query生成的数量
# 关键：必须包含 {question} 作为输入变量，并要求输出按指定格式分隔。
custom_prompt_template = """
你是一名专业的研究助手。请针对以下问题，生成 4 个不同的搜索查询，以便从向量数据库中查找相关信息。每个查询应从不同的技术角度或应用层面切入。请确保查询具体、专业且与原始问题高度相关。

原始问题：{question}

请严格按照以下格式输出，每个查询单独一行，无需编号：
查询1
查询2
...
查询N
"""
# 创建 PromptTemplate 对象
custom_prompt = PromptTemplate(
    input_variables=["question"], # 必须包含这些变量
    template=custom_prompt_template
)

# ============ 曲线救国：通过封装来实现生成query数量的灵活控制 ============
def create_multi_query_retriever(num_queries=3):
    print("============ 曲线救国：通过封装来实现生成query数量的灵活控制 ============")
    custom_prompt_template = f"""
    你是一名专业的研究助手。请针对以下问题，生成 {num_queries} 个不同的搜索查询，以便从向量数据库中查找相关信息。每个查询应从不同的技术角度或应用层面切入。请确保查询具体、专业且与原始问题高度相关。

    原始问题：{{question}}

    请严格按照以下格式输出，每个查询单独一行，无需编号：
    查询1
    查询2
    ...
    查询N
    """
    # 创建 PromptTemplate 对象
    custom_prompt = PromptTemplate(
        input_variables=["question"], # 必须包含这些变量
        template=custom_prompt_template
    )

    retrieval_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=custom_prompt  # 通过自定义prompt修改生成相关问题的数量，但是只能在custom_prompt写死，而不能当做灵活变量
    )

    return retrieval_from_llm

# 引入日志组件查看llm在原查询的基础上生成的多个查询
import logging
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

# 默认的MultiQueryRetriever
# MultiQueryRetriever 默认模板生成3个相关问题
# retrieval_from_llm = MultiQueryRetriever.from_llm(
#     retriever=retriever,
#     llm=llm,
#     prompt=custom_prompt    # 通过自定义prompt修改生成相关问题的数量，但是只能在custom_prompt写死，而不能当做灵活变量
# )

# 曲线救国：控制生成的query数目
retrieval_from_llm = create_multi_query_retriever(num_queries=2)


# 测试代码
unique_docs = retrieval_from_llm.invoke({"question":'deepseek的应用场景'})
print("------------ unique_docs  ------------")
print(len(unique_docs))
print(unique_docs)
