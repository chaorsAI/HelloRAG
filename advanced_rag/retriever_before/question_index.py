import langchain
from typing import List
from langchain_core.stores import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import MultiVectorRetriever
import uuid
from langchain_core.documents import Document  
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnableMap, RunnableParallel
from pydantic import BaseModel, Field

from advanced_rag.models.models import get_ali_clients

#获得访问大模型和嵌入模型客户端
client,embeddings_model = get_ali_clients()

# 初始化文档加载器列表
loader = TextLoader("../data/deepseek百度百科.txt", encoding="utf-8")
docs = loader.load()

# 初始化递归文本分割器（设置块大小和重叠）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# 初始化Chroma向量数据库（存储生成的问题向量）
vectorstore = Chroma(
    collection_name="hypo-questions", embedding_function=embeddings_model
)
# 初始化内存存储（存储原始文档）
store = InMemoryByteStore()

id_key = "doc_id" # 文档标识键名

# 配置多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore, #  向量数据库，存储生成的问题向量（调用对话模型生成）
    byte_store=store, # 字节存储，存储原始文档
    id_key=id_key,
)

# 为每个原始文档生成唯一ID
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 以下开始用大模型生成假设性问题
# 用于定义假设性答案的规定格式
class HypotheticalQuestions(BaseModel):
    """约束生成假设性问题的格式"""
    questions: List[str] = Field(..., description="List of questions")

# 定义一个简单的“索引”和处理器（心智模型的代码化），用于生成假设性答案
class HypotheticalQuestionIndex:
     def process(self) -> HypotheticalQuestions:
        # hy_question = HypotheticalQuestions(original_query)
        # 查询理解与增强（此处可调用大模型或规则引擎）
        # 此处使用双括号 {{ 和 }} 是为了在字符串中转义出单个 { 和 }，以确保最终输出的 JSON 格式正确。
        prompt = ChatPromptTemplate.from_template(
            """请基于以下文档生成3个假设性问题（必须使用JSON格式）:
            {doc}

            要求：
            1. 输出必须为合法JSON格式，包含questions字段
            2. questions字段的值是包含3个问题的数组
            3. 使用中文提问
            示例格式：
            {{
                "questions": ["问题1", "问题2", "问题3"]
            }}"""
        )

        # 创建假设性问题链
        '''
        其中的client.with_structured_output可以理解为输出解析器的一种更高级用法
        将大模型的输出转换为HypotheticalQuestions所限定的格式，
        而HypotheticalQuestions要求的格式是：
        定义了一个字段 questions，它具有以下特性：
        类型注解：List[str] 表示 questions 字段应该是一个字符串列表。
        必需性：Field(...) 中的省略号 ... 表示这个字段是必需的。
        描述信息：description="List of questions" 为该字段添加了描述，这对于生成文档或帮助理解模型结构很有用。
        '''
        chain = (
                {"doc": lambda x: x.page_content}
                | prompt
                # 将LLM输出构建为字符串列表
                | client.with_structured_output(
            HypotheticalQuestions
            )
                # # 提取问题列表
                # | (lambda x: x.questions)
        )

        # 测试-在单个文档上调用链，链的最终输出是大模型答复的假设性问题列表
        # print("测试：",docs[0])
        # print("测试生成问题：",chain.invoke(docs[0]))

        # 批量处理所有文档生成假设性问题（最大并行数5），每个切块后的文档块都对应的生成三个问题
        hy_question = chain.batch(docs, {"max_concurrency": 5})
        # print("假设性问题列表：",hypothetical_questions)

        return hy_question

# 取出答案索引生成器中生成的问题列表
questions = [hy_questions.questions  for hy_questions in HypotheticalQuestionIndex().process()]

# 将生成的问题转换为带元数据的文档对象
question_docs = []
for i, question_list in enumerate(questions):
    question_docs.extend(
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )

# print(question_docs)


retriever.vectorstore.add_documents(question_docs) # 将问题文档存入向量数据库
retriever.docstore.mset(list(zip(doc_ids, docs))) # 将原始文档存入字节存储（通过ID关联）
#以上的过程是可以在构建知识库的时候提前完成的

# 测试-执行相似性搜索
query = "deepseek受到哪些攻击？"
# sub_docs = retriever.vectorstore.similarity_search(query)
# print("-------------相似性：--------------")
# print("测试-执行相似性搜索：",sub_docs)

prompt =  ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}")

# 生成问题回答链
chain = RunnableParallel({
    "doc": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | client | StrOutputParser()

# 生成问题回答
answer = chain.invoke({"question": query})
print("-------------回答--------------")
print(answer)
#  返回的是知识块
retrieved_docs = retriever.invoke(query) 
print("-------------检索到的问题--------------")
print(retrieved_docs)
