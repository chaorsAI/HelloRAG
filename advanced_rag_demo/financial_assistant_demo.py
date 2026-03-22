import os
import time
import uuid
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.stores import InMemoryStore

from langchain_classic.retrievers import MultiVectorRetriever

from advanced_rag.models import get_ali_clients


print("\033[1;31m本程序的运行需要科学上网！请先检查你是否开启了科学上网！\033[0m")

#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

print("\033[1;34m请根据自己的实际情况修改代码中TESSDATA_PREFIX环境变量、Poppler 的 bin 目录、"
      "tesseract安装目录这三者的具体值！\033[0m")
# 设置 TESSDATA_PREFIX 环境变量
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# 添加 Poppler 的 bin 目录到系统 PATH
os.environ["PATH"] += os.pathsep + r"D:\workspace\poppler-24.08.0\Library\bin"
# 添加tesseract安装目录到系统 PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

# 定义文件路径（示例路径，实际使用时需修改）
path = "data/2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告.pdf"

# ------------------------ 第一阶段：PDF解析处理 ------------------------
print("正在解析pdf文件，请耐心等候!")
start = time.time_ns()
# 使用unstructured库解析PDF文档
raw_pdf_elements = partition_pdf(
    filename=path,
    extract_images_in_pdf=False,   # 不提取PDF中的图片
    infer_table_structure=True,    # 启用表格结构识别
    chunking_strategy="by_title",  # 按标题分块策略
    max_characters=4000,          # 每个文本块最大字符数
    new_after_n_chars=3800,       # 达到3800字符后分新块
    combine_text_under_n_chars=2000, # 合并小于2000字符的碎片文本
    image_output_dir_path=path,   # 图片输出目录（本例未使用）
)

# 统计各类元素数量
category_counts = {}
for element in raw_pdf_elements:
    category = str(type(element))
    category_counts[category] = category_counts.get(category, 0) + 1

print("解析完成，元素类型统计:", category_counts)
print(f"解析运行时间：{ (time.time_ns() - start) / 1_000_000:.2f} ms")

# ------------------------ 第二阶段：元素分类处理 ------------------------
start = time.time_ns()
# 定义文档元素模型
class Element(BaseModel):
    """文档元素基类"""
    type: str     # 元素类型（table/text）
    text: Any     # 元素内容

# 分类处理PDF元素，并提取表格和文本元素
table_elements = []
text_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        table_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        text_elements.append(Element(type="text", text=str(element)))
print(f"识别到表格数量: {len(table_elements)}, 文本块数量: {len(text_elements)}")
print("表格内容示例:", table_elements[0:10])
print(f"元素分类运行时间：{ (time.time_ns() - start) / 1_000_000:.2f} ms")
# ------------------------ 第三阶段：内容摘要生成 ------------------------
start = time.time_ns()
# 定义摘要生成提示模板
prompt_text = """
您是一个专业的内容摘要助手，请对以下表格或文本块进行简洁的总结：
{element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# 构建摘要生成链
summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

print("准备提取表格摘要......")
# 批量生成表格摘要
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})  # 并发处理
print("----表格摘要示例:", table_summaries[0:10])

print("准备提取文本摘要......")
# 批量生成文本摘要
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
print("----文本摘要示例:", text_summaries[0:10])
print(f"内容摘要生成运行时间：{ (time.time_ns() - start) / 1_000_000:.2f} ms")
# ------------------------ 第四阶段：构建多向量检索器 ------------------------
start = time.time_ns()

# 创建向量数据库（用于存储摘要）
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embeddings_model
)

# 创建内存存储（用于存储原始内容）
store = InMemoryStore()
id_key = "doc_id"  # 文档标识键

# 初始化多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

#添加文本数据到向量数据库
text_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: text_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(text_ids, texts)))

# 添加表格数据到向量数据库
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

# ------------------------ 第五阶段：构建问答链 ------------------------
# 定义问答提示模板
template = """请仅根据以下上下文（包含文本和表格）回答问题：
{context}
问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建问答链
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 示例问答测试
question = "2019年比去年的营业收入增长了百分之多少"
print("回答：", chain.invoke(question))
print("检索结果：", retriever.invoke(question))
print(f"构建知识库及问答运行时间：{ (time.time_ns() - start) / 1_000_000:.2f} ms")