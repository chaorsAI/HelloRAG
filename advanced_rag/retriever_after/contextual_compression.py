# contextual_compression.py
# 后检索-上下文压缩过滤。使用给定查询的上下文来压缩它们，以便只返回相关信息，而不是立即按原样返回检索到的文档。
"""
# LLMChainExtractor:实现 “提取式”上下文压缩
- 核心：**“提问-提取”循环**
- 流程：
    1.   **输入**：接收一个原始查询（query）和一组由基础检索器（如向量库检索器）返回的原始文档（documents）。
    2.   **处理**：利用一个大语言模型（LLM），针对**每个原始文档**提出一个本质相同的问题：“**给定当前查询，这个文档中哪些部分是相关的？** ”
    3.   **输出**：LLM 会分析文档内容，并直接**提取（extract）** ​ 出与查询相关的原文片段（可以是句子或段落），过滤掉所有不相关的内容。最终，返回一个由这些精炼片段组成的新文档列表。

# LLMChainFilter** 是LangChain的 `retrievers.document_compressors`模块中实现 “过滤式”上下文压缩的核心类。他只针对文档作出”**是否相关**“的结论，输出”**是/否**“。
核心流程：
1.   **输入**：接收一个用户查询（Query）和一组通过向量检索或其他方式初步获取的文档列表。

2.   **处理**：针对**每一个**检索到的文档，构造一个特定的提示（Prompt），交由一个大语言模型（LLM）进行判断。这个提示通常包含查询和文档内容，要求LLM判断“该文档是否与回答查询相关”。

3.   **输出**：LLM给出一个“是/否”的二值判断。LLMChainFilter会收集所有被判断为“是”的文档，过滤掉被判断为“否”的文档，最终返回一个精炼后的、相关性更高的文档子集。

# EmbeddingsFilter：直接依赖于词向量（Embeddings）相似度，“这个文档和我的问题有关吗？”。并加入一个阈值判断：
# - similarity_threshold：所有相似度**大于等于**此阈值的文档被保留，低于此阈值的文档被过滤掉。

# EmbeddingsRedundantFilter：对文档内容进行过滤，“这些文档的内容是不是重复了？”
# DocumentCompressorPipeline：定义执行管道，管道会**按顺序**执行每个压缩器。
# 组合压缩:几种方式组合
"""

from langchain_chroma.vectorstores import Chroma

from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

from advanced_rag.models import get_ali_clients


#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 80}\n".join(
            [f"Document {i+1}: {str(len(d.page_content))} \n\n" + d.page_content[:50] for i, d in enumerate(docs)]
        )
    )

documents = TextLoader("../data/deepseek百度百科.txt",encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,  
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

#使用基础检索器
retriever = Chroma.from_documents(texts, embeddings_model).as_retriever()

query = "deepseek的发展历程"
docs = retriever.invoke(query)
print("-------------------压缩前--------------------------")
pretty_print_docs(docs)

# print('=' * 20 + " 第一种：LLMChainExtractor 压缩 " '=' * 20)
# 使用上下文压缩检索器
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# compressed_docs = compression_retriever.invoke(query)
# print('-' * 15 + " LLMChainExtractor 压缩后 " + '-' * 15)
# pretty_print_docs(compressed_docs)


# print('=' * 20 + " 第二种：LLMChainFilter 压缩 " + '=' * 20)
# #LLMChainFilter 是稍微简单但更强大的压缩器
# _filter = LLMChainFilter.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=_filter,
#     base_retriever=retriever
# )
#
# compressed_docs = compression_retriever.invoke(query)
# pretty_print_docs(compressed_docs)


print('=' * 20 + " 第三种：EmbeddingsFilte " + '=' * 20)
#对每个检索到的文档进行额外的 LLM 调用既昂贵又缓慢。
#EmbeddingsFilter 通过嵌入文档和查询并仅返回那些与查询具有足够相似嵌入的文档来提供更便宜且更快的选项
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
vectorstore1 = Chroma.from_texts(
    texts=texts,
    embedding= embeddings_model
)
retriever1 = vectorstore1.as_retriever()

# docs1 = retriever.invoke("人工智能的应用？")
# print("-------------------压缩前--------------------------")
# pretty_print_docs(docs1)
#
# print("-------------------压缩后--------------------------")
# embeddings_filter = EmbeddingsFilter(
#     embeddings=embeddings_model,
#     similarity_threshold=0.69)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=embeddings_filter,
#     base_retriever=retriever1
# )
#
# compressed_docs = compression_retriever.invoke("人工智能的应用？")
# pretty_print_docs(compressed_docs)


print('=' * 20 + " 第四种：组合压缩 " + '=' * 20)
# DocumentCompressorPipeline轻松地按顺序组合多个压缩器
'''首先TextSplitters可以用作文档转换器，将文档分割成更小的块，
然后EmbeddingsRedundantFilter 根据文档之间嵌入的相似性来过滤掉冗余文档，
该过滤操作以文本的嵌入向量为依据，也就是借助余弦相似度来衡量文本之间的相似程度，
进而判定是否存在冗余，它会把文本列表转化成对应的嵌入向量，然后计算每对文本之间的余弦相似度。
一旦相似度超出设定的阈值，就会将其中一个文本判定为冗余并过滤掉。
最后 EmbeddingsFilter 根据与查询的相关性进行过滤。'''
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model)
relevant_filter = EmbeddingsFilter(
    embeddings=embeddings_model,
    similarity_threshold=0.6)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "deepseek的发展历程"
)
pretty_print_docs(compressed_docs)