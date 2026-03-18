# index_metadata_field_SelfQueryRetriever.py
# # 预检索-索引优化-元数据索引优：通过为文档附加“标签”来缩小检索范围、提升精度的核心技术
# SelfQueryRetriever：LangChain中可以让大模型（LLM）自己分析用户提问，并自动生成数据库查询指令的智能查询器。
# 在Langchain1.0之后通过langchain_classic导入！！！
""""
- 核心目标：解决朴素 RAG 仅靠语义相似度检索时 **“找不准”和“找得慢”** 的问题。
- **本质**：预筛过滤器，不参与予以匹配
- 作用：在向量数据库执行昂贵的向量相似度计算**之前**，先通过元数据快速圈定一个小的候选集，从而**降低计算量**并**排除无关噪声**。
- 核心流程：
    - 定义元数据标签，如果文档本身没有，可以利用大模型推理出输入问题的元数据
    - 通过标签先对文档进行过滤
    - 结合向量检索进一步定位到最相关的前 K 个知识块
"""


from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

from advanced_rag.models.models import get_ali_clients

#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

# 加载文档：构造示例文档
docs = [
    Document(
        page_content="作者A团队开发出基于人工智能的自动驾驶决策系统，在复杂路况下的响应速度提升300%",
        metadata={"year": 2024, "rating": 9.2, "genre": "AI", "author": "A"},
    ),
    Document(
        page_content="区块链技术成功应用于跨境贸易结算，作者B主导的项目实现交易确认时间从3天缩短至30分钟",
        metadata={"year": 2023, "rating": 9.8, "genre": "区块链", "author": "B"},
    ),
    Document(
        page_content="云计算平台实现量子计算模拟突破，作者C构建的新型混合云架构支持百万级并发计算",
        metadata={"year": 2022, "rating": 8.6, "genre": "云", "author": "C"},
    ),
    Document(
        page_content="大数据分析预测2024年全球经济趋势，作者A团队构建的模型准确率超92%",
        metadata={"year": 2023, "rating": 8.9, "genre": "大数据", "author": "A"},
    ),
    Document(
        page_content="人工智能病理诊断系统在胃癌筛查中达到三甲医院专家水平，作者B获医疗科技创新奖",
        metadata={"year": 2024, "rating": 7.1, "genre": "AI", "author": "B"},
    ),
    Document(
        page_content="基于区块链的数字身份认证系统落地20省市，作者C设计的新型加密协议通过国家级安全认证",
        metadata={"year": 2022, "rating": 8.7, "genre": "区块链", "author": "C"},
    ),
    Document(
        page_content="云计算资源调度算法重大突破，作者A研发的智能调度器使数据中心能效提升40%",
        metadata={"year": 2023, "rating": 8.5, "genre": "云", "author": "A"},
    ),
    Document(
        page_content="大数据驱动城市交通优化系统上线，作者B团队实现早晚高峰通行效率提升25%",
        metadata={"year": 2024, "rating": 7.4, "genre": "大数据", "author": "B"},
    )
]

vectorstore = Chroma.from_documents(docs, embeddings_model)

# 文档内容描述（指导LLM理解文档内容）
document_content_description = "技术文章简述"


# 元数据字段定义（指导LLM如何解析查询条件）
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="文章的技术领域，选项:['AI '，'区块链'，'云'，'大数据']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="文章的出版年份",
        type="integer",
    ),
    AttributeInfo(
        name="author",
        description="署名文章的作者姓名",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="技术价值评估得分（1-10分）",
        type="float"
    )
]

# 创建自查询检索器（核心组件）
'''SelfQueryRetriever 是 langchain 库中的一个工具，其主要功能是把自然语言查询转变为结构化查询，
以此提升检索的精准度。它整合了检索器和语言模型，能依据查询内容自动推断出筛选条件，还能识别出相关的元数据字段。'''
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    # enable_limit=True, 限定只返回一个结果
)


# # ============ 测试代码 - 元数据索引 start ============
# print("============ 测试代码 - 元数据索引 ============")
# print("---------------------评分在9分以上的文章-------------------------------")
# #查询条件：查询只约束分数 rating>9
# print(retriever.invoke("我想了解评分在9分以上的文章"))
#
# print("---------------------作者B在2023年发布的文章-------------------------------")
# # 第二个查询只约束作者和年份 author="B", year=2023
# print(retriever.invoke("作者B在2023年发布的文章"))
# exit()
# # ============ 测试代码 - 元数据索引 start ============


# ============ 测试代码 - 原理探究-内部工作机制 start ============
print("============ 测试代码 - 原理探究-内部工作机制 ============")
# 原理：构建查询解析器（分析内部工作机制用）
'''构建查询提示模板
document_content_description：对文档内容的概括性描述，例如 "有关各种主题的文章"。
metadata_field_info：元数据字段的详细描述，涵盖字段名称、类型以及描述。
此函数会生成一个提示模板，其用途是指导语言模型如何将自然语言查询转换为结构化查询。'''
prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
#解析器的作用是把语言模型的输出转换为 StructuredQuery 对象，
# 这个对象包含了 query（文本查询）和 filter（元数据筛选条件）
output_parser = StructuredQueryOutputParser.from_components()
#链式操作，先将用户查询填入提示模板，接着由语言模型生成结构化输出，最后通过解析器得到结构化查询。
query_constructor = prompt | llm | output_parser

# 打印查询构造提示
print("------------ 提示词显示 ------------")
print(prompt.format(query="我想了解评分在9分以上的文章"))

# 打印结构化查询的结果
print("------------ 结构化查询结果 ------------")
print(query_constructor.invoke(
    {
        "query": "作者B在2023年发布的文章"
    }
))
# ============ 测试代码 - 原理探究-内部工作机制 end ============
