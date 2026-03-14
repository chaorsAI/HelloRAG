"""
================================================================================
智识通 - 企业智能知识库系统
RAG核心逻辑文件 (main.py)

【什么是RAG】
RAG = Retrieval-Augmented Generation（检索增强生成）
简单说就是：先搜索相关资料，再让AI根据资料回答问题

【RAG的工作流程】
1. 用户上传文档 -> 文档被切分成小块 -> 存入向量数据库
2. 用户提问 -> 在数据库中搜索相关内容 -> 把相关内容+问题一起给AI -> AI生成答案

【这个文件的作用】
1. 处理文档上传：读取文档、切分、存入向量库
2. 处理用户提问：检索相关内容、构建提示词、调用AI生成答案

【依赖安装】
pip install python-docx    # 读取Word文档
pip install pypinyin       # 中文转拼音
================================================================================
"""

# 从function_tools.py导入所有工具函数和类
# 包括：向量数据库类、文档读取函数、AI调用函数等
from utils import *
from MyPDF2Text import extract_text_from_pdf

# =============================================================================
# 第一部分：文档处理（上传文档到知识库）
# =============================================================================

# 创建一个向量数据库连接对象
# 这个对象会在整个程序运行期间一直存在，负责和向量数据库交互
# 向量数据库的作用：把文字变成向量（数字），方便快速搜索相似内容
vector_db = MyVectorDBConnector()


# @to_pinyin 是一个装饰器，作用是把中文文件名转成拼音
# 因为有些数据库不支持中文名称，用拼音更保险
@to_pinyin
def save_to_db(filepath, collection_name='demo'):
    """
    【功能】将文档存入向量数据库
    
    【参数】
        filepath: 文档的完整路径，比如 "uploads/合同.docx"
        collection_name: 集合名称（知识库的名字），默认是'demo'
    
    【处理流程】
        1. 读取文档内容
        2. 将文档切分成小块（方便检索）
        3. 将小块存入向量数据库
    
    【什么是集合（collection）】
        可以把集合想象成一个"文件夹"，每个上传的文档创建一个文件夹
        查询时就在指定的文件夹里搜索
    """
    # 打印分隔线，让日志更清晰
    print('-' * 100)
    print('【文档上传】文件路径:', filepath)
    print('【文档上传】集合名称:', collection_name)
    
    # documents变量用来存储文档内容
    documents = ''

    # 根据文件扩展名判断文件类型，调用对应的读取函数
    # 目前只支持Word文档（.docx 或 .doc）
    if filepath.endswith('.docx') or filepath.endswith('.doc'):
        # 调用extract_text_from_docx函数读取Word文档
        # 这个函数在 utils.py中定义
        documents = extract_text_from_docx(filepath)

    elif filepath.endswith('.pdf'):
        # 调用 extract_text_from_pdf 函数读取Word文档
        # 这个函数在 MyPDF2Text.py 中定义
        documents = extract_text_from_pdf(filepath)

        # extract_text_from_pdf返回的documents列表元素为Document，add_documents入参documents要求其元素是字符串
        formatted_docs = []
        for doc in documents:
            content = getattr(doc, "page_content")
            formatted_docs.append(content)
        documents = formatted_docs

    # 检查是否成功读取到内容
    if not documents:
        print('【错误】读取文件内容为空')
        return '读取文件内容为空'


    # 将文档内容存入向量数据库
    # vector_db.add_documents() 是向量数据库类的方法
    # 参数1: documents - 文档内容列表（已经被切分成小块）
    # 参数2: collection_name - 要存入哪个集合
    vector_db.add_documents(documents, collection_name=collection_name)
    
    print(f'【成功】文档已存入向量数据库，共 {len(documents)} 个片段')


# =============================================================================
# 第二部分：智能问答（RAG核心流程）
# =============================================================================

@to_pinyin
def rag_chat(user_query, collection_name='demo', n_results=5):
    """
    【功能】RAG智能问答
    
    【参数】
        user_query: 用户的问题，比如 "公司的年假有多少天？"
        collection_name: 在哪个知识库中搜索，默认是'demo'
        n_results: 返回最相关的几条结果，默认是5条
    
    【返回值】
        response: AI生成的答案
        search_results: 检索到的相关文档片段列表
    
    【RAG流程详解】
        ┌─────────────────────────────────────────────────────────┐
        │  第1步: 检索（Retrieval）                                │
        │  - 把用户问题转成向量                                    │
        │  - 在向量数据库中搜索最相似的文档片段                      │
        │  - 返回最相关的n_results条结果                           │
        └─────────────────────────────────────────────────────────┘
                                ↓
        ┌─────────────────────────────────────────────────────────┐
        │  第2步: 增强（Augmentation）                             │
        │  - 把检索到的文档片段整合成一个"已知信息"                 │
        │  - 构建完整的Prompt（提示词）                            │
        │  - Prompt告诉AI：你是谁、你有什么资料、用户问了什么        │
        └─────────────────────────────────────────────────────────┘
                                ↓
        ┌─────────────────────────────────────────────────────────┐
        │  第3步: 生成（Generation）                               │
        │  - 把Prompt发给大语言模型（如通义千问）                    │
        │  - AI根据提供的资料生成答案                              │
        │  - 返回生成的答案                                        │
        └─────────────────────────────────────────────────────────┘
    """
    
    print('=' * 100)
    print('【RAG问答】用户问题:', user_query)
    print('【RAG问答】搜索集合:', collection_name)
    print('=' * 100)
    
    # -------------------------------------------------------------------------
    # 第1步：检索知识库
    # -------------------------------------------------------------------------
    print('\n>>> 第1步：在向量数据库中检索相关内容...')
    
    # 调用向量数据库的search方法进行搜索
    # 参数1: user_query - 用户的问题
    # 参数2: collection_name - 在哪个集合中搜索
    # 参数3: n_results - 返回最相关的几条结果
    search_results = vector_db.search(
        user_query, 
        collection_name=collection_name, 
        n_results=n_results
    )
    
    # search_results是一个字典，包含多个字段：
    # - documents: 检索到的文档内容列表
    # - ids: 文档的ID列表
    # - distances: 相似度距离列表（越小越相似）
    # - metadatas: 元数据列表
    
    # 获取检索到的文档内容（取第一个列表，因为query只有一个）
    retrieved_docs = search_results['documents'][0]
    
    print(f'检索完成，找到 {len(retrieved_docs)} 条相关内容')
    print('-' * 100)

    # -------------------------------------------------------------------------
    # 第2步：构建增强Prompt
    # -------------------------------------------------------------------------
    print('\n>>> 第2步：构建Prompt（提示词）...')
    
    # 将检索到的文档片段用换行符连接成一个字符串
    # 这就是提供给AI的"已知信息"
    info = '\n'.join(retrieved_docs)
    
    # 构建完整的Prompt
    # f"""...""" 是Python的f-string，可以在字符串中嵌入变量
    prompt = f"""
你是一个专业的企业知识库问答助手。
你的任务是根据下述给定的已知信息回答用户问题。
请确保你的回复完全依据下述已知信息，不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"根据现有资料，我无法回答您的问题"。

【已知信息】:
{info}

================================================================================
【用户问题】：
{user_query}
================================================================================

请用中文回答用户问题，回答要简洁明了。
"""
    
    # 打印Prompt（用于调试，可以看到发给AI的完整内容）
    print('Prompt构建完成:')
    print(prompt)
    print('-' * 100)

    # -------------------------------------------------------------------------
    # 第3步：调用大语言模型生成答案
    # -------------------------------------------------------------------------
    print('\n>>> 第3步：调用AI模型生成答案...')
    
    # 调用get_completion函数，把Prompt发给AI
    # 这个函数在function_tools.py中定义
    response = get_completion(prompt)
    
    print('AI回答生成完成')
    print('=' * 100)
    
    # 返回两个值：
    # 1. AI生成的答案
    # 2. 检索到的原始文档片段（用于展示给用户，增加可信度）
    return response, retrieved_docs


# =============================================================================
# 第三部分：测试代码
# =============================================================================
# 当直接运行这个文件时（不是被导入时），执行以下测试代码
if __name__ == '__main__':
    
    print("=" * 100)
    print("开始测试 RAG 系统")
    print("=" * 100)
    
    # ------ 测试1：上传文档 ------
    print("\n【测试1】上传文档到知识库...")
    
    # 调用save_to_db函数，将测试文档存入向量数据库
    # filepath: 文档路径（..表示上级目录）
    # collection_name: 用文件名作为集合名
    # save_to_db(
    #     filepath='../data/人事管理流程.docx',
    #     collection_name='人事管理流程.docx'
    # )
    save_to_db(
        filepath='../data/公司采购流程.pdf',
        collection_name='公司采购流程.pdf'
    )

    print('-' * 100)
    
    # ------ 测试2：智能问答 ------
    print("\n【测试2】测试智能问答功能...")
    
    # 定义测试问题
    # user_query = "你好啊，后天我要入职了。"
    # user_query = "紧急采购流程怎么走？。"
    user_query = "我想下周采购一台家用彩电？。"
    print(f"测试问题: {user_query}")
    
    # 调用rag_chat函数获取答案
    response, search_results = rag_chat(
        user_query, 
        # collection_name='人事管理流程.docx',
        collection_name='公司采购流程.pdf',
        n_results=5
    )
    
    # 打印最终结果
    print("\n" + "=" * 100)
    print("【最终答案】")
    print("=" * 100)
    print(response)
    print("=" * 100)
