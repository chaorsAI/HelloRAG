
import chromadb  # ChromaDB向量数据库
from chromadb.config import Settings  # ChromaDB配置

import uuid  # 生成唯一ID的工具

from models import *  # 从models.py导入所有模型配置


class MyVectorDBConnector:
    """
    【类名】MyVectorDBConnector

    【作用】封装ChromaDB向量数据库的操作

    【什么是ChromaDB】
    ChromaDB是一个开源的向量数据库，专门用来存储和搜索向量数据。
    它的特点是：
    - 轻量级，可以嵌入到Python程序中
    - 支持持久化（数据保存到文件，重启后还在）
    - 查询速度快

    【核心概念】
    - Collection（集合）：类似于关系数据库中的"表"，每个文档一个集合
    - Document（文档）：存储的文本内容
    - Embedding（向量）：文本的数字表示
    - Query（查询）：搜索相似向量的过程
    """

    def __init__(self):
        """
        【构造函数】创建向量数据库连接对象时自动调用

        【作用】初始化数据库连接
        """
        # 创建ChromaDB客户端
        # PersistentClient表示"持久化客户端"，数据会保存到本地文件
        # path="../chroma" 表示数据保存在一级目录的chroma文件夹中
        self.chroma_client = chromadb.PersistentClient(path="../chroma")

        # 创建AI模型客户端（用于生成向量）
        # get_normal_client() 在models.py中定义，返回OpenAI格式的客户端
        self.client = get_normal_client()

        print("【向量数据库】连接成功，数据保存在 ../chroma 文件夹")

    def get_embeddings(self, texts, model=ALI_TONGYI_EMBEDDING_V4):
        """
        【功能】将文本转换为向量（Embedding）

        【参数】
            texts: 要转换的文本列表，比如 ["你好", "世界"]
            model: 使用的向量化模型，默认是阿里云的text-embedding-v4

        【返回值】
            向量列表，每个文本对应一个向量（一串数字）

        【什么是Embedding】
        把文字变成数字的过程。比如：
        "苹果" -> [0.1, 0.2, 0.3, ...] (几百个数字)
        "香蕉" -> [0.1, 0.25, 0.3, ...] (和苹果很相似)
        "汽车" -> [0.8, 0.1, 0.9, ...] (和水果不相似)

        【注意】
        这个函数一次只能处理少量文本（受API限制）
        大量文本请使用 get_embeddings_batch
        """
        # 调用Embedding API
        # self.client.embeddings.create() 是OpenAI格式的API调用
        # input=texts 传入要转换的文本列表
        # model=model 指定使用的模型
        data = self.client.embeddings.create(input=texts, model=model).data

        # 从返回数据中提取向量
        # x.embedding 是每个文本对应的向量
        return [x.embedding for x in data]

    def get_embeddings_batch(self, texts, model=ALI_TONGYI_EMBEDDING_V4, batch_size=10):
        """
        【功能】批量将文本转换为向量（处理大量文本）

        【参数】
            texts: 要转换的文本列表
            model: 使用的向量化模型
            batch_size: 每批处理多少条，默认10条

        【为什么要分批】
        API一次能处理的文本数量有限制，太多会报错。
        所以我们把大列表分成小批次，分批处理。

        【例子】
        如果有25条文本，batch_size=10：
        - 第1批：0-9条
        - 第2批：10-19条
        - 第3批：20-24条
        """
        all_embeddings = []  # 存储所有向量

        # range(0, len(texts), batch_size) 生成批次起始索引
        # 比如 len=25, batch_size=10，生成：0, 10, 20
        for i in range(0, len(texts), batch_size):
            # 切出当前批次的文本
            # texts[0:10], texts[10:20], texts[20:25]
            batch_text = texts[i:i + batch_size]

            # 调用API转换这一批文本
            data = self.client.embeddings.create(input=batch_text, model=model).data

            # 将结果添加到总列表中
            all_embeddings.extend([x.embedding for x in data])

            print(f"【Embedding】已处理 {min(i + batch_size, len(texts))}/{len(texts)} 条")

        return all_embeddings

    def add_documents(self, documents, collection_name='demo'):
        """
        【功能】向集合中添加文档

        【参数】
            documents: 文档内容列表，每个元素是一段文本
            collection_name: 集合名称（知识库名字）

        【处理流程】
            1. 获取或创建集合
            2. 将文档转换为向量
            3. 将文档、向量、ID一起存入数据库

        【什么是集合（Collection）】
        集合是ChromaDB中的概念，类似于文件夹。
        每个上传的文档创建一个集合，查询时指定在哪个集合中搜索。
        """
        print(f'【向量数据库】正在添加文档到集合: {collection_name}')

        # get_or_create_collection: 获取集合，如果不存在就创建
        collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # collection.add() 添加数据到集合
        # 需要三个参数：
        # 1. embeddings: 每个文档的向量表示
        # 2. documents: 文档的原文
        # 3. ids: 每个文档的唯一标识符
        collection.add(
            embeddings=self.get_embeddings_batch(documents),  # 向量列表
            documents=documents,  # 原文列表
            ids=[str(uuid.uuid4()) for _ in documents]  # 唯一ID列表
        )

        print(f'【向量数据库】成功添加 {len(documents)} 个文档片段')

    def search(self, query, collection_name='demo', n_results=5):
        """
        【功能】在向量数据库中搜索相关内容

        【参数】
            query: 查询文本（用户的问题）
            collection_name: 在哪个集合中搜索
            n_results: 返回最相关的几条结果

        【返回值】
            字典，包含：
            - documents: 检索到的文档内容列表
            - ids: 文档ID列表
            - distances: 相似度距离列表（越小越相似）
            - metadatas: 元数据列表

        【搜索原理】
            1. 把查询文本转成向量
            2. 在向量空间中找最接近的向量
            3. 返回对应的文档内容

        【什么是相似度】
        两个向量之间的距离表示相似度：
        - 距离越小，内容越相似
        - 距离越大，内容越不相关
        """
        print(f'【向量数据库】正在搜索: "{query}"')
        print(f'【向量数据库】集合: {collection_name}, 返回{n_results}条结果')

        # 获取集合
        collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # collection.query() 执行向量相似度搜索
        # query_embeddings: 查询文本的向量（把问题转成向量）
        # n_results: 返回最相似的n条结果
        results = collection.query(
            query_embeddings=self.get_embeddings_batch([query]),
            n_results=n_results
        )

        print(f'【向量数据库】搜索完成，找到 {len(results["documents"][0])} 条结果')

        return results