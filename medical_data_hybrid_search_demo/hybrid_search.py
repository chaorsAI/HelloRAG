
# 实现了一个混合搜索系统，结合了 BM25 和 密集向量检索两种方法
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
import json
import chromadb
from chromadb.config import Settings
import csv

from models import *

instructions = []
outputs = []

# 1、读取文件准备处理
# with open('../Data/medical_train.json', 'r', encoding='utf-8') as f:
#     data = [json.loads(line) for line in f.readlines()]
#
# #   instruction -> 症状描述；
# #   output -> 症状解释或治疗方案
# print(len(data))
# instructions = [entry['instruction'] for entry in data]
# outputs = [entry['output'] for entry in data]
# print("instructions[0]:", instructions[0])
# print("outputs[0]:", outputs[0])
# print('-' * 100)

with open('../Data/medical_train.csv', 'r', encoding='utf-8') as f:
    # 创建reader对象
    csv_reader = csv.reader(f)
    # 可选：如果第一行是表头，跳过它
    next(csv_reader)

    for row in csv_reader:
        # 假设 instruction 在第0列，output 在第1列
        instructions.append(row[0])
        outputs.append(row[3])

print(f"数据量: {len(instructions)}")
print("instructions[0]:", instructions[0])
print("outputs[0]:", outputs[0])
print('-' * 100)

# ---------------------------------------------------------
# 2、BM25进行全文检索
def bm25_search(query):
    # 在运用 BM25 算法进行全文检索时，需要对文档进行分词，以此把文本拆分成一个个独立的词语，方便后续计算词语在文档中的频率等统计信息
    # 文档分词 jieba.lcut(doc)  函数会把 instructions 列表里的每个文档 doc 进行分词
    tokenized_corpus = [jieba.lcut(doc) for doc in instructions]
    # print("tokenized_corpus:", tokenized_corpus)

    # 初始化一个BM25Okapi对象，用于基于BM25算法的文本检索或相似度计算
    # 对传入的文档计算必要的统计信息
    bm25 = BM25Okapi(tokenized_corpus)
    # 问题分词 :对查询的问题也需要进行分词处理，这样才能计算查询词和文档的相似度分数。
    tokenized_query = jieba.lcut(query)

    # 通过BM25算法计算查询词与文档的相似度分数
    bm25_scores = bm25.get_scores(tokenized_query)
    # 通过BM25算法获取与查询最相关的前3个结果
    bm25_results = bm25.get_top_n(tokenized_query, outputs, n=3)
    # print("BM25 Score: ", bm25_scores)
    # print("BM25 Results: ", bm25_results)

    # BM25分数归一化到[0,1]区间
    #   用数组中的 （每个元素-最小值）/（最大值-最小值），实现将分数缩放到 [0, 1] 区间的目的。
    #   例如:[1, 2, 3, 4, 5] 归一化后的结果是：[0, 0.25, 0.5, 0.75, 1]
    #
    #   使用 np.array() 函数把 bm25_scores 转换为 NumPy 数组。
    #   bm25_scores 原本可能是 Python 列表，转换为 NumPy 数组后，能更方便地进行数值计算，因为 NumPy 提供了很多高效的数组操作函数。
    bm25_scores = np.array(bm25_scores)
    max_score = bm25_scores.max()  # 最高分数
    min_score = bm25_scores.min()  # 最低分数
    bm25_scores_normalized = (bm25_scores - min_score) / (max_score - min_score)
    # print("bm25_scores_normalized:", bm25_scores_normalized)
    print('-' * 100)

    return bm25_scores_normalized



#---------------------------------------------------------
# 封装向量数据库（ChromaDB）
class MyVectorDBConnector:
    # collection_name：向量数据库中集合的名称。
    def __init__(self, collection_name):
        # 初始化 ChromaDB 客户端 并重置数据库
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 创建一个 集合 collection 在向量数据库中，集合是存储向量数据以及相关元数据的容器
        #get_or_create_collection 方法用于获取或创建一个集合，如果集合不存在则创建一个新集合。
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        # 定义一个函数，用于将文本转换为向量表示，并返回一个包含向量表示的列表。

        self.client = get_normal_client()


    # 封装向量模型与API的交互操作，通过自定义函数 get_embeddings 提供向量模型的调用。
    def get_embeddings(self, texts, model=ALI_TONGYI_EMBEDDING_V2):
        data = self.client.embeddings.create(input=texts, model=model).data
        return [x.embedding for x in data]

    # get_embeddings函数的变体版，因为各个模型对一次能处理的文本条数有限制且每个平台不一致，新增一个batch_size参数用以控制。
    def get_embeddings_batch(self,texts, model=ALI_TONGYI_EMBEDDING_V2, batch_size=10):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]
            data = self.client.embeddings.create(input=batch_text, model=model).data
            all_embeddings.extend([x.embedding for x in data])
        return all_embeddings

    # 添加文档与向量
    def add_documents(self, instructions, outputs):
        '''向 collection 中添加文档与向量'''
        embeddings = self.get_embeddings_batch(instructions)
        # 向 collection 中添加文档与向量
        self.collection.add(
            embeddings=embeddings,  # 每个文档的向量
            documents=outputs,  # 文档的原文
            ids=[f"id{i}" for i in range(len(instructions))]  # 每个文档的 id
        )

        # print(self.collection.count())

    # 定义检索函数， 在向量数据库里进行检索操作
    def search(self, query, top_n):
        '''检索向量数据库'''
        # self.collection.query() 这是 ChromaDB 集合对象的一个方法，用于在集合中执行查询操作。
        results = self.collection.query(
            #  query_embeddings是查询文本的向量表示
            # 调用在类初始化时传入的嵌入函数 self.embedding_fn，把查询文本 query 转换为向量。
            # 要注意的是，期望接收一个字符串列表作为输入，所以这里把 query 放在列表 [query] 里。
            query_embeddings=self.get_embeddings_batch([query]),
            # 指定要返回的最相似文档的数量。
            n_results=top_n
        )
        # 返回检索结果  results 是一个字典，其中包含了和查询向量最相似的 top_n 个文档的相关信息，像文档的原文、向量、ID 等。
        return results

'''
# vector_db = MyVectorDBConnector("demo")
# 向向量数据库中添加文档
# vector_db.add_documents(instructions, outputs)
# 根据问题进行检索，返回最相似的 top_n 个结果
# results = vector_db.search(query, 3)
# # 打印最相似的 top_n 个结果
# print("vector_db(documents):", results['documents'][0])
# # 打印最相似的 top_n 个分数(距离)
# print("vector_db(distances):", results['distances'][0])
# print('-' * 100, '*' * 100, '-' * 100)
'''

# 3、向量相似度检索
def vector_search(query):
    # 创建一个向量数据库对象
    vector_db = MyVectorDBConnector("demo")

    query_embedding = np.array(vector_db.get_embeddings_batch(query))  # 获取查询的向量表示，并把结果转换为 NumPy 数组
    doc_embeddings = np.array(vector_db.get_embeddings_batch(instructions))  # 获取文档的向量表示，并把结果转换为 NumPy 数组
    # print("query_embedding:", query_embedding)
    # print("doc_embeddings:", doc_embeddings)

    # 计算查询向量和文档向量之间的欧氏距离
    # np.linalg.norm 函数用于计算向量的范数，这里计算的是向量差的 L2 范数，即欧氏距离。axis=1 表示按第二个维度计算。
    vector_scores = np.linalg.norm(query_embedding - doc_embeddings, axis=1)
    # print("vector_scores:", vector_scores)

    # 将距离转换为相似度分数并归一化到[0,1]区间
    #   将欧氏距离转换为相似度分数，并将其归一化到 [0, 1] 区间。
    max_score = np.max(vector_scores)
    min_score = np.min(vector_scores)
    # 因为欧氏距离越小，相似度越高。但是bm25的分数是值越大，相似度越高。
    # 所以用 1 减去归一化后的距离得到相似度分数。和bm25度量方式进行统一
    vector_scores_normalized = 1 - (vector_scores - min_score) / (max_score - min_score)
    print("vector_scores_normalized:", vector_scores_normalized)
    print('-' * 100)

    return vector_scores_normalized



#---------------------------------------------------------
# 4、混合检索：组合BM25和词向量相似度检索的结果
def hybrid_search(query, top_k=3, bm25_weight=0.5):
    bm25_scores_normalized = bm25_search(query)  # 得到的BM25分数的归一化结果
    vector_scores_normalized = vector_search(query)  # 向量相似度分数的归一化结果。

    # 将两种方法的分数进行加权组合：
    #   权重均为 0.5。这样可以综合考虑两种方法的优点，得到更准确的文档相关性评分。
    combined_scores = bm25_weight * bm25_scores_normalized + (1-bm25_weight) * vector_scores_normalized
    print('combined_scores:', combined_scores)

    # 根据组合分数对结果排序并返回前3个最相关的文档
    # 对 combined_scores 数组中的值进行降序排序，并返回排序后的索引值
    # argsort：默认升序；[::-1]：倒序
    top_index = combined_scores.argsort()[::-1]
    # print("top_index:", top_index)
    print("top_index[:top_k]:", top_index[:top_k])

    # 输出混合搜索的结果: 最相关的文档outputs
    hybrid_results = [outputs[i]  for i in top_index[:top_k]]
    # hybrid_results = np.array(outputs)[top_index[:top_k]]

    return hybrid_results


if __name__ == '__main__':
    # 查询的问题
    query = "扁桃体发炎怎么办"

    # 混合检索
    hybrid_results = hybrid_search(query, top_k=3, bm25_weight=0.5)
    print("Hybrid Search Results: \n")
    for st in hybrid_results:
        print(st)
        print("-" * 100)

