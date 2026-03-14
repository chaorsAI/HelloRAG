
# RAG_Ollama.py
#    本地部署模型后，访问嵌入模型以及输出文本的向量表示

from ollama import Client

# 创建Ollama客户端实例，本机服务
client = Client(host="http://127.0.0.1:11434")
# 或者更简单地使用默认值
client = Client() # 默认就是连接 http://127.0.0.1:11434

# 获取模型列表并打印端口和访问链接
models = client.list()
for model in models:
    print(model)

print('-' * 100)


def get_embedding(text, model_name="bge-m3"):
    # 使用ollama库获取嵌入向量
    response = client.embed(model_name, text)
    embedding = response['embeddings']
    return embedding


test_query = "我爱你"
# test_query = ["我爱你", 'hello我是Jeff，I am a goodman']

vec = get_embedding(test_query)
print(vec)
#  "我爱你" 文本的嵌入表示的维度。
print("维度:",len(vec))
print("维度:",len(vec[0]))
print('-' * 100)


# 方式2
'''
import ollama
def get_embedding(text, model_name="bge-m3"):
    # 使用ollama库获取嵌入向量
    response = ollama.embed(model_name, text)
    embedding = response['embeddings']
    return embedding

test_query = "我爱你"
vec = get_embedding(test_query)
print(vec[0])
#  "我爱你" 文本的嵌入表示的维度。
print("维度:",len(vec[0]))

'''
