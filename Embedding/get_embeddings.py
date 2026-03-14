
# 向量 和 Embeddings.py
#   如何访问嵌入模型以及输出文本的向量表示

from models import *

# 创建 client: 默认通义千问
#   通义千问
client_qwen = OpenAI(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
                     base_url=ALI_TONGYI_URL)

# 嵌入模型
def get_embeddings(texts, model, dimensions=1024):
    #  texts: 是一个包含要获取嵌入表示的文本的列表，
    #  model: 则是用来指定要使用的模型的名称
    #  生成文本的嵌入表示。结果存储在data中。
    data = client_qwen.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    print(data)
    print("-" * 100)

    # 返回了一个包含所有嵌入表示的列表
    return [x.embedding for x in data]


test_query = ["我爱你",
              "由此上溯到一千八百四十年，从那时起，为了反对内外敌人，争取民族独立和人民自由幸福，在历次斗争中牺牲的人民英雄们永垂不朽！"]


# vec = get_embeddings(test_query, model=ALI_TONGYI_EMBEDDING_V3, dimensions=64)
vec = get_embeddings(test_query, model=ALI_TONGYI_EMBEDDING_V4)
# vec = get_embeddings(test_query, model=ALI_TONGYI_EMBEDDING_V4, dimensions=512)
print(vec)
print("=" * 100)
#
# #  "我爱你" 文本的嵌入表示。
print(vec[0])
#
# #  "我爱你" 文本的嵌入表示的维度
print("第1句话的维度:", len(vec[0]))
# #  "由此上溯到...." 文本的嵌入表示的维度
print("第2句话的维度:", len(vec[1]))

