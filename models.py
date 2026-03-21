# models.py
# 可用模型列表，以及获得访问模型的客户端
# 实际使用时可以根据自己的实际情况调整

# ---------- 模型与平台配置常量 (合并自文档1和文档2，以文档2为基础扩展) ----------
# 阿里的通义千问大模型
# 官网: https://bailian.console.aliyun.com/#/home
ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
# 文档1中独有的Turbo模型常量予以保留
ALI_TONGYI_TURBO_MODEL = "qwen-turbo-latest"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qwq-plus"  # 采用文档2的更新值
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v2"  # 采用文档2的命名
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"   # 从文档2引入：后检索-重排序模型

# DeepSeek
# 官网：https://platform.deepseek.com/api_keys
# 采用文档2的环境变量名，更常见
DEEPSEEK_API_KEY_OS_VAR_NAME = "Deepseek_Key"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"

# 腾讯混元
# 官网：https://hunyuan.cloud.tencent.com/#/app/modelSquare
TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME = "HUNYUAN_API_KEY"
TENCENT_HUNYUAN_URL = "https://api.hunyuan.cloud.tencent.com/v1"
TENCENT_HUNYUAN_TURBO_MODEL = "hunyuan-turbos-latest"
TENCENT_HUNYUAN_REASONER_MODEL = "hunyuan-t1-latest"
TENCENT_HUNYUAN_LONGCONTEXT_MODEL = "hunyuan-large-longcontext"
TENCENT_HUNYUAN_EMBEDDING_MODEL = "hunyuan-embedding"
TENCENT_SECRET_ID_OS_VAR_NAME = "Tencent_SecretId"
TENCENT_SECRET_KEY_OS_VAR_NAME = "Tencent_SecretKey"

# 百川 (从文档2引入)
BAICHUAN_API_KEY_OS_VAR_NAME = "Baichuan_API_Key"
BAICHUAN_EMBEDDING_MODEL = "Baichuan-Text-Embedding"
BAICHUAN_EMBEDDING_URL = "https://api.baichuan-ai.com/v1/embeddings"

# LangSmith (从文档2引入，用于应用监控和测试)
LANGSMITH_API_KEY_OS_VAR_NAME = "LANGSMITH_API_KEY"
LANGSMITH_API_URL = "https://api.smith.langchain.com"

# ---------- 依赖导入 ----------
import os
import inspect
from langchain_openai import ChatOpenAI
from openai import OpenAI  # 保留文档1的原生OpenAI客户端支持
from langchain_community.embeddings import DashScopeEmbeddings, HunyuanEmbeddings, BaichuanTextEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
from langsmith import Client

# ---------- 核心客户端工厂函数 ----------
def get_normal_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
                      base_url=ALI_TONGYI_URL,
                      verbose=False, debug=False):
    """
    使用原生OpenAI SDK获得指定平台的客户端，默认平台为阿里云百炼。
    适用于需要更精细控制或使用LangChain未封装特性的场景。
    （此函数保留自文档1，文档2中无对应实现）
    """
    function_name = inspect.currentframe().f_code.co_name
    if verbose:
        print(f"{function_name}-平台：{base_url}")
    if debug:
        print(f"{function_name}-平台：{base_url}, key：{api_key[:8]}...")  # 安全起见，key不完整打印
    return OpenAI(api_key=api_key, base_url=base_url)

def get_lc_model_client(api_key=os.getenv(TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME),
                        base_url=TENCENT_HUNYUAN_URL,
                        model=TENCENT_HUNYUAN_TURBO_MODEL,
                        temperature=0.7, verbose=False, debug=False):
    """
    通过LangChain获得指定平台和模型的客户端。默认连接腾讯混元平台。
    此为通用工厂函数，优先采用文档2的实现，因其去除了厂商特定参数，兼容性更好。
    """
    function_name = inspect.currentframe().f_code.co_name
    if verbose:
        print(f"{function_name}-平台：{base_url}, 模型：{model}, 温度：{temperature}")
    if debug:
        print(f"{function_name}-平台：{base_url}, 模型：{model}, 温度：{temperature}, key：{api_key[:8]}...")
    # 采用文档2的实现，不再包含`extra_body`等可能造成兼容性问题的参数
    return ChatOpenAI(api_key=api_key,
                      base_url=base_url,
                      model=model,
                      temperature=temperature)

def get_ali_model_client(model=ALI_TONGYI_MAX_MODEL, temperature=0.7, verbose=False, debug=False):
    """通过LangChain获得阿里大模型的客户端，默认使用qwen-max-latest。"""
    return get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
                               base_url=ALI_TONGYI_URL,
                               model=model,
                               temperature=temperature,
                               verbose=verbose,
                               debug=debug)

def get_ds_model_client(model=DEEPSEEK_CHAT_MODEL, temperature=0.7, verbose=False, debug=False):
    """通过LangChain获得DeepSeek大模型的客户端，默认使用deepseek-chat。"""
    return get_lc_model_client(api_key=os.getenv(DEEPSEEK_API_KEY_OS_VAR_NAME),
                               base_url=DEEPSEEK_URL,
                               model=model,
                               temperature=temperature,
                               verbose=verbose,
                               debug=debug)

# ---------- 嵌入模型客户端函数 ----------
def get_ali_embeddings():
    """通过LangChain获得一个阿里通义千问嵌入模型的实例，默认使用text-embedding-v2。"""
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL,
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    )

def get_tencent_embeddings():
    """通过LangChain获得一个腾讯混元嵌入模型的实例。"""
    return HunyuanEmbeddings(
        hunyuan_secret_id=os.getenv(TENCENT_SECRET_ID_OS_VAR_NAME),
        hunyuan_secret_key=os.getenv(TENCENT_SECRET_KEY_OS_VAR_NAME),
        region="ap-guangzhou",
    )

def get_baichuan_embeddings():
    """
    通过LangChain获得一个百川嵌入模型的实例。
    注意：百川嵌入模型服务限流严重，可能有10~20%的概率访问报错。
    """
    return BaichuanTextEmbeddings(
        api_key=os.getenv(BAICHUAN_API_KEY_OS_VAR_NAME)
    )

# ---------- 组合客户端函数 (方便常用组合) ----------
def get_ali_clients():
    """返回一个元组：(阿里大模型客户端, 阿里嵌入模型客户端)。"""
    return get_ali_model_client(), get_ali_embeddings()

def get_tencent_clients():
    """返回一个元组：(腾讯混元大模型客户端, 腾讯嵌入模型客户端)。"""
    return get_lc_model_client(), get_tencent_embeddings()

def get_a_t_mix_clients():
    """返回一个元组：(阿里大模型客户端, 腾讯嵌入模型客户端)。用于混合实验。"""
    return get_ali_model_client(), get_tencent_embeddings()

# ---------- 其他工具客户端 (从文档2引入) ----------
def get_ali_rerank(top_n=3):
    """通过LangChain获得一个阿里重排序(Rerank)模型的实例，用于提升RAG检索精度。"""
    return DashScopeRerank(
        model=ALI_TONGYI_RERANK_MODEL,
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        top_n=top_n
    )

def get_langsmith_client():
    """获取LangSmith客户端，用于AI链的追踪、监控和测试。"""
    return Client(
        api_key=os.getenv(LANGSMITH_API_KEY_OS_VAR_NAME),
        api_url=LANGSMITH_API_URL,
    )