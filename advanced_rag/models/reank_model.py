from langchain_core.documents import Document

from models import get_ali_rerank

reranker = get_ali_rerank()
query = "孕妇感冒了怎么办"

documents = [
    "感冒应该吃999感冒灵",
    "高血压患者感冒了吃什么",
    "感冒了可以吃感康，但是孕妇禁用",
    "感冒了可以咨询专业医生"
]

scores = reranker.rerank(documents,query)
print(scores)

documents = [
    Document(
        page_content="感冒应该吃999感冒灵",
        metadata={"source": "999感冒灵"},
    ),
    Document(
        page_content="高血压患者感冒了吃什么",
        metadata={"source": "高血压患者"},
    ),
    Document(
        page_content="感冒了可以吃感康，但是孕妇禁用",
        metadata={"source": "感康"},
    ),
    Document(
        page_content="感冒了可以咨询专业医生",
        metadata={"source": "专业建议"},
    ),
]
scores = reranker.compress_documents(documents, query)
print(scores)
