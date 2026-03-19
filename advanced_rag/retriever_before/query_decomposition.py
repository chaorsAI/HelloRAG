# query_decomposition.py
# 预检索-查询-问题分解优化。
# - 核心：解决复杂、多步骤或隐含多子问题的查询时，因单一查询向量与知识库匹配度低而导致的检索失败或答案质量低下的问题。
# - 本质: **”让系统像人类专家一样思考“**
"""
- 核心流程：
1. **问题拆解**
    -   **输入**：用户的原始复杂查询（例如：“比较iPhone 15和华为Mate 60的摄像头和电池续航”）。

    -   **处理**：LLM根据预设的Prompt，将问题分解为多个语义独立、可检索的子问题。例如：
        -   子问题A：“iPhone 15的摄像头参数和评测如何？”
        -   子问题B：“华为Mate 60的摄像头参数和评测如何？”
        -   子问题C：“iPhone 15的电池续航表现如何？”

    -   **输出**：一组结构化的子查询列表，为后续检索提供明确的输入。

2. **并行 vs. 串行：两种执行策略**

-   **并行策略**：当子问题之间**没有强依赖关系**，可以独立解答时，采用并行处理。

-   **串行策略**：当子问题之间存在**逻辑递进或依赖关系**，前一个问题的答案是后一个问题的基础时，采用串行处理。
"""

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate,BasePromptTemplate
from langchain_core.callbacks import  CallbackManagerForRetrieverRun
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import Runnable,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from advanced_rag.models import  get_ali_clients

#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# 模拟文档
documents = [
    Document(page_content="西红柿炒蛋的食材：\n\n- 新鲜鸡蛋：3-4个（根据人数调整）\n- 西红柿：2-3个中等大小\n- 盐：适量\n- 白糖：一小勺（可选，用于提鲜）\n- 食用油：适量\n- 葱花：少许（可选，用于增香）\n\n这些是最基本的材料，当然也可以根据个人口味添加其他调料或配料。"),
    Document(page_content="西红柿炒蛋的步骤：鸡蛋打入碗中，加入少许盐，用筷子或打蛋器充分搅拌均匀；\n   - 西红柿洗净后切成小块备用。\n\n3. **炒鸡蛋**：锅内倒入适量食用油加热至温热状态，然后将搅拌好的鸡蛋液缓缓倒入锅中。待鸡蛋凝固时轻轻翻动几下，让其受热均匀直至完全熟透，随后盛出备用。\n\n4. **炒西红柿**：在同一锅里留下的底油中放入切好的西红柿块，中小火慢慢翻炒至出汁，可根据个人口味加一点点白糖提鲜。\n\n5. **合炒**：当西红柿炒至软烂并开始释放大量汤汁时，再把之前炒好的鸡蛋倒回锅里，快速与西红柿混合均匀，同时加入适量的盐调味。如果喜欢的话还可以撒上一些葱花增加香气。\n\n6. **完成**：最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的西红柿炒蛋啦！"),
    Document(page_content="技巧与注意事项：1. **选材**：选择新鲜的鸡蛋和成熟的西红柿。新鲜的食材是做好这道菜的基础。\n2. **打蛋液**：将鸡蛋打入碗中后加入少许盐（根据个人口味调整），然后充分搅拌均匀。这样做可以让蛋更加松软且味道更佳。\n3. **处理西红柿**：西红柿最好先用开水稍微焯一下皮，然后去皮切块。这样可以去除表皮的硬质部分，让西红柿更容易入味，并且口感更好。\n4. **热锅冷油**：先用中小火把锅烧热，再倒入适量食用油，待油温五成热时下蛋液。这样的做法可以使蛋快速凝固形成漂亮的形状而不易粘锅。\n5. **分步烹饪**：通常建议先炒鸡蛋至半熟状态取出备用；接着利用剩下的底油继续翻炒西红柿至出汁，最后再将之前炒好的鸡蛋倒回锅里与西红柿混合均匀加热即可。\n6. **调味品**：除了基本的盐之外，还可以根据喜好添加少量糖来提鲜或者一点酱油增色添香。注意调味料不宜过多以免掩盖了食材本身的味道。\n7. **出锅前加葱花**：如果喜欢的话，在即将完成时撒上一些葱花不仅能增加菜品色泽还能增添香气。")
]

vectorstore = Chroma.from_documents(documents=documents,
                                    embedding=embeddings_model,
                                    collection_name="decomposition")

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

print("============ 直接回答（不使用问题拆解） ============")
# 从实际来说，该问题的答案应该包括原材料、步骤，以及注意事项才算完整
print(retriever.invoke("新手如何制作西红柿炒蛋？"))
pretty_print_docs(retriever.invoke("新手如何制作西红柿炒蛋？"))


# print("============ 问题拆解 ============")
template = """
你是一名AI语言模型助理。你的任务是将输入问题分解成4个子问题，通过一个个解决这些子问题从而解决完整的问题。
            子问题需要在矢量数据库中检索相关文档。通过分解用户问题生成子问题，你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。
            请提供这些用换行符分隔的子问题本身，不需要额外内容。
            原始问题: {question}
            """
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=template,
)

# print("-------------测试大模型对问题的拆解，实际业务中可不用--------------")
# chain = DEFAULT_QUERY_PROMPT | llm | StrOutputParser()
# result = chain.invoke({"question":"新手如何制作西红柿炒蛋？"})
# print("问题拆解：",result)
# print("-------------完成测试大模型对问题的拆解--------------")
# exit()


DEFAULT_SUB_QUESTION_PROMPT = PromptTemplate(
    input_variables=["question", "sub_question", "documents"],
    template="""
    要解决主要问题{question}，需要先解决子问题{sub_question}。
    以下是为支持您的推理而提供的参考文档：{documents}。请直接给出当前子问题的答案。不需要额外内容。
    """,
)

# 字符串按"\n"分割得到列表
def str2List(text:str) -> List[str]:
    # 移除首尾空白，按换行分割，过滤空行
    lines = [line.strip() for line in text.strip().split("\n")]
    return [line for line in lines if line]

#自定义一个检索器，将对子问题的生成、获得子问题的答案组合起来，通过组合简化使用过程
class DecompositionQueryRetriever(BaseRetriever):
    # 向量数据库检索器
    retriever: BaseRetriever
    # 生成子问题链
    make_sub_chain: Runnable
    # 解决子问题链
    resolve_sub_chain: Runnable

    @classmethod
    def from_llm(
            cls,
            retriever: BaseRetriever,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
            sub_prompt: BasePromptTemplate = DEFAULT_SUB_QUESTION_PROMPT
    ) -> "DecompositionQueryRetriever":
        output_parser = StrOutputParser()
        # make_sub_chain = prompt | llm | output_parser
        # resolve_sub_chain = sub_prompt | llm
        return cls(
            retriever=retriever,
            make_sub_chain=prompt | llm | output_parser,
            resolve_sub_chain=sub_prompt | llm
        )

    #生成子问题
    def generate_queries(self, question: str) -> List[str]:
        response = self.make_sub_chain.invoke({"question": question})
        lines = str2List(response)
        print(f"生成子问题: {lines}")
        return lines

    # 获得子问题答案
    def retrieve_documents(self, query: str, sub_queries: List[str]) -> List[Document]:
        sub_llm_chain = RunnableLambda(
            # 传入子问题，检索文档并回答
            lambda sub_query: self.resolve_sub_chain.invoke(
                {
                    "question": query,
                    "sub_question": sub_query,
                    "documents": [doc.page_content for doc in self.retriever.invoke(sub_query)]
                }
            )
        )
        # 批量执行所有的子问题
        responses = sub_llm_chain.batch(sub_queries)
        # 将子问题和答案合并作为解决主问题的文档
        documents = [
            Document(page_content=sub_query + "\n" + response.content)
            for sub_query, response in zip(sub_queries, responses)
        ]
        return documents

    # 核心执行函数
    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 生成子问题
        sub_queries = self.generate_queries(query)
        # 解决子问题
        documents = self.retrieve_documents(query, sub_queries)
        return documents


print("============ DecompositionQueryRetriever ============")
decompositionQueryRetriever = DecompositionQueryRetriever.from_llm(
    llm=llm,
    retriever=retriever
)
decomposition_docs = decompositionQueryRetriever.invoke("西红柿炒蛋怎么制作？")
print("-------------检索到的文档（拆解后）--------------")
pretty_print_docs(decomposition_docs)

# 创建prompt模板
template = """请根据以下文档回答问题:
### 文档:
{context}
### 问题:
{question}
"""
# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm

print("-------------回答--------------")
question = "新手如何制作西红柿炒蛋？"
response = chain.invoke({"context": [doc.page_content for doc in decomposition_docs], "question": question})
print(response.content)