# compare_summary_vs_naive.py
# 这是一个用来对比Naive RAG和摘要索引的演示代码


import os
import sys
import shutil
import time
import uuid
from pydantic import BaseModel, Field

from langchain_chroma import Chroma

from langchain_core.stores import InMemoryByteStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_openai import ChatOpenAI,OpenAIEmbeddings

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore

from models import get_ali_clients, get_ali_model_client, get_ali_embeddings
from chunking import *


# 获得访问大模型和嵌入模型客户端
# 使用阿里百炼的模型
client, embeddings_model = get_ali_clients()

# ==================== 配置常量 ====================
NAIVE_PERSIST_DIR = "./chroma_db_naive"
SUMMARY_PERSIST_DIR = "./chroma_db_summary"
DOCSTORE_DIR = "./docstore_files"
COLLECTION_NAIVE = "naive_docs"
COLLECTION_SUMMARY = "summaries"

# 标准答案库 (Ground Truth)
GROUND_TRUTH = {
    "坎地沙坦酯片的规格是多少？": "本品的规格为 8mg。",
    "本品的贮藏条件是什么？": "常温（10~30℃）保存。请将本品放在儿童不能接触的地方。",
    "说明书中提到，发生率在 0.1% 到 5% 之间的消化系统不良反应有哪些？": "包括恶心、呕吐、食欲不振、胃部不适、剑下疼痛、腹泻、口腔炎。",
    "哪些人群需要调整起始剂量为4mg？": "一般成人 1 日 1 次，起始剂量通常为 4~8mg。此外，老年原发性高血压患者、伴有肾功能障碍或肝功能障碍的高血压患者在研究中有单次服用 4mg 的记录。",
    "对于哪些特定的疾病患者或人群，该药是严格禁止同时服用阿利吉仑的？": "糖尿病患者或中至重度肾损伤（GFR < 60ml/min/1.73m²）患者禁止同时服用阿利吉仑。",
    "请列出说明书中所有明确要求'应从小剂量开始服用'的情形及其原因。": "1. 肝功能障碍（可能恶化/清除率降低）；2. 严重肾功能障碍（防止过度降压使肾功能恶化）；3. 血液透析、严格限盐、服用利尿降压药、低钠血症、肾功能障碍或心衰患者（防止血压急剧下降导致休克）。",
    "如果患者正在进行血液透析，服用此药时在剂量控制上有什么具体要求？": "应从较低的剂量开始服用；如有必要增加剂量，应密切观察并缓慢进行。坎地沙坦不能通过血液透析清除。",
    "一旦发现怀孕，应该如何处理？为什么？": "应立即停止使用。直接作用于RAAS系统的药物可能造成发育期胚胎损伤甚至死亡，引起胎儿及新生儿肾衰、发育不良等。",
    "药物过量时应该采取什么紧急措施？": "对症治疗及监控生命体征。患者仰卧并抬高双腿。若效果不显，应输液增加血浆容量，或服用拟交感神经药。",
    "一位65岁男性患者，有双侧肾动脉狭窄病史，正在服用利尿剂，能否使用本品？如果可以，起始剂量和监测要求是什么？": "原则上应尽量避免。若必需使用，因正在服用利尿剂必须从小剂量开始，且需密切观察患者的血压和肾功能。",
    "对比本品与ACEI类药物在禁忌症和不良反应方面的异同点。": "相同点：对孕妇禁用、可能引起血钾/肌酐升高。不同点：本品不抑制激肽酶 II，不影响缓激肽降解，通常不会引起 ACEI 类常见的干咳副作用。"
}


# 检查索引是否存在
def check_index_exists(persist_dir, collection_name):
    """检查向量索引是否存在且有效"""
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return False, None

    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings_model,
            collection_name=collection_name
        )
        count = vectorstore._collection.count()
        if count > 0:
            return True, vectorstore
        return False, None
    except Exception as e:
        print(f"检查索引失败: {e}")
        return False, None

# 创建向量索引
def build_naive_index():
    """建立Naive RAG索引"""
    print("  [Naive] 正在提取文档并建立索引...")
    # 固定分块
    docs = fixed_size_chunking("./Data/坎地沙坦酯片.pdf")
    print(f"  [Naive] 文档分割完成，共 {len(docs)} 个块")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings_model,
        persist_directory=NAIVE_PERSIST_DIR,
        collection_name=COLLECTION_NAIVE
    )
    print(f"  [Naive] ✓ 索引已建立")
    return vectorstore

# 获取向量存储
def get_naive_vectorstore(force_rebuild=False):
    """获取Naive向量存储（加载或新建）"""
    if not force_rebuild:
        exists, vs = check_index_exists(NAIVE_PERSIST_DIR, COLLECTION_NAIVE)
        if exists:
            print("  [Naive] 加载已有索引")
            return vs

    if force_rebuild:
        print("  [Naive] 强制重建索引...")
    else:
        print("  [Naive] 索引不存在，创建新索引...")

    return build_naive_index()

# 创建摘要索引
def build_summary_index():
    """建立摘要索引（包含摘要生成）"""
    # 固定分块
    docs = fixed_size_chunking("./Data/坎地沙坦酯片.pdf")

    print(f"  [Summary] 文档分割完成，共 {len(docs)} 个块")

    # 生成摘要
    summary_prompt = ChatPromptTemplate.from_template(
        """你是一位专业的医药数据分析师。请对以下药品说明书文档块进行精炼总结。
        要求：
        1. 保留所有关键医学实体（药品名称、剂量、症状、疾病、禁忌症、不良反应、用法用量等）。
        2. 提取核心信息和关键数值数据。
        3. 保留表格中的重要数据（如发生率、剂量范围、条件限制等）。
        4. 保持语义完整性，便于向量检索准确匹配。
        5. 字数控制在200字以内，确保信息密度高。

        文档内容：
        {doc}"""
    )

    chain = (
            {"doc": lambda x: x.page_content}
            | summary_prompt
            | client
            | StrOutputParser()
    )

    print("  [Summary] 正在生成文档摘要，请耐心等待...")
    start_time = time.time()
    summaries = chain.batch(docs, {"max_concurrency": 5})
    print(f"  [Summary] ✓ 摘要生成完成，共 {len(summaries)} 条，耗时 {time.time() - start_time:.2f}s")

    # 创建存储
    vectorstore = Chroma(
        collection_name=COLLECTION_SUMMARY,
        embedding_function=embeddings_model,
        persist_directory=SUMMARY_PERSIST_DIR
    )

    # 清理并创建文档存储目录
    if os.path.exists(DOCSTORE_DIR):
        shutil.rmtree(DOCSTORE_DIR)
    os.makedirs(DOCSTORE_DIR, exist_ok=True)

    docstore = LocalFileStore(DOCSTORE_DIR)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=docstore,
        id_key="doc_id",
    )

    # 生成ID并保存
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    summary_docs = [
        Document(page_content=s, metadata={"doc_id": doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    print("  [Summary] 正在保存到数据库...")
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    print(f"  [Summary] ✓ 索引已建立")

    return retriever

# 获取摘要检索器
def get_summary_retriever(force_rebuild=False):
    """获取摘要检索器（加载或新建）"""
    store_exists = os.path.exists(DOCSTORE_DIR) and os.listdir(DOCSTORE_DIR)
    vector_exists, vectorstore = check_index_exists(SUMMARY_PERSIST_DIR, COLLECTION_SUMMARY)

    if not force_rebuild and vector_exists and store_exists:
        print("  [Summary] 加载已有索引")
        docstore = LocalFileStore(DOCSTORE_DIR)
        return MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=docstore,
            id_key="doc_id",
        )

    if force_rebuild:
        print("  [Summary] 强制重建索引...")
    else:
        print("  [Summary] 索引不完整，创建新索引...")

    return build_summary_index()

# Naive RAG查询
def naive_rag_query(query, vectorstore):
    """使用已有vectorstore执行Naive RAG查询"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "根据下面的文档回答问题:\n\n{context}\n\n问题: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | client
            | StrOutputParser()
    )

    start_time = time.time()
    answer = chain.invoke(query)
    retrieve_time = time.time() - start_time

    retrieved_docs = retriever.invoke(query)
    return answer, retrieved_docs, retrieve_time

# 摘要索引查询
def summary_rag_query(query, retriever):
    """使用已有retriever执行摘要索引查询"""
    prompt = ChatPromptTemplate.from_template(
        "根据下面的文档回答问题:\n\n{context}\n\n问题: {question}"
    )

    chain = RunnableParallel({
        "context": lambda x: retriever.invoke(x["question"], search_kwargs={"k": 3}),
        "question": lambda x: x["question"]
    }) | prompt | client | StrOutputParser()

    start_time = time.time()
    answer = chain.invoke({"question": query})
    retrieve_time = time.time() - start_time

    retrieved_docs = retriever.invoke(query, search_kwargs={"k": 3})

    # 去重：确保返回的文档数量符合预期
    unique_docs = []
    seen_content = set()
    for doc in retrieved_docs:
        content_key = doc.page_content[:100]  # 使用前100字符作为去重依据
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_docs.append(doc)

    return answer, unique_docs, retrieve_time

# 质量评估
# 数据模型定义
class QualityEvaluation(BaseModel):
    reasoning: str = Field(..., description="审计过程：先列出参考答案的关键要点，再检查模型回答是否覆盖、是否误导")
    accuracy: float = Field(..., description="准确性：基准10分。每出现一个事实错误扣3分，编造数值扣5分。")
    completeness: float = Field(..., description="完整性：基准10分。每遗漏一个参考要点扣2分。")
    safety: float = Field(..., description="安全性：基准10分。漏掉禁忌、过量处置或存储警告扣5分。")
    # 不再输出 comment，统一由 reasoning 承载逻辑


def evaluate_quality(query, response):
    """
    通用型医药 RAG 审计评估函数
    采用：关键信息点(KPI)核对法 + 强约束扣分制
    """
    try:
        reference = GROUND_TRUTH.get(query, "无标准答案")

        # 预判断：如果模型直接说没找到，但标准答案有，直接给低分
        if any(kw in response for kw in ["未找到", "没找到", "没有提及"]) and len(reference) > 3:
            return "0.0|0.0|5.0" # 安全性给5分是因为至少它没瞎编

        eval_prompt = ChatPromptTemplate.from_template(
            """你是一位专业的医药文档合规审计员。
            请根据【参考答案】对【模型回答】进行严格审计。

            ### 第一步：提取参考答案的关键信息点 (KPIs)
            从参考答案中列出所有核心事实（如：剂量、症状、禁忌、存储条件）。

            ### 第二步：逐项核对并扣分
            1. 准确性 (Accuracy): 
               - 基础10分。
               - 关键数值、药品名、核心结论错误：直接扣 10 分。
               - 描述性文字细微偏差：每处扣 2 分。
            2. 完整性 (Completeness): 
               - 基础10分。
               - 遗漏参考答案中的一个核心 KPI：扣 4 分。
               - 遗漏辅助性说明：扣 2 分。
            3. 安全性 (Safety): 
               - 基础10分。
               - 涉及“用法、用量、禁忌、不良反应、存储”的信息，若模型漏掉或写错：直接扣 10 分。
               - 不涉及安全风险的问题：默认 10 分。

            ---
            参考答案: {reference}
            模型回答: {response}
            ---

            请严格按以下 JSON 格式输出，不要包含任何额外文字：
            {{
                "kpi_analysis": "简述提取了哪些KPI，哪些命中了，哪些漏了",
                "accuracy": 0.0,
                "completeness": 0.0,
                "safety": 0.0
            }}
            """
        )

        # 锁定 temperature 为 0 保证稳定性
        eval_client = client.with_config({"temperature": 0})
        output_parser = JsonOutputParser()
        eval_chain = eval_prompt | eval_client | output_parser

        result = eval_chain.invoke({"reference": reference, "response": response})

        # 获取分数并确保是 float
        acc = float(result.get('accuracy', 0))
        comp = float(result.get('completeness', 0))
        safe = float(result.get('safety', 0))

        # 逻辑兜底：数值容错（针对规格、剂量等关键数字的硬核二次校验）
        import re
        ref_nums = set(re.findall(r'\d+', reference))
        res_nums = set(re.findall(r'\d+', response))
        # 如果参考答案有数字，但模型回答里的数字不匹配（且模型不是完全没数字）
        if ref_nums and res_nums and not (ref_nums & res_nums):
            acc = min(acc, 2.0) # 强制压低准确性

        return f"{acc}|{comp}|{safe}"
    except Exception as e:
        return "0.0|0.0|0.0"

from typing import List, Dict
# 打印对比结果
def print_comparison(query, q_type, naive_result, summary_result, test_idx, total_tests):
    """打印对比结果"""
    naive_answer, naive_docs, naive_time = naive_result
    summary_answer, summary_docs, summary_time = summary_result

    # 获取预设的标准答案
    reference = GROUND_TRUTH.get(query, "暂无标准答案")

    # 获取质量评分字符串
    n_eval = evaluate_quality(query, naive_answer)
    s_eval = evaluate_quality(query, summary_answer)

    # 解析评分字符串
    def get_scores(eval_str):
        try:
            parts = eval_str.split('|')
            return parts[0], parts[1], parts[2]  # 返回 准确, 完整, 安全
        except:
            return "0", "0", "0"

    # 获取准确性, 完整性, 安全性评分
    n_acc, n_comp, n_safe = get_scores(n_eval)
    s_acc, s_comp, s_safe = get_scores(s_eval)

    print("\n" + "=" * 80)
    print(f"测试 {test_idx}/{total_tests} - 【{q_type}】")
    print(f"【问   题】{query}")
    print(f"【参考答案】{reference}")
    print("=" * 80)

    # Naive RAG 结果
    print(f"\n{'─' * 60}")
    print(f"【方法1: Naive RAG】")
    print(f"\n{'─' * 60}")
    print(f"答案: {naive_answer[:300]}..." if len(naive_answer) > 300 else f"答案: {naive_answer}")

    # 摘要索引结果
    print(f"\n{'─' * 60}")
    print(f"【方法2: 摘要索引】")
    print(f"{'─' * 60}")
    print(f"答案: {summary_answer[:300]}..." if len(summary_answer) > 300 else f"答案: {summary_answer}")

    # # 列出检索的文档
    # print(f"\n{'─' * 60}")
    # print("【检索文档对比】")
    # print(f"{'─' * 60}")
    #
    # # Naive RAG 检索的文档
    # print(f"\n【Naive RAG 检索的文档】")
    # for i, doc in enumerate(naive_docs, 1):
    #     doc_content = doc.page_content[:100]
    #     print(f"文档 {i}: {doc_content}...")
    #
    # # 摘要索引 检索的文档
    # print(f"\n【摘要索引 检索的文档】")
    # for i, doc in enumerate(summary_docs, 1):
    #     doc_content = doc.page_content[:100]
    #     print(f"文档 {i}: {doc_content}...")
    #
    # # 计算相同文档数量（基于前100个字）
    # naive_contents = {d.page_content[:100] for d in naive_docs}
    # summary_contents = {d.page_content[:100] for d in summary_docs}
    # common_docs = naive_contents & summary_contents
    # common_count = len(common_docs)
    #
    # print(f"\n【相同文档数量】: {common_count} 个")
    # if common_count > 0:
    #     print("相同的文档内容:")
    #     for i, content in enumerate(common_docs, 1):
    #         print(f"{i}. {content}...")
    #

    # 对比分析
    print(f"\n{'─' * 60}")
    print("【对比分析】")
    print(f"{'─' * 60}")

    # 打印 Markdown 表格格式
    print(f"|{'检索方法':^5}|{'准确性':^5}|{'完整性':^5}|{'安全性':^5}|{'耗时':^5}|")
    print(f"|{'Naive RAG':^5}|{n_acc:^5}| {n_comp:^5}| {n_safe:^5}|{naive_time:^5.2f}s|")
    print(f"|{'摘要索引':^5}|{s_acc:^5}| {s_comp:^5}| {s_safe:^5}|{summary_time:^5.2f}s|")
    print("-" * 50)

# 生成最终报告
def generate_final_report(results: List[Dict]):
    """生成最终汇总对比报告（含二维表格与优势分析）"""
    if not results:
        print("无测试结果")
        return

    total_tests = len(results)
    # 初始化统计数据
    metrics = {
        "Naive RAG": {"acc": [], "comp": [], "safe": [], "time": []},
        "摘要索引": {"acc": [], "comp": [], "safe": [], "time": []}
    }

    for r in results:
        def parse_scores(eval_str):
            try:
                parts = eval_str.split('|')
                return [float(p) if p != 'N/A' else 0.0 for p in parts[:3]]
            except:
                return [0.0, 0.0, 0.0]

        n_s = parse_scores(r['naive_quality'])
        s_s = parse_scores(r['summary_quality'])

        # 记录 Naive RAG 数据
        metrics["Naive RAG"]["acc"].append(n_s[0])
        metrics["Naive RAG"]["comp"].append(n_s[1])
        metrics["Naive RAG"]["safe"].append(n_s[2])
        metrics["Naive RAG"]["time"].append(r['naive_time'])

        # 记录 摘要索引 数据
        metrics["摘要索引"]["acc"].append(s_s[0])
        metrics["摘要索引"]["comp"].append(s_s[1])
        metrics["摘要索引"]["safe"].append(s_s[2])
        metrics["摘要索引"]["time"].append(r['summary_time'])

    # 计算平均值
    stats = {}
    for method, data in metrics.items():
        stats[method] = {
            "avg_acc": sum(data["acc"]) / total_tests,
            "avg_comp": sum(data["comp"]) / total_tests,
            "avg_safe": sum(data["safe"]) / total_tests,
            "avg_time": sum(data["time"]) / total_tests,
        }
        stats[method]["avg_total"] = (stats[method]["avg_acc"] + stats[method]["avg_comp"] + stats[method][
            "avg_safe"]) / 3

    # 1. 打印二维对比表格
    print("\n" + "=" * 80)
    print("### 最终对比报告 (汇总)")
    print("=" * 80)
    print("| 检索方法 | 准确性 | 完整性 | 安全性 | 平均总分 | 平均耗时 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for method in ["Naive RAG", "摘要索引"]:
        s = stats[method]
        print(
            f"| **{method}** | {s['avg_acc']:.2f} | {s['avg_comp']:.2f} | {s['avg_safe']:.2f} | {s['avg_total']:.2f} | {s['avg_time']:.2f}s |")
    print("=" * 80)

    # 2. 自动总结优势部分
    print("\n【摘要索引核心优势分析】")

    # 速度优势判断
    time_diff = stats["Naive RAG"]["avg_time"] - stats["摘要索引"]["avg_time"]
    if time_diff > 0:
        print(
            f"1. 速度优势: 显著 (平均快 {time_diff:.2f}s，提升约 {(time_diff / stats['Naive RAG']['avg_time'] * 100):.1f}%)")
    else:
        print(f"1. 速度优势: 无明显差异")

    # 质量优势判断
    quality_diff = stats["摘要索引"]["avg_total"] - stats["Naive RAG"]["avg_total"]
    if quality_diff > 1.0:
        print(f"2. 质量优势: 显著 (总分领先 {quality_diff:.2f} 分，摘要索引更具全局理解力)")
    elif quality_diff > 0:
        print(f"2. 质量优势: 轻微领先")
    else:
        print(f"2. 质量优势: Naive RAG 在当前测试集表现更优")

    # 3. 适用场景建议
    print("\n【架构建议与适用场景】")
    print(f"✓ **摘要索引 (Summary Index)**: 推荐用于“{results[4]['q_type']}”等需要跨章节关联、数值筛选的复杂医药问答。")
    print(f"✓ **Naive RAG**: 仅推荐用于极简单的 Key-Value 型事实检索（如单点规格查询）。")
    print("=" * 80)

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 解析参数
    rebuild_naive = "--rebuild-naive" in sys.argv or "--rebuild-all" in sys.argv
    rebuild_summary = "--rebuild-summary" in sys.argv or "--rebuild-all" in sys.argv

    print("=" * 120)
    print("RAG 方法对比测试 - 摘要索引 vs Naive RAG")
    print("=" * 120)
    print(f"Naive索引: {'强制重建' if rebuild_naive else '自动检测'}")
    print(f"摘要索引: {'强制重建' if rebuild_summary else '自动检测'}")
    print("参数说明: --rebuild-naive | --rebuild-summary | --rebuild-all")
    print("=" * 120)

    # 预加载/建立索引
    print("\n>>> 阶段1: 初始化索引")
    naive_vs = get_naive_vectorstore(force_rebuild=rebuild_naive)
    summary_retriever = get_summary_retriever(force_rebuild=rebuild_summary)

    # 测试问题
    test_cases = [
        # ========== 基础事实检索（简单）==========
        (
            "坎地沙坦酯片的规格是多少？",
            "基础事实检索（考察精确信息定位）"
        ),
        (
            "本品的贮藏条件是什么？",
            "基础事实检索（考察属性查询）"
        ),

        # ========== 条件筛选类（中等）==========
        (
            "说明书中提到，发生率在 0.1% 到 5% 之间的消化系统不良反应有哪些？",
            "数值区间筛选（考察表格理解与范围匹配）"
        ),
        (
            "哪些人群需要调整起始剂量为4mg？",
            "条件触发检索（考察多条件筛选）"
        ),

        # ========== 跨章节关联（较难）==========
        (
            "对于哪些特定的疾病患者或人群，该药是严格禁止同时服用阿利吉仑的？",
            "跨章节逻辑关联（禁忌+药物相互作用）"
        ),
        (
            "请列出说明书中所有明确要求'应从小剂量开始服用'的情形及其原因。",
            "跨章节逻辑汇总（考察全局视野）"
        ),
        (
            "如果患者正在进行血液透析，服用此药时在剂量控制上有什么具体要求？",
            "条件触发+跨章节（用法用量+注意事项+药代动力学）"
        ),

        # ========== 极端风险与处置（关键信息）==========
        (
            "一旦发现怀孕，应该如何处理？为什么？",
            "极端风险处置（考察关键信息捕捉）"
        ),
        (
            "药物过量时应该采取什么紧急措施？",
            "应急处置流程（考察操作步骤完整性）"
        ),

        # ========== 推理与综合（最难）==========
        (
            "一位65岁男性患者，有双侧肾动脉狭窄病史，正在服用利尿剂，能否使用本品？如果可以，起始剂量和监测要求是什么？",
            "多条件综合推理（禁忌+注意事项+用法用量）"
        ),
        (
            "对比本品与ACEI类药物在禁忌症和不良反应方面的异同点。",
            "对比分析（考察知识整合与推理）"
        ),
    ]

    # 执行对比测试
    print("\n>>> 阶段2: 执行对比测试")
    print("测试类型: 基础事实检索、条件筛选、跨章节关联、极端风险处置、综合推理")
    print(f"总测试数: {len(test_cases)}")

    # 存储测试结果
    test_results = []

    for idx, (query, q_type) in enumerate(test_cases, 1):
        print(f"\n\n{'#' * 120}")
        print(f"测试 {idx}/{len(test_cases)}")

        # Naive RAG
        naive_result = naive_rag_query(query, naive_vs)

        # 摘要索引
        summary_result = summary_rag_query(query, summary_retriever)

        # 打印对比
        print_comparison(query, q_type, naive_result, summary_result, idx, len(test_cases))

        # 存储结果
        naive_answer, naive_docs, naive_time = naive_result
        summary_answer, summary_docs, summary_time = summary_result
        
        # 质量评估
        n_eval = evaluate_quality(query, naive_answer)
        s_eval = evaluate_quality(query, summary_answer)
        
        test_results.append({
            "query": query,
            "q_type": q_type,
            "naive_time": naive_time,
            "summary_time": summary_time,
            "naive_quality": n_eval,
            "summary_quality": s_eval
        })

    # 生成最终报告
    generate_final_report(test_results)

    print("\n" + "=" * 120)
    print("所有测试完成！")
    print("=" * 120)