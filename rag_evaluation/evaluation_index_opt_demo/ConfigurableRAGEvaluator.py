# ConfigurableRAGEvaluator 一个高度可配置的RAG系统评估器。
# 支持参数化控制文本分块、混合检索（BM25+向量）以及结果重排。


import os
from typing import List, Optional, Dict, Any
from datasets import Dataset
from rapidfuzz.fuzz_py import ratio

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.bm25 import BM25Retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from QAEvaluator import QAEvaluator
from models import get_ali_embeddings, get_ali_model_client, get_ali_rerank


class ConfigurableRAGEvaluator:
    # 类级别默认值，方便统一修改
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_BM25_WEIGHT = 0.0
    DEFAULT_USE_RERANK = False
    DEFAULT_USE_COMPRESS = False

    def __init__(self,
                 pdf_path=None,
                 llm_client=get_ali_model_client(),
                 embeddings_model=get_ali_embeddings(),
                 chunk_size:int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap_ratio:float=0.2,
                 bm25_weight: float = DEFAULT_BM25_WEIGHT,
                 use_rerank: bool = DEFAULT_USE_RERANK,
                 use_compress: bool = DEFAULT_USE_COMPRESS,
                 rerank_top_n: int = 3,
                 vector_top_k: int = 10,
                 index_folder: str = "../data/faiss_index"
                 ):
        # 参数验证
        if not 0 <= bm25_weight <= 1:
            raise ValueError("bm25_weight 必须在 [0.0, 1.0] 区间内")
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if chunk_overlap_ratio < 0 or chunk_overlap_ratio >= 1:
            raise ValueError("chunk_overlap_ratio 必须在 [0, 1) 区间内")

        # 核心配置
        self.pdf_path = pdf_path
        self.llm = llm_client or get_ali_model_client(temperature=0)
        self.embeddings = embeddings_model or get_ali_embeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * chunk_overlap_ratio)
        self.bm25_weight = bm25_weight
        self.use_rerank = use_rerank
        self.use_compress = use_compress
        self.rerank_top_n = rerank_top_n
        self.vector_top_k = vector_top_k
        self.index_folder = index_folder
        self.index_name = f"rag_index_c{chunk_size}_b{int(bm25_weight*100)}"
        self.result = None

        # 内部状态
        self._vectorstore: Optional[FAISS] = None
        self._retriever: Optional[BaseRetriever] = None
        self._evaluator: Optional['QAEvaluator'] = None
        self._split_docs = None

        # 初始化核心组件
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._prompt_template = self._create_prompt_template()

    # 创建系统提示词模板
    def _create_prompt_template(self):
        system_prompt = """
        您是问答任务的助理。使用以下的上下文来回答问题，
                上下文：<{context}>
                如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

    # 加载并分割PDF文档
    def _load_and_split_documents(self):
        print(f"[CRAGAsE]-[步骤1] 加载文档: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        raw_docs = loader.load()
        print(f"[CRAGAsE]-  原始文档数: {len(raw_docs)}")

        print(f"[CRAGAsE]-[步骤2] 分割文本 (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        self._split_docs = self._text_splitter.split_documents(raw_docs)
        print(f"[CRAGAsE]-  分块后文档数: {len(self._split_docs)}")
        return self._split_docs

    # 创建或加载FAISS向量存储
    def _create_vector_store(self, force_rebuild=False):
        """创建或加载FAISS向量存储"""
        index_file_path = os.path.join(self.index_folder, f"{self.index_name}.faiss")

        if not force_rebuild and os.path.exists(index_file_path):
            print(f"[CRAGAsE]-[步骤3] 加载现有向量索引: {self.index_name}")
            self._vectorstore = FAISS.load_local(
                self.index_folder,
                self.embeddings,
                self.index_name,
                allow_dangerous_deserialization=True
            )
        else:
            if self._split_docs is None:
                self._load_and_split_documents()
            print(f"[CRAGAsE]-[步骤3] 创建新向量索引: {self.index_name}")
            self._vectorstore = FAISS.from_documents(self._split_docs, self.embeddings)
            self._vectorstore.save_local(self.index_folder, self.index_name)
            print("[CRAGAsE]-  向量化完成并已保存")
        return self._vectorstore

    def _create_retriever(self):
        """
        根据配置创建检索器（纯向量、纯BM25或混合）
        """
        if self._vectorstore is None:
            self._create_vector_store()

        # 1. 向量检索器
        vector_retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.vector_top_k}
        )

        # 2. 纯向量检索
        if self.bm25_weight == 0.0:
            print(f"[CRAGAsE]-[步骤4] 使用纯向量检索 (top_k={self.vector_top_k})")
            if self.use_compress:
                compressor = LLMChainExtractor.from_llm(self.llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=vector_retriever
                )
                self._retriever = compression_retriever
            else:
                self._retriever = vector_retriever
            return self._retriever

        # 3. 准备BM25检索器
        if self._split_docs is None:
            self._load_and_split_documents()
        bm25_retriever = BM25Retriever.from_documents(self._split_docs)
        bm25_retriever.k = self.vector_top_k

        # 4. 纯BM25检索
        if self.bm25_weight == 1.0:
            print(f"[CRAGAsE]-[步骤4] 使用纯BM25检索 (top_k={self.vector_top_k})")
            self._retriever = bm25_retriever
            return self._retriever

        # 5. 混合检索
        print(f"[CRAGAsE]-[步骤4] 使用混合检索 (向量权重={1 - self.bm25_weight:.2f}, BM25权重={self.bm25_weight:.2f})")
        compression_retriever = vector_retriever
        if self.use_compress:
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=vector_retriever
            )
            compression_retriever = compression_retriever
        self._retriever = EnsembleRetriever(
            retrievers=[compression_retriever, bm25_retriever],
            weights=[1 - self.bm25_weight, self.bm25_weight]
        )
        return self._retriever

    def _create_rerank_chain(self, retriever):
        """在检索链中加入重排模型"""
        if not self.use_rerank:
            return create_retrieval_chain(retriever, self._create_document_chain())

        print(f"[CRAGAsE]-[步骤5] 启用结果重排 (top_n={self.rerank_top_n})")

        # 使用您提供的函数
        reranker = get_ali_rerank(top_n=self.rerank_top_n)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever
        )
        return create_retrieval_chain(compression_retriever, self._create_document_chain())

    def _create_document_chain(self):
        """创建文档问答链"""
        return create_stuff_documents_chain(self.llm, self._prompt_template)

    def initialize(self):
        """初始化评估器，创建所有必要的组件"""
        self._create_retriever()
        final_chain = self._create_rerank_chain(self._retriever)
        self._evaluator = QAEvaluator(final_chain)
        print("[CRAGAsE]-[系统] 评估器初始化完成。")
        return self

    def run(
            self,
            questions: List[str],
            ground_truths: List[str],
            verbose: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整的RAG评估流程。
        返回包含答案、上下文和评估结果的字典。
        """
        if self._evaluator is None:
            self.initialize()

        print(f"[CRAGAsE]-\n{'=' * 60}")
        print(f"[CRAGAsE]-开始评估 [chunk_size={self.chunk_size}, bm25_weight={self.bm25_weight}, rerank={self.use_rerank}]")
        print(f"[CRAGAsE]-{'=' * 60}")

        # 生成答案
        answers, contexts = self._evaluator.generate_answers(questions)

        # RAGAS评估
        evaluate_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        vllm = LangchainLLMWrapper(self.llm)
        vllm_e = LangchainEmbeddingsWrapper(self.embeddings)

        evaluation_result = evaluate(
            evaluate_dataset,
            llm=vllm,
            embeddings=vllm_e,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )

        # 提取指标分数
        evaluation_dict = {}
        scores = evaluation_result.scores[0]
        for metric in scores:
            evaluation_dict[metric] = scores[metric]

        # 计算F1分数
        f1_score = self._calc_f1_score(evaluation_result)

        self.result = {
            "config": {
                "chunk_size": self.chunk_size,
                "bm25_weight": self.bm25_weight,
                "use_rerank": self.use_rerank,
                "vector_top_k": self.vector_top_k,
            },
            "questions": questions,
            "answers": answers,
            "contexts": contexts,
            "ragas_metrics": evaluation_dict,
            "f1_score": f1_score,
        }

        if verbose:
            print(f"\n评估结果: {dict(evaluation_result)}")
            print(f"综合F1分数: {f1_score:.4f}")
            print(f"{'=' * 60}")

        return self.result

    # 格式化打印评估结果
    def format_evaluation_result(self, show_details: bool = False) -> str:
        """
        格式化输出RAG评估结果

        Args:
            result_summary: 评估结果字典，包含config、questions、answers、contexts、ragas_metrics、f1_score
            show_details: 是否显示详细的问题和答案内容
        """
        output_lines = []

        # 1. 添加分隔线和标题
        output_lines.append("=" * 60)
        output_lines.append("RAG 系统评估")
        output_lines.append("=" * 60)

        result_summary : dict = self.result
        if not result_summary:
            output_lines.append("\n---- 未产生评估结果，请先进行评估...")
            return "\n".join(output_lines)

        # 2. 配置信息
        config = result_summary.get("config", {})
        output_lines.append("\n---- 配置参数:")
        config_str = f"  chunk_size={config.get('chunk_size', 'N/A')}"
        config_str += f", bm25_weight={config.get('bm25_weight', 0.0)}"
        config_str += f", use_rerank={config.get('use_rerank', False)}"
        config_str += f", vector_top_k={config.get('vector_top_k', 10)}"
        output_lines.append(config_str)

        # 3. 问题列表
        questions = result_summary.get("questions", [])
        output_lines.append(f"\n---- 评估问题 (共{len(questions)}个):")
        for i, q in enumerate(questions, 1):
            # 截断过长的文本
            display_q = q[:50] + "..." if len(q) > 50 else q
            output_lines.append(f"  {i}. {display_q}")

        # 4. 详细问答内容（可选）
        if show_details and "answers" in result_summary and "contexts" in result_summary:
            answers = result_summary["answers"]
            contexts = result_summary["contexts"]
            output_lines.append("\n---- 详细问答结果:")
            for i, (q, a) in enumerate(zip(questions, answers), 1):
                output_lines.append(f"\n  【问题 {i}】: {q[:80]}...")
                output_lines.append(f"  【答案 {i}】: {a[:100]}...")
                if i <= len(contexts):
                    context_count = len(contexts[i - 1])
                    output_lines.append(f"  【上下文 {i}】: 检索到{context_count}个相关片段")

        # 5. RAGAS评估指标
        ragas_metrics = result_summary.get("ragas_metrics", {})
        output_lines.append("\n---- RAGAS 评估指标:")

        # 定义要显示的指标及其中文名称
        metric_names = {
            "context_precision": "上下文精度",
            "context_recall": "上下文召回率",
            "faithfulness": "答案忠实度",
            "answer_relevancy": "答案相关性"
        }

        for metric_key, display_name in metric_names.items():
            if metric_key in ragas_metrics:
                values = ragas_metrics[metric_key]
                if isinstance(values, list) and values:
                    # 计算平均值
                    avg_value = sum(values) / len(values)
                    output_lines.append(f"  {display_name}: {avg_value:.4f}")
                else:
                    output_lines.append(f"  {display_name}: {values}")
            else:
                output_lines.append(f"  {display_name}: 未计算")

        # 6. F1分数和其他汇总信息
        f1_score = result_summary.get("f1_score", 0.0)
        output_lines.append(f"\n---- 综合分数:")
        output_lines.append(f"  F1 分数: {f1_score:.4f}")

        # 7. 性能分析建议
        output_lines.append("\n---- 性能分析:")

        # 基于F1分数给出评价
        if f1_score >= 0.85:
            evaluation = "优秀"
            suggestion = "当前配置表现良好，可考虑投入生产环境使用。"
        elif f1_score >= 0.70:
            evaluation = "良好"
            suggestion = "配置表现尚可，可尝试微调参数以进一步提升效果。"
        elif f1_score >= 0.50:
            evaluation = "一般"
            suggestion = "存在改进空间，建议检查检索质量和答案生成准确性。"
        else:
            evaluation = "待改进"
            suggestion = "建议重新审视chunk_size、检索策略等核心参数配置。"

        output_lines.append(f"  综合评价: {evaluation}")
        output_lines.append(f"  优化建议: {suggestion}")

        # 8. 各问题指标详情（可选）
        if show_details and ragas_metrics:
            output_lines.append("\n---- 各问题详细指标:")
            for i in range(len(questions)):
                output_lines.append(f"\n  【问题 {i + 1}】:")
                for metric_key, display_name in metric_names.items():
                    if metric_key in ragas_metrics and i < len(ragas_metrics[metric_key]):
                        value = ragas_metrics[metric_key][i]
                        output_lines.append(f"    {display_name}: {value:.4f}")

        output_lines.append("\n" + "=" * 60)

        return "\n".join(output_lines)


    @staticmethod
    def _calc_f1_score(evaluation_dict):
        """计算基于context_precision和context_recall的F1分数"""
        precisions = evaluation_dict["context_precision"]
        recalls = evaluation_dict["context_recall"]

        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)

        if (avg_precision + avg_recall) == 0:
            return 0.0
        return (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

