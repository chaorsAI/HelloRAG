# 负责与LangChain链交互，生成答案和获取上下文


from typing import List


class QAEvaluator:
    def __init__(self, retrieval_chain):
        self.chain = retrieval_chain

    def generate_answers(self, questions: List[str]):
        """对问题列表生成答案，并收集使用的上下文"""
        answers = []
        contexts = []
        for i, q in enumerate(questions, 1):
            print(f"\n问题 {i}/{len(questions)}: {q}")
            response = self.chain.invoke({"input": q})
            answers.append(response["answer"])
            contexts.append([doc.page_content for doc in response["context"]])
            print(f"答案: {response['answer'][:150]}...")  # 预览前150字符
        return answers, contexts
