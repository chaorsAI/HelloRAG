import numpy as np
def _calculate_average_precision(verdict_list) -> float:
    score = np.nan

    denominator = sum(verdict_list) + 1e-10
    numerator = sum(
        [
            (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
            for i in range(len(verdict_list)) # i= 0,1,2,3,4
        ]
    )
    print(denominator)
    print(numerator)
    score = numerator / denominator
    return score

verdict_list = [1, 0, 1, 1, 0]
score = _calculate_average_precision(verdict_list)
print(score)  # 输出结果
'''
i= 0,1,2,3,4
第一个片段：(1 / 0+1) * 1 = 1
第二个片段：(1 / 1+1) * 0 = 0
第三个片段：(2 / 2+1) * 1 = 0.6667
第四个片段：(3 / 3+1) * 1 = 0.75
第五个片段：(3 / 4+1) * 0 = 0
分子总和：1 + 0 + 0.6667 + 0.75 + 0 = 2.4167
相关片段的总数为3，所以分母为3
最终得分：2.4167 / 3 ≈ 0.8056
'''

verdict_list = [1, 1, 0, 1, 0]
score = _calculate_average_precision(verdict_list)
print(score)  # 输出结果