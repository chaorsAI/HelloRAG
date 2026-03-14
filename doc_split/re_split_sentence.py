
# 正则表达式
import re

text = ("自然语言处理（NLP），作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力。"
        "在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，机器得以分析、领会乃至创造我们所理解的语言。"
        "从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。随着深度学习技术的飞速进步，"
        "NLP的精确度与效能均实现了巨大飞跃！如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，"
        "如问答系统、语音识别和对话系统等。NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。")

'''
re.split(r'(。|？|！|\..\..)', text) 是使用正则表达式对字符串 text 进行分割的操作。
re.split(pattern, string)：这是 Python 的 re 模块中的一个函数，用于根据正则表达式 pattern 将字符串 string 分割成多个部分。
r'(。|？|！|\..\..)'：这是一个正则表达式模式，用于匹配中文句子结束的标点符号。
，re.split 函数会返回一个列表，其中包含分割后的子字符串和匹配到的标点符号
'''
sentences = re.split(r'(。|？|！|\..\..)', text)
print(sentences)
print('-' * 100)

# 重新组合句子和结尾的标点符号
'''[::2] 是 Python 中列表切片的语法，表示从列表中以步长为 2 提取元素。具体含义如下：
: 表示切片操作。
第一个位置为空，默认从索引 0 开始。
第二个位置为空，默认到列表末尾结束。
2 表示步长，即每隔 2 个元素提取一次'''
'''使用 zip 将 sentences 的偶数索引部分 (sentences[::2]) 和奇数索引部分 (sentences[1::2]) 配对
会形成类似的列表[
('自然语言处理（NLP），作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力','。'),
('在这个领域，机器学习发挥着至关重要的作用','。'),
(..,..),...
]'''
# chunks = [sentence + (punctuation if punctuation else '')
#           for sentence, punctuation in zip(sentences[::2], sentences[1::2])]

chunks = [f'{sentences[i]}{sentences[i+1]}'  for i in range(0, len(sentences)-2, 2)]

for i, chunk in enumerate(chunks):
    print(f"块 {i + 1} - 长度{len(chunk)}，内容: {chunk}")
