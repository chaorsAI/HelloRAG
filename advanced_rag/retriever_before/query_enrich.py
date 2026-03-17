
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
import json

from advanced_rag.models.models import get_ali_model_client

#获得访问大模型客户端
llm = get_ali_model_client()

# 用户的要求
user_input = "我想订一张长沙去北京的机票"

# 首先根据用户的要求，进行意图识别，获取对应的模板
# 示例业务模板
templates = {
    "订机票": ["起点", "终点", "时间", "座位等级", "座位偏好"],
    "订酒店": ["城市", "入住日期", "退房日期", "房型", "人数"],
}

# 意图识别提示模板
intent_prompt = PromptTemplate(
    input_variables=["user_input", "templates"],
    template="根据用户输入 '{user_input}'，选择最合适的业务模板。可用模板如下：{templates}。请返回模板名称。"
)

# 创建意图识别链
intent_chain = intent_prompt | llm

# 识别意图
intent = intent_chain.invoke({"user_input": user_input, "templates": str(list(templates.keys()))}).content

print("意图：", intent)
# 获取对应模板
selected_template = templates.get(intent)
print("模板：", selected_template)

# 根据用户意图已经获取对应的模板，然后判断是否需要补充信息
# 补充信息提示模板
info_prompt = f"""
    请根据用户原始问题和模板，判断原始问题是否完善。如果问题缺乏需要的信息，请生成一个友好的请求，明确指出需要补充的信息。若问题完善后，返回包含所有信息的完整问题。

    ### 原始问题    
    {user_input}

    ### 模板
    {",".join(selected_template)}                                   

    ### 输出示例
    {{
        "isComplete": true,
        "content": "`完整问题`"
    }}
    {{
        "isComplete": false,
        "content": "`友好的引导用户补充需要的信息`"
    }}                                       
"""

# 历史记录
chat_history = ChatMessageHistory()

# 聊天模版
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个信息补充助手，任务是分析用户问题是否完整。"),
        ("placeholder", "{history}"),  # 历史记录的占位
        ("human", "{input}"),
    ]
)

# 补充信息链
info_chain = prompt | llm

# 自动处理历史记录，将记录注入输入并在每次调用后更新它
with_message_history = RunnableWithMessageHistory(
    info_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 判断问题是否完整，如果不完整，则生成追问请求
info_request = with_message_history.invoke(input={"input": info_prompt},
                                           config={"configurable": {"session_id": "unused"}}).content
parser = JsonOutputParser()
json_data = parser.parse(info_request)
print("json_data：",json_data)

# 循环判断是否完整，并提交用户补充信息
while json_data.get('isComplete', False) is False:
    try:
        # 显示引导信息并等待用户输入，用\033[1;33m和\033[0m设置和重置文本颜色及样式（黄色加粗）
        user_answer = input(f"\033[1;33m{json_data['content']}\033[0m\n请补充：")

        # 提交补充信息给AI处理
        info_request = with_message_history.invoke(
            input={"input": user_answer},
            config={"configurable": {"session_id": "unused"}}
        ).content

        # 解析AI响应
        json_data = parser.parse(info_request)

    except json.JSONDecodeError:
        #\033[1;31m 是 ANSI 转义字符，用于设置字体颜色为红色并加粗，\033[0m 用于恢复默认字体样式
        print("\033[1;31m[错误] AI返回了无效的JSON格式，请重试\033[0m")
        continue
    except KeyError:
        print("\033[1;31m[错误] 响应格式异常，正在终止流程\033[0m")
        break

# 输出最终结果
print(f"\033[1;32m[最终查询] {info_request}\033[0m")