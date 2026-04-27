import os
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(encoding='utf-8')

llm = ChatZhipuAI(
    model="GLM-4-Flash",
    api_key=os.getenv("ZHIPU_API_KEY"),
    streaming=True
)

# 和invoke("你是谁")相同效果，但是列表更适合后续增加系统消息、助手消息等，灵活性更高
response = llm.stream([HumanMessage(content="介绍一下你自己，20字以内")])

for chunk in response:
    print(chunk.content, end="")
