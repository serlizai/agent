# from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
# from langchain.tools import tool

# 加载环境变量
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

chatLLM = ChatTongyi(streaming=False, api_key=api_key, base_url=base_url)

res = chatLLM.invoke([HumanMessage(content="hi")], streaming=False)  # 非流式

for r in res:
    print("chat resp:", r)