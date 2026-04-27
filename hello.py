from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

@tool 
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together.""" 
    return first_int * second_int


llm = ChatTongyi(model="qwen-max", api_key=api_key, base_url=base_url)

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("What's 5 times forty two")

print(msg)