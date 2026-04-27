import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from loguru import logger
from dotenv import load_dotenv
from search import search

load_dotenv()


llm = init_chat_model(
    model="MiniMax-M2.1",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

prompt1 = PromptTemplate.from_template(
    """
    你是一个专业的搜索引擎助手，你将收到搜索结果：{search}
    你负责将第一个搜索结果以简洁的方式返回给用户。
    """
)

# 将模型与工具绑定，使其能够调用 search 工具，顺序调用
llm_with_tools = llm.bind_tools([search])

# 解析模型输出中的工具调用参数，并把参数真正传给 search 工具执行
search_parser = JsonOutputKeyToolsParser(key_name=search.name, first_tool_only=True)


search_chain = llm_with_tools | search_parser | search
output_chain = prompt1 | llm | StrOutputParser()
full_chain = search_chain | (lambda x: {"search": x}) | output_chain

question = input("请输入你的问题：")
result = full_chain.invoke(question)
logger.info(f"搜索结果：{result}")