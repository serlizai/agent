from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="MiniMax-M2.1",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

prompt_template1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请简短回答我提出的问题"),
    ("human", "请回答:{question} {format_instructions}")
])

prompt_template2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请简短回答我提出的问题,并说出你的名字"),
    ("human", "我是：{name}，请回答:{question} {format_instructions}")
])

prompt_template3 = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请简短回答我提出的问题,用英文回答"),
    ("human", "Please answer:{question} {format_instructions}")
])


parser = JsonOutputParser()
name = ["mike", "lucy"]

chain1 = prompt_template1 | model | parser
chain2 = prompt_template2 | model | parser
chain3 = prompt_template3 | model | parser
full_chain = chain1 | (lambda x: {"role": "agent技术专家", 
                                  "question": f"{x['answer']} 总结以上内容，简洁回答20字以内`",
                                  "format_instructions": parser.get_format_instructions()}) | chain3

# human 消息里的语言暗示比 system 消息的语言指令权重更高，模型倾向于跟随问题的语言风格回答
def middle(content):
    logger.info(f"中间结果: {content}")
    return {"role": "agent技术专家", 
            "question": f"{content['answer']} 总结以上内容，简洁回答20字以内`",
            "format_instructions": parser.get_format_instructions()}

# RunnableBranch(
    # (条件1, Runnable1),   # 条件分支：二元组
    # (条件2, Runnable2),   # 条件分支：二元组
    # ...
    # 默认Runnable          # 默认分支：直接是 Runnable，不加括号 ✅
# )
chain_B = RunnableBranch(
    (lambda x: x["name"] == "mike", chain1),
    (chain2)  # 默认Runnable, 这不是元组！只是加了括号的普通表达式，等价于 chain2
    # (chain2,)  # ← 这才是元组，注意有逗号
)

chain_P = RunnableParallel({
    "mike": chain1,
    "lucy": chain3
})

middle_node = RunnableLambda(middle)

chain_L = chain1 | middle_node | chain3

# 自动生成格式化指令，指导模型输出符合要求的JSON结构
# result = chain_B.invoke({"role": "agent技术专家", "question": "什么是agent技术，简洁回答100字以内`", "name": "lucy",
#                         "format_instructions": parser.get_format_instructions()})
# result2 = full_chain.invoke({"role": "agent技术专家", "question": "什么是agent技术，简洁回答100字以内`", "name": "mike",
#                              "format_instructions": parser.get_format_instructions()})
# result3 = chain_P.invoke({"role": "agent技术专家", "question": "什么是agent技术，简洁回答100字以内`",
#                          "format_instructions": parser.get_format_instructions()})
result4 = chain_L.invoke({"role": "agent技术专家", "question": "什么是agent技术，简洁回答100字以内`",
                          "format_instructions": parser.get_format_instructions()})


# logger.info(f"最终结果111:{result}") 
# logger.info(f"最终结果222:{result2}") 
# logger.info(f"最终结果333:{result3}") 
logger.info(f"最终结果444:{result4}") 

