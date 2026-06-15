from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()  # 加载环境变量

# 状态
class MyState(TypedDict):
    msg: Annotated[str, add_messages]
    time: str

# 模型
model = init_chat_model(
    model="qwen3.6-flash",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# node
def graph_node(state: MyState):
    answer = model.invoke(state["msg"])
    return {"msg": answer}

# 图
graph = StateGraph(MyState)

# 添加节点
graph.add_node("model_node", graph_node)

# 添加边
graph.add_edge(START, "model_node")
graph.add_edge("model_node", END)

# 编译
app = graph.compile()

# 运行
result = app.invoke({"msg": "请用一句话解释什么是 rag。", "time": "2026-06-11 20:00:00"})
print("模型回答：", result["msg"][-1].content)

print()

print("图的ascii可视化结构：")
app.get_graph().print_ascii()