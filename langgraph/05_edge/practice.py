from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from loguru import logger
from pydantic import BaseModel

class Mystate(TypedDict):
    x: int
    result: str = ""

def node0(state: Mystate) -> Mystate:
    logger.info("inital node")
    return state

def node1(state: Mystate) -> Mystate:
    logger.info(f"node1 received state: {state}")
    state["x"] += 1
    state["result"] = "node1"
    return state

def node2(state: Mystate) -> Mystate:
    logger.info(f"node2 received state: {state}")
    state["x"] = 2
    state["result"] = "node2"
    return state

graph = StateGraph(Mystate)
graph.add_node("node0", node0)  # 添加 node0 节点
graph.set_entry_point("node0")  # 设置入口点为 node0,必须是字符串
graph.add_node("node1", node1)
graph.add_node("node2", node2)

def route(state: Mystate) -> str:
    if state["x"] == 1:
        return "node1"
    return "node2"

graph.add_conditional_edges(
    "node0",  # 从 node0 出发, 要求传入字符串
    route,
    {
        "node1": "node1",
        "node2": "node2"
    }
)
graph.add_edge("node1", END)
graph.add_edge("node2", END)
app = graph.compile()
initial_state = {"x": 1}
final_state = app.invoke(initial_state)
logger.info(f"Final state: {final_state}")
logger.info(f"Final result: {final_state['result']}")

print()
print("=================================")
print(app.get_graph().print_ascii())
    
    