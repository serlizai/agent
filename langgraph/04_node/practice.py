from typing import Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


# 定义状态类型
class AtguiguState(TypedDict):
    result: str

# 全局计数器：记录API尝试次数
attempt_counter = 0


# 工具函数
def build_retry_graph(node_name: str, node_func, retry_policy: RetryPolicy):
    builder = StateGraph(AtguiguState)
    #为节点添加重试策略，需要在add_node中设置retry_policy参数。
    # retry_policy参数接受一个RetryPolicy命名元组对象。
    # 默认情况下，retry_on参数使用default_retry_on函数，该函数会在遇到任何异常时重试
    builder.add_node(node_name, node_func, retry_policy=retry_policy)
    builder.add_edge(START, node_name)
    builder.add_edge(node_name, END)
    return builder.compile()

def error_call(state: AtguiguState) -> Dict[str, Any]:
    """模拟错误：前2次失败，第3次成功（全局计数器记录尝试次数）"""
    global attempt_counter
    attempt_counter += 1
    # 纯文本打印尝试次数
    print(f"尝试重连，这是第 {attempt_counter} 次尝试")

    # 模拟失败/成功逻辑：前2次抛异常，第3次返回结果
    if attempt_counter < 3:
        raise Exception(f"尝试重连 (尝试 {attempt_counter})")
    return {"result": f"重连成功，经过 {attempt_counter} 次尝试"}

def custom_retry(exception: Exception) -> bool:
    if "尝试重连" in str(exception):
        print(f"捕获到可重试异常: {str(exception)}")
        return True
    print(f"捕获到不可重试异常: {str(exception)}")
    return False

# 测试重试策略
graph = build_retry_graph("error_node", error_call, retry_policy=RetryPolicy(max_attempts=5, retry_on=custom_retry))
custom_result = graph.invoke({"result": ""})
print("\n测试结果:", custom_result)
