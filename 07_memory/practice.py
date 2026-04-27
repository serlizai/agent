from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
import os
import redis  #导入原生redis库，pip install redis==5.3.1
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = "redis://localhost:26379"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

llm = init_chat_model(
    model="MiniMax-M2.1",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的Java技术专家"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """获取或创建会话历史（使用 Redis）"""
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        # ttl=3600  # 注释：关闭自动过期，避免重启后数据被清理
    )
    return history  

chain = RunnableWithMessageHistory(
    prompt | llm | StrOutputParser(),
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

config = RunnableConfig(configurable={"session_id": "user-002"})

# response1 = chain.invoke({"question": "Java中的多态是什么？50字说明"}, config=config)
# logger.info(f"模型回答：{response1.content}")

# response2 = chain.invoke({"question": "能举个例子吗？50字以内"}, config=config)
# logger.info(f"模型回答：{response2.content}")

# ---------------- 流式输出 ----------------
is_first_chunk = True  # 🔴 修改点 2：增加首块标记

for chunk in chain.stream({"question": "我刚刚问了什么？50字以内说明"}, config=config):
    # 因为加了 StrOutputParser，chunk 现在直接就是字符串，不需要再用 chunk.content
    
    if is_first_chunk:
        chunk = chunk.lstrip()  # 去除第一个有效 chunk 左侧的换行和空格
        if chunk:  # 如果去除后还有内容，说明第一个有意义的字来了
            is_first_chunk = False
            
    if chunk:  # 只有 chunk 不为空才打印（过滤掉纯空字符串的 chunk）
        print(chunk, end="", flush=True)

print()  # 打印完整个回答后，补一个换行，防止终端提示符跟在最后面

# 等同于redis-cli的SAVE命令，强制写入dump.rdb
redis_client.save()