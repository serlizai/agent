# pip install langchain-community dashscope redis redisvl
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# 1. 初始化阿里千问 Embedding 模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 支持 v1 或 v2
    dashscope_api_key=os.getenv("QWEN_API_KEY")  # 从环境变量读取
)


# 2. 准备要向量化的文本（Document 列表）
texts = [
    "通义千问是阿里巴巴研发的大语言模型。",
    "Redis 是一个高性能的键值存储系统，支持向量检索。",
    "LangChain 可以轻松集成各种大模型和向量数据库。"
]

# Document = {
#     "page_content": "真正要被检索/喂给大模型的文本内容",
#     "metadata": "这段文本的来源、页码、类别等附加信息"
# }
documents = [Document(page_content=text, metadata={"source": "manual"}) for text in texts]

# 3. 连接到 Redis 并存入向量（自动调用 embeddings 嵌入）
vector_store = Redis.from_documents(
    documents=documents,
    embedding=embeddings,
    redis_url="redis://localhost:26379",  # 替换为你的 Redis 地址
    index_name="my_index11",               # 向量索引名称
)

# 4. （可选）后续可直接用于检索
# retriever 是 LangChain 里专门用于“根据问题检索相关文档”的统一接口
# LangChain 很多链式组件，比如 RAG Chain，默认接收的是 retriever，不是直接接收 vector_store
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # 每次检索时，返回最相关的 2 条文档
results = retriever.invoke("LangChain 和 Redis 怎么结合？")
for res in results:
    print(res.page_content)

