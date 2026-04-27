import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

model = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("QWEN_API_KEY")
)

texts = [
    "SpringBoot 是一个基于 Spring 的快速开发框架，简化了 Java 企业应用的开发流程。",
    "Spring是一个开源的 Java 企业级应用开发框架，提供了全面的基础设施支持，帮助开发者构建高效、可维护的应用程序。",
    "SpringCloud 是基于 SpringBoot 的微服务开发框架，提供了分布式系统的常用组件和解决方案。"
]

documents = [Document(
    page_content=text,
    metadata={
        "source": "net",
        "catogory": "tech"
    }) for text in texts]

vector_store = Redis.from_documents(
    documents=documents,
    embedding=model,
    redis_url="redis://localhost:26379",
    index_name="my_index12",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("SpringBoot 和 SpringCloud 的区别？")
for res in results:
    print(res.page_content)

