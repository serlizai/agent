from pathlib import Path

from langchain.chat_models  import  init_chat_model
import os

from langchain_community.document_loaders import Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
load_dotenv()  # 加载环境变量

llm = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


prompt_template = ChatPromptTemplate.from_template(
    """请根据下面提供的上下文信息来回答问题。
        请确保你的回答完全基于这些上下文。
        如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

        上下文:
        {context}

        问题: {question}

        回答:
    """
)

embedding = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("QWEN_API_KEY")
)

# 加载文档
# docs = UnstructuredMarkdownLoader(
#     # 文件路径
#     file_path="10_rag/easy-rl-chapter1.md",
#     # 加载模式:
#     #   single 返回单个Document对象
#     #   elements 按标题等元素切分文档
#     # element模式向量检索认为标题 1.1 强化学习概述 很相关，于是返回了标题 chunk，
#     # 但这个 chunk 只有标题，没有正文，所以大模型看到的上下文不够，才回答“找不到相关信息
#     mode="elements"
# ).load()
markdown_text = Path("10_rag/easy-rl-chapter1.md").read_text(encoding="utf-8")

# 文档切分
# headers_to_split_on：按哪些标题切
# return_each_line：一行一条，还是同 header 下聚合
# strip_headers：标题留不留在内容里
# custom_header_patterns：要不要扩展非标准标题语法
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    return_each_line=False,
    strip_headers=False,
)
# md_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )

content = md_splitter.split_text(markdown_text)
print("="*50)
print(f"文档个数：{len(content)}")

# 连接到 Redis 并存入向量（自动调用 embeddings 嵌入）
vector_store = Redis.from_documents(
    documents=content,
    embedding=embedding,
    redis_url="redis://localhost:26379",  # 替换为你的 Redis 地址
    index_name="rag_index1",  # 向量索引名称
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 只打印检索结果里的正文文本
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    # 会被自动转换成一个 RunnableParallel，意思是：把同一个输入并行传给多个 Runnable，然后组装成一个字典输出。
        {
            "context": retriever | format_docs,  # retriever.invoke("00000和A0001分别是什么意思")
            "question": RunnablePassthrough()  # 输入原样传下去，原始问题不做任何处理，直接作为 question
        }
        | prompt_template  # 将 retriever 的输出作为 context，用户输入作为 question，格式化到 prompt 模板中
        | llm
        | StrOutputParser()  # 将 LLM 输出转换成字符串
)

question = input("请输入你的问题：")
print(format_docs(retriever.invoke(question)))  # 只打印检索结果中的文本内容，方便检查上下文是否合理
print("="*60)
answer = rag_chain.invoke(question)
print("\n\n")
print(f"\n回答：{answer}")
