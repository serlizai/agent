from pathlib import Path

# import chromadb
from langchain_chroma import Chroma
import os
# from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
# from chromadb.utils.embedding_functions import register_embedding_function
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embedding = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("QWEN_API_KEY")
)

llm = init_chat_model(
    model="qwen3.5-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

prompt_template = """请根据下面提供的上下文信息来回答问题。
        请确保你的回答完全基于这些上下文。
        如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

        上下文:
        {context}

        问题: {question}

        回答:
    """

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

markdown_text = Path("10_rag/easy-rl-chapter1.md").read_text(encoding="utf-8")

# 文档切分
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    return_each_line=False,
    strip_headers=False,
)
doc = md_splitter.split_text(markdown_text)


# @register_embedding_function
# class DashScopeEmbeddingFunction(EmbeddingFunction[Documents]):
#     def __init__(self):
#         api_key = os.getenv("QWEN_API_KEY")
#         if not api_key:
#             raise RuntimeError("未读取到 QWEN_API_KEY，请检查 .env 文件或环境变量配置。")

#         self.embeddings = DashScopeEmbeddings(
#             model="text-embedding-v3",
#             dashscope_api_key=api_key,
#         )

#     def __call__(self, input: Documents) -> Embeddings:
#         return self.embeddings.embed_documents(list(input))
    
# 内存版
# chroma_client = chromadb.Client()
# Chroma原生持久版
# chroma_client = chromadb.PersistentClient(path="10_rag/chroma_db")
# collection = chroma_client.get_or_create_collection(
#     name="rag_chroma",
#     embedding_function=DashScopeEmbeddingFunction(),  # 自定义嵌入函数，可以指定使用的 Embedding 模型
# )

# collection.add(
#     ids=[f"id{i}" for i in range(len(doc))],
#     documents=doc
# )

vector_store = Chroma.from_documents(
    documents=doc,
    embedding=embedding,
    collection_name="rag_chroma",
    persist_directory="10_rag/chroma_db",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 删除
# collection.delete(ids=["id1", "id2"])

# 打印数据库数据
# data = collection.get()
# print(data)

# 只打印检索结果里的正文文本
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    # 会被自动转换成一个 RunnableParallel，意思是：把同一个输入并行传给多个 Runnable，然后组装成一个字典输出。
        {
            "context": retriever | format_docs,  # retriever.invoke("00000和A0001分别是什么意思")
            "question": RunnablePassthrough()  # 输入原样传下去，原始问题不做任何处理，直接作为 question
        }
        | prompt  # 将 retriever 的输出作为 context，用户输入作为 question，格式化到 prompt 模板中
        | llm
        | StrOutputParser()  # 将 LLM 输出转换成字符串
)

question = input("请输入你的问题：")
print(format_docs(retriever.invoke(question)))  # 只打印检索结果中的文本内容，方便检查上下文是否合理
print("="*60)
answer = rag_chain.invoke(question)
print("\n")
print(f"\n回答：{answer}")