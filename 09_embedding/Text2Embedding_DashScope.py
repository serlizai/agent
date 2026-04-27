"""
https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2587654
pip install langchain-community dashscope
"""

from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
if not api_key:
    raise RuntimeError("未读取到 QWEN_API_KEY，请检查 .env 文件或环境变量配置。")

embeddings = DashScopeEmbeddings(  # 多模态模型要调用 dashscope.MultiModalEmbedding.call() 来获取向量
    model="text-embedding-v4",
    dashscope_api_key=api_key,
    # other params...
)

text = "This is a test document."

query_result = embeddings.embed_query(text)  # 阿里默认返回 1024 维向量
print("文本向量长度：", len(query_result), sep='')
#
doc_results = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ])
print(doc_results)
print("文本向量数量：", len(doc_results), "，文本向量长度：", len(doc_results[0]), sep='')
