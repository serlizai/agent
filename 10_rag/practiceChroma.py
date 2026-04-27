import chromadb
import os
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from chromadb.utils.embedding_functions import register_embedding_function


load_dotenv()

@register_embedding_function
class DashScopeEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError("未读取到 QWEN_API_KEY，请检查 .env 文件或环境变量配置。")

        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=api_key,
        )

    def __call__(self, input: Documents) -> Embeddings:
        return self.embeddings.embed_documents(list(input))


chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="my_collection",
    embedding_function=DashScopeEmbeddingFunction(),  # 自定义嵌入函数，可以指定使用的 Embedding 模型
)
collection.add(
    ids=["id1", "id2"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ]
)

results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)
