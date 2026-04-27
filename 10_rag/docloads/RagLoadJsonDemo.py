# pip install jq
from langchain_community.document_loaders import JSONLoader

# 提取所有字段
docs = JSONLoader(
    file_path="assets/sample.json",  # 文件路径

    # jq_schema=".data.items"  提取 items 数组
    # jq_schema=".data"   提取 data 字段
    # jq_schema=".data.items[]"  把 items 数组里的每一项分别提取成一个 Document
    # jq_schema=".data.items[].content"  只提取每篇文章的 content 字段
    jq_schema=".",  # 提取所有字段
    text_content=False,  # 提取内容是否为字符串格式
).load()

print(docs)
