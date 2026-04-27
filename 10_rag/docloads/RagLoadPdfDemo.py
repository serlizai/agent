# pip install langchain_community
from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader(
    # 文件路径，支持本地文件和在线文件链接，如"https://arxiv.org/pdf/alg-geom/9202012"
    file_path="assets/sample.pdf",
    # 提取模式:
    #   plain 提取文本 适合：普通文章、合同、说明书，只关心文字内容
    #   layout 按布局提取 适合：表格、双栏论文、简历、发票、带明显排版结构的 PDF
    extraction_mode="plain",
).load()

print(docs)
