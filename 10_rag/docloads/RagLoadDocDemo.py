# pip install langchain_community unstructured[docx]
# pip install -U unstructured
# pip install python-docx
# pip install regex==2026.1.14
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

docs = UnstructuredWordDocumentLoader(
    # 文件路径
    file_path="assets/alibaba-more.docx",
    # 加载模式:
    #   single 返回单个Document对象  整个 Word 文件作为一个 Document 返回
    #   elements 按标题等元素切分文档 按文档元素拆成多个 Document，比如标题、段落、列表、表格等
    # 想保留 Word 结构，比如标题、段落、列表、表格分别处理，就用 elements 模式；只关心文本内容，不在意结构，就用 single 模式
    mode="single",
).load()

#print(type(docs))
print(docs)

