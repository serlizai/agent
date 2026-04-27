"""
模型参数演示
"""
import os
from langchain.chat_models import init_chat_model

# 实例化模型，设置 temperature 参数，控制生成文本的随机程度，数值越大生成的文本越随机，数值越小生成的文本越确定
model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com",
    temperature=2.0

)

# 3.调用模型
for x in range(3):
    print(model.invoke("写一句关于春天的词,14字以内").content)