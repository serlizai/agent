# https://bailian.console.aliyun.com/cn-beijing/?productCode=p_efm&tab=doc#/doc/?type=model&url=2842587

import dashscope
from http import HTTPStatus
import os
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("QWEN_API_KEY")

input_text = "衣服的质量杠杠的"

# resp = dashscope.TextEmbedding.call(  使用text-embedding-v4对应TextEmbedding接口
resp = dashscope.MultiModalEmbedding.call(  # qwen3-vl-embedding必须用MultiModalEmbedding接口
    model="qwen3-vl-embedding",
    # input=input_text,
    input=[{"text": input_text}],
)

if resp.status_code == HTTPStatus.OK:
    print(resp)


