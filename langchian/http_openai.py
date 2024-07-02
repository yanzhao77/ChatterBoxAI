import random
from http import HTTPStatus
import dashscope
from openai import OpenAI
import requests
dashscope.api_key = "sk-3f2d7473809f4f0492976b33f3146299"


def call_with_messages():
    messages = [
        {'role': 'user', 'content': '用萝卜、土豆、茄子做饭，给我个菜谱'}]
    response = dashscope.Generation.call(
        'qwen1.5-0.5b-chat',
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


def get_response():
    client = OpenAI(
        api_key='sk-3f2d7473809f4f0492976b33f3146299',  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
    )
    completion = client.chat.completions.create(
        model="qwen1.5-0.5b-chat",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': '你能做什么'}]
    )
    print(completion.model_dump_json())


def get_response2():
    url = "https://spark-api.xf-yun.com/v1.1/chat"
    data = {
        "model": "generalv3.5",  # 指定请求的模型
        "messages": [
            {
                "role": "user",
                "content": "你是谁"
            }
        ]
    }
    header = {
        "Authorization": "Bearer 3e1295178045e81ac519b8e3477e95d5:OTU3MjNkMmIxZTYzOTY5NWVmZDRjMzA3"  # 注意此处替换自己的key和secret
    }
    response = requests.post(url, headers=header, json=data)
    print(response.text)


if __name__ == '__main__':
    # call_with_messages()
    get_response()

