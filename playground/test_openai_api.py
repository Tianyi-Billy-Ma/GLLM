import os
import torch
from openai import OpenAI
from openai import OpenAI
from fastchat.model import get_conversation_template

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


def chatgpt():
    model = "gpt-3.5-turbo"
    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], None)

    messages = conv.to_openai_api_messages()
    print(messages)

    res = client.chat.completions.create(model=model, messages=messages)
    msg = res.choices[0].message.content
    print(msg)

    res = client.chat.completions.create(model=model, messages=messages, stream=True)
    msg = ""
    for chunk in res:
        msg += chunk.choices[0].delta.content
    print(msg)


chatgpt()
