import openai
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8082/v1', api_key='token-abc123')

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="Qwen/Qwen2.5-Coder-3B-Instruct",
    n=5
)
print(chat_completion.choices[0])
choice = chat_completion.choices[0]
message_content = choice.message.content  # Get the message content

print(message_content)