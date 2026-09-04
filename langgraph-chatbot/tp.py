from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    token=os.getenv("HF_TOKEN")
)

response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ],
    max_tokens=50
)

print(response.choices[0].message.content)