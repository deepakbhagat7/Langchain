from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"],
)


completion = client.chat.completions.create(
    model="Menlo/Jan-nano-128k",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of India?"
        }
    ],
)

print(completion.choices[0].message)


