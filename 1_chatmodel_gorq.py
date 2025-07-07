from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model  = ChatGroq(model = 'llama-3.1-8b-instant')

res = model.invoke("Which city is known as the coal capital of india?")

print(res.content)