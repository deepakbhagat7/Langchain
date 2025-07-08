from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(model = "llama-3.1-8b-instant")
# storing the message in chat history
chat_history = [
    SystemMessage(content="You are a helpful assistant")
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    model_response = model.invoke(chat_history)
    chat_history.append(AIMessage(content = model_response.content))
    print('AI ',model_response.content)

# Output where we need do save the history
# You:which one is greater 2 or 10
# AI 10 is greater than 2.
# You:multiply the greater number by 10
# AI However, I don't see a number provided. Could you please give me the two numbers you'd like me to compare and multiply the greater number by 10?