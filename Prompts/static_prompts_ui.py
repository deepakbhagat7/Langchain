from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

model  = ChatGroq(model = 'llama-3.1-8b-instant')

st.header("Research Tool")

user_input= st.text_input("Enter your prompt")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content )