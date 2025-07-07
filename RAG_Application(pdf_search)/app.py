import streamlit as st
import torch
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import tempfile

# Load API key
load_dotenv()

st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 Soch k likhunga")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    with st.spinner("🔄 Processing PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyMuPDFLoader(tmp_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata['source'] = uploaded_file.name
            all_docs.extend(documents)
            os.remove(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(all_docs)

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )

        vectorstore = FAISS.from_documents(docs, embedding)
        st.success("✅ PDFs processed successfully!")

        llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("💬 Ask a question about the PDFs")
        if query:
            with st.spinner("🤖 Thinking..."):
                try:
                    result = qa_chain.invoke({"query": query})
                    answer = result["result"]
                    sources = result.get("source_documents", [])

                    st.session_state.chat_history.append(("You", query))
                    st.session_state.chat_history.append(("Bot", answer))

                    for role, msg in st.session_state.chat_history:
                        st.markdown(f"**{role}:** {msg}")

                    if sources:
                        st.markdown("---")
                        st.markdown("### 📄 Related Passages")
                        for i, doc in enumerate(sources[:5]):
                            page_number = doc.metadata.get("page")
                            page_number = page_number + 1 if page_number is not None else "N/A"
                            st.markdown(f"**Match {i+1}**")
                            st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')} | **Page:** {page_number}")
                            st.write(doc.page_content[:500] + "...")
                            st.markdown("---")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")