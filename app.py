import os
import tempfile
import streamlit as st
from main import ingest_pdf, chat

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("PDF Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Select PDF", type="pdf")
    if uploaded_file and st.button("Process"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with st.spinner("Processing PDF..."):
            num_chunks = ingest_pdf(tmp_path)
        os.unlink(tmp_path)
        st.success(f"{num_chunks} chunks stored!")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chat(prompt, st.session_state.history[:-1])
        st.write(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})