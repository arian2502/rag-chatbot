import os
import torch
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


st.set_page_config(page_title="Local RAG Chatbot", layout="centered")
st.title("🤖 Local RAG Chatbot")
st.caption("Ask questions based on your uploaded documents. No internet or API required.")


@st.cache_resource
def load_vectorstore():
    documents = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("docs", file))
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(os.path.join("docs", file))
        else:
            continue
        documents.extend(loader.load())

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=0 if torch.cuda.is_available() else -1,
    )
    return HuggingFacePipeline(pipeline=pipe)

# === Load Resources ===
vectorstore = load_vectorstore()
llm = load_llm()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using only the context below.
If the answer is not present, respond with "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        result = qa_chain(user_input)
        st.session_state.chat_history.append(("bot", result["result"]))

for sender, message in st.session_state.chat_history:
    st.chat_message(sender).write(message)
