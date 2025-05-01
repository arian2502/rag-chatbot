import torch
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# === Load Vector Store ===
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# === Load Local LLM ===
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

# === Setup Prompt Template ===
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using only the context below. If the answer is not present, respond with "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

# === Initialize Components ===
st.set_page_config(page_title="Local RAG Chatbot", layout="centered")
st.title("🤖 Local RAG Chatbot")
st.caption("Chat with your support documents and Angel One help pages (fully offline).")

vectorstore = load_vectorstore()
llm = load_llm()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt}
)

# === Session State for Conversation ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Chat Input and Handling ===
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        result = qa_chain(user_input)
        st.session_state.chat_history.append(("bot", result["result"]))

# === Display Chat History ===
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
