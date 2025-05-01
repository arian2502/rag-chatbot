from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import torch

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# === Step 2: Load Local LLM (FLAN-T5 Base is small and good for QA) ===
model_name = "google/flan-t5-base"  # or try "tiiuae/falcon-rw-1b" for bigger model

print("Loading local model... (this may take ~30s on first run)")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

local_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device=0 if torch.cuda.is_available() else -1,
)

llm = HuggingFacePipeline(pipeline=local_pipe)

# === Step 3: Setup Prompt ===
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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# === Step 4: Start Chat ===
def chat():
    print("🤖 Local RAG Chatbot (No Internet, No API Keys)")
    print("Ask anything based on uploaded docs and support pages.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain(query)
        print(f"\nBot: {result['result']}\n")

if __name__ == "__main__":
    chat()
