import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from data_ingestion import all_documents

print("Chunking documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(all_documents)
print(f"Generated {len(chunks)} text chunks.")


print("Generating embeddings...")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Creating FAISS vector index...")
vectorstore = FAISS.from_documents(chunks, embedding_model)

VECTOR_DIR = "vectorstore"
if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)

vectorstore.save_local(VECTOR_DIR)
print(f"✅ Vector store saved to: {VECTOR_DIR}")
