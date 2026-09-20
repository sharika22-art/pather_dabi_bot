import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_db():
    data_path = "data/pather_dabi.json"
    if not os.path.exists(data_path):
        print("Data file not found. Please run 1_scraper.py first.")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        book_data = json.load(f)

    documents = []
    for chapter in book_data:
        documents.append(
            Document(
                page_content=chapter["content"],
                metadata={
                    "book": chapter["book_name"], 
                    "chapter": chapter["chapter_name"], 
                    "url": chapter["url"]
                }
            )
        )

    # Chunking optimized for Bengali
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "।", " ", ""],
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split book into {len(chunks)} chunks.")

    print("Loading embedding model (this might take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    print("Building Vector Database...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("Vector database built successfully at ./chroma_db!")

if __name__ == "__main__":
    build_vector_db()