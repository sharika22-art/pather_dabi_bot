import json
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_db():
    print("Loading scraped book data...")
    if not os.path.exists("data/pather_dabi.json"):
        print("Error: data/pather_dabi.json not found. Run 1_scraper.py first.")
        return

    with open("data/pather_dabi.json", "r", encoding="utf-8") as f:
        book_data = json.load(f)

    documents = []
    for item in book_data:
        doc = Document(
            page_content=item["text"],
            metadata={"chapter": item["chapter"], "source": item["url"]}
        )
        documents.append(doc)

    print("Splitting book into chunks...")
    # Custom separators for Bengali text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "।", " ", ""] 
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split book into {len(chunks)} chunks.")

    print("Loading BAAI/bge-m3 embedding model (this will take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Building ChromaDB vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Vector database built successfully!")

if __name__ == "__main__":
    build_vector_db()