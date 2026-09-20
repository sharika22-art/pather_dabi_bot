import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_eval_db(chunk_size, persist_dir):
    print(f"\nBuilding temporary DB for Chunk Size: {chunk_size}...")
    with open("data/pather_dabi.json", "r", encoding="utf-8") as f:
        book_data = json.load(f)

    documents = [
        Document(
            page_content=chapter["content"],
            metadata={"chapter": chapter["chapter_name"]}
        ) for chapter in book_data
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "।", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=50, # smaller overlap for testing
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    return vector_store

def evaluate_hit_rate():
    # Generate DBs
    db_400 = build_eval_db(400, "./chroma_db_400")
    db_800 = build_eval_db(800, "./chroma_db_800")
    
    # Ground Truth Test Set: (Question, Expected Chapter in Bengali Numerals)
    # Note: These are example pairs. You can update these as you read through your generated JSON.
    test_set = [
        ("অপূর্ব্ব রেলওয়ে স্টেশনের দিকে যাচ্ছিল কেন?", "১১"),
        ("পুলিশের কাছে চুরির ব্যাপার গোচর করে ফল নেই বলে মনে হলো কেন?", "৬"),
        ("সুমিত্রার আগমন সংবাদ কেন অপ্রত্যাশিত ছিল?", "২৮")
    ]
    
    print("\n--- Evaluating Hit Rates ---")
    for name, db in [("Chunk Size 400", db_400), ("Chunk Size 800", db_800)]:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        hits = 0
        
        for query, expected_chapter in test_set:
            docs = retriever.invoke(query)
            # Extract chapters from top 3 results
            retrieved_chapters = [doc.metadata["chapter"] for doc in docs]
            
            if expected_chapter in retrieved_chapters:
                hits += 1
                
        hit_rate = (hits / len(test_set)) * 100
        print(f"Approach: {name} | Hit Rate: {hit_rate:.2f}%")

if __name__ == "__main__":
    evaluate_hit_rate()