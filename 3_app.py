import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
#from langchain.retrievers import EnsembleRetriever

load_dotenv()
# 1. Load the Model & Database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Setup Retrievers
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 7})

db_data = vector_store.get(include=["documents"])
bm25_retriever = BM25Retriever.from_texts(db_data["documents"])
bm25_retriever.k = 7

# 3. Custom Hybrid Search Engine
def get_hybrid_context(query):
    # Strip common Bengali question words for the keyword search engine
    stop_words = ["কে", "কি", "কী", "কোথায়", "এর", "নাম", "আসল", "ধর্ম", "?", "।"]
    clean_query = " ".join([w for w in query.split() if w not in stop_words])
    
    # Run searches (Vector gets full question, BM25 gets cleaned keywords)
    bm25_docs = bm25_retriever.invoke(clean_query)
    vector_docs = vector_retriever.invoke(query)
    
    # Combine results and remove duplicates
    unique_docs = []
    seen = set()
    
    # Prioritize exact keyword matches by adding them first
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
            
    # Format the top 12 combined chunks for the LLM
    return "\n\n".join(doc.page_content for doc in unique_docs[:12])
    
# 6. Setup the LLM
llm = ChatGroq(
    model_name="qwen/qwen3.8-27b",
    temperature=0.3,
    max_tokens=500,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# 7. Create the Prompt
prompt = PromptTemplate.from_template(
    """You are an expert assistant for the Bengali book 'Pather Dabi'. 
    Read the Context below. Answer the Question in Bengali using ONLY the provided Context. 
    If the Question asks about a character, summarize who they are based on their actions or dialogue in the Context.
    If the Context is completely irrelevant, output: "প্রদত্ত তথ্যে এর উত্তর নেই।"
    Do not use outside knowledge.

    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
)

# 8. Build the Modern RAG Chain (Bypasses langchain.chains entirely)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Build the Modern RAG Chain
rag_chain = (
    {"context": get_hybrid_context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- STREAMLIT UI ---
st.title("📚 Pather Dabi AI Agent")
question = st.text_input("Ask a question about the book:")

if st.button("Search"):
    if question:
        with st.spinner("Searching the book..."):
            # --- DEBUG: Print the new hybrid search results ---
            context_found = get_hybrid_context(question)
            print(f"\n\n--- HYBRID CONTEXT FOUND FOR: {question} ---")
            print(context_found[:800] + "...\n") # Prints the first 800 characters
            # -----------------------------------------------------
            
            answer = rag_chain.invoke(question)
            st.write(answer)