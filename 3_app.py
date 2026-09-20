import streamlit as st
import os
from dotenv import load_dotenv # Add this import
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- CONFIGURATION ---
load_dotenv() 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Retrieve top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize Groq LLM (Llama 3.1 is lightning fast and excellent at Bengali translation/generation)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-70b-versatile", 
        temperature=0
    )

    prompt_template = """
    You are an AI assistant answering questions about the Bengali historical novel "পথের দাবী" (Pather Dabi) by Sarat Chandra Chattopadhyay.
    Use ONLY the following retrieved context to answer the user's question. 
    If the answer is NOT present in the context, you MUST clearly say: "দুঃখিত, এই প্রশ্নের উত্তর বইটিতে পাওয়া যায়নি।" (Sorry, the answer is not found in the book).
    Do NOT make up any information. Answer clearly in Bengali.
    
    Context: {context}
    
    Question: {input}
    
    Answer (in Bengali):
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- UI ---
st.set_page_config(page_title="Pather Dabi Chatbot", page_icon="📖")
st.title("📖 Knowledge Base Chatbot: পথের দাবী")
st.write("Ask questions about Sarat Chandra Chattopadhyay's revolutionary novel *Pather Dabi*.")

if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
    st.warning("⚠️ Please insert your Groq API Key in the code (Line 12) to use the chatbot.")

rag_chain = load_rag_pipeline()

user_query = st.text_input("আপনার প্রশ্ন লিখুন (Ask your question):")

if st.button("Search") and user_query:
    with st.spinner("Groq is searching the book..."):
        try:
            response = rag_chain.invoke({"input": user_query})
            
            st.subheader("Answer:")
            st.write(response["answer"])
            
            # Display Citations directly from metadata
            with st.expander("Sources / Citations"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Source {i+1}:** Chapter {doc.metadata.get('chapter', 'N/A')}")
                    st.markdown(f"**URL:** [Wikisource Link]({doc.metadata.get('url', '#')})")
                    st.caption(f"Snippet: {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"An error occurred: {e}")