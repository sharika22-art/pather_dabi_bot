# Pather Dabi AI Agent (RAG Chatbot)

## Book Information
* **Book Title:** Pather Dabi (পথের দাবী) by Sarat Chandra Chattopadhyay
* **Bengali Wikisource Link:** [https://bn.wikisource.org/wiki/পথের_দাবী_(শরৎচন্দ্র_চট্টোপাধ্যায়,_১৯৫৮)](https://bn.wikisource.org/wiki/পথের_দাবী_(শরৎচন্দ্র_চট্টোপাধ্যায়,_১৯৫৮))
* **Brief Description:** *Pather Dabi* is a masterpiece Bengali political novel released in 1926. It is based on the story of an underground revolutionary organization in British India and Southeast Asia led by Sabyasachi Mallick (The Doctor), struggling for freedom from India.

## Setup & Running Instructions
* **Required Python Version:** Python 3.9+
* **Installation Steps:**
  1. Clone the repository.
  2. Ensure you have your Groq API key.
  3. Create a `.env` file in the root directory and add: `GROQ_API_KEY=your_key_here`
* **How to Install Dependencies:**
  Run the following command in your terminal:
  `pip install -r requirements.txt`
  *(Note: If building from scratch, requires `langchain`, `langchain-groq`, `langchain-chroma`, `sentence-transformers`, `streamlit`, `rank_bm25`, `python-dotenv`)*
* **How to Run the Chatbot:**
  Run the Streamlit application using:
  `python -m streamlit run 3_app.py`

## Technical Details
* **Embedding Model Used:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Why this embedding model was selected:** It provides strong multilingual support (crucial for Bengali) while remaining incredibly lightweight (under 500MB). This allowed it to run seamlessly within the memory constraints of a free cloud environment (GitHub Codespaces) without crashing.
* **Chunk Size:** 1000 characters
* **Chunk Overlap:** 200 characters
* **Preprocessing Approach:** The raw text was scraped directly from Wikisource, stripping away HTML tags, navigation elements, and wiki-markup. After that, the text was normalized and split using LangChain's `RecursiveCharacterTextSplitter` to maintain paragraph and sentence integrity.
* **Vector Database Used:** ChromaDB (Persistent local storage)
* **Retriever Configuration:** Custom Hybrid Search. The pipeline runs an ensemble retrieval combining AI Vector Search (via ChromaDB for semantic meaning) and BM25 Keyword Search (acting as an exact-match `Ctrl+F` for Bengali names and locations), returning the top 12 most relevant chunks.
* **LLM Used:** `qwen/qwen3.8-27b` via Groq (Note: `llama3-8b-8192` was used as a fallback testing model). Qwen was chosen for its superior multilingual handling.

## RAG Pipeline
**Workflow:** Wikisource → Crawling → Cleaning → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer + Citation.
1. **Wikisource:** Target the *Pather Dabi* index.
2. **Crawling & Cleaning:** Scrape all chapter links, extract the raw Bengali text, and remove web markup.
3. **Chunking:** Split the clean text into overlapping 1000-character segments.
4. **Embeddings & Vector DB:** Convert chunks into mathematical vectors using the multilingual MiniLM model and store them locally in ChromaDB.
5. **Retrieval:** When a user asks a question, the custom Hybrid Retriever searches ChromaDB for semantic matches and uses BM25 for exact keyword matches, filtering out stop-words to handle archaic Bengali grammar.
6. **LLM:** The top retrieved text chunks are injected into a strict prompt.
7. **Answer:** The Groq-hosted Qwen LLM synthesizes the context and outputs a hallucination-free answer in Bengali.

## Bonus Features
* Includes an additional file targeting bonus objectives outlined in the project requirements.