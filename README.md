# PDF Chatbot with RAG

AI-powered PDF chatbot using Retrieval-Augmented Generation.

## Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM
- Streamlit

## Architecture

PDF → Chunking → Embeddings → FAISS Vector DB → Retriever → LLM → Streamlit UI

## Features

- PDF document loading
- Text chunking
- Semantic search with FAISS
- LLM-based answers
- Source page references
- Streamlit web interface

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py