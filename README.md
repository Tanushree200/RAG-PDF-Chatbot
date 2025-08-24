# RAG PDF Chatbot

A **PDF-based RAG (Retrieval-Augmented Generation) Chatbot** built using **Python, Streamlit, LangChain, Google Gemini, and Chroma**.  
Ask questions from your uploaded PDFs and get context-aware answers with citations.

## Link - https://rag-pdf-chatbot-syjgd8m5rzasfkkamklk5m.streamlit.app/

---
<img width="3111" height="2022" alt="image" src="https://github.com/user-attachments/assets/bb845d50-5739-40f6-aeb7-4b1daae754ad" />

## 🚀 Features

Upload multiple PDF files for context-aware question answering.

Uses LangChain for RAG architecture.

Google Gemini LLM for natural language answers.

FAISS embeddings for efficient vector-based document retrieval.

Prompt engineering & few-shot learning to improve response accuracy.

Optional citation support for answers.

Streamlit UI for a simple and interactive chat experience.

## 🛠️ Tech Stack

Python – Core programming language

Streamlit – Web interface for interactive chat

LangChain – RAG and retrieval chains

Google Gemini API – Large language model integration

FAISS – Vector store for document embeddings

PyPDF2 – PDF text extraction


## ⚡ Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/Tanushree200/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot
```
### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```
### 3. Add your API key
GOOGLE_API_KEY = "your_google_gemini_api_key_here"
4. Run the app locally
```bash
streamlit run app.py
```

## Usage

Upload your PDFs from the sidebar.

Click Submit & Process to generate embeddings.

Ask questions in the chat box.

Clear chat history or reset the vectorstore using the sidebar buttons.

