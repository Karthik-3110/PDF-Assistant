# 📘 Academic PDF Assistant

An intelligent web-based assistant that allows users to upload academic PDF documents and interact with them using natural language questions.  
The system uses **Retrieval-Augmented Generation (RAG)** to provide accurate, fast, and context-aware answers.

---

## 🚀 Features

- 📄 Upload and process academic PDF files
- 🔢 Displays total number of pages
- 🔍 Semantic search using FAISS
- 🧠 Context-aware question answering
- ⚡ Ultra-fast responses using Groq Cloud (LLaMA-3.1)
- 💬 Continuous chat interface
- 🤖 Hybrid answering (PDF content + model reasoning)

---

## 🧠 How It Works (Architecture)

PDF Upload
↓
Text Extraction (PyPDFLoader)
↓
Text Chunking
↓
Embeddings (Sentence Transformers)
↓
FAISS Vector Store
↓
Relevant Context Retrieval
↓
Groq LLaMA-3.1 Model
↓
Chat Response


