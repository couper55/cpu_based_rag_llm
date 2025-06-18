# 📄 Project Documentation  
**Title:** CPU-Based RAG (Retrieval-Augmented Generation) Inference Pipeline

---

## 🚀 Overview

This project implements a fully modular and CPU-optimized Retrieval-Augmented Generation (RAG) pipeline designed for efficient document retrieval and natural language question answering. It uses lightweight models and minimal dependencies to ensure compatibility on machines without GPU acceleration.

---

## ⚙️ Installation & Setup

### 🔧 Prerequisites
Ensure you have:
- Python 3.9+
- pip
- Git (optional)

### 🧩 Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install fastapi streamlit sentence-transformers faiss-cpu llama-cpp-python PyPDF2 uvicorn matplotlib seaborn pandas
```

> **Note:**  
> You need a compatible `.gguf` model file for `llama.cpp` (e.g., **LLaMA-3-1B**) placed inside a `./models` folder in your current working directory.  
> I have personally used **LLaMA-3-1B** after facing hallucination issues with smaller models like **TinyLLaMA** and **microsoft/DialoGPT-medium**.  
>
> ⚠️ **Upload Functionality Limitation:**  
> The current **upload document** feature in the RAG LLM pipeline works reliably for **1-page documents** only.  
> This is a **proof of concept (PoC)** developed and tested on a machine with **8 GB of RAM**.  
> With increased memory, the solution can be **scaled to support larger documents and multi-file ingestion**.


---

## 📦 Folder Structure

```
├── rag_pipeline_clean.py     # Core RAG implementation
├── fastapiapp.py             # API interface using FastAPI
├── streamlit.py              # Streamlit app for interaction
├── benchmark_rag.py          # Benchmarking module
├── indexing.py               # Manual indexing script
├── models/                   # Folder to store GGUF LLM model files
├── data/                     # Stores FAISS index and document metadata
├── benchmark_results/        # Stores benchmark metrics and plots
```

---

## 🧠 RAG Pipeline Structure

### 1. **Document Preprocessing**
- Documents are cleaned, de-hyphenated, and split into sections.
- Chunking is done with overlap to preserve context (`chunk_size=512`, `overlap=50`).

### 2. **Embedding & Indexing**
- **Embedding Model**: `all-MiniLM-L6-v2` (from `sentence-transformers`).
- **Storage**: FAISS (Inner Product search) with in-memory CPU backend.
- **Persistence**: Index and metadata saved in `./data`.

### 3. **Retrieval**
- Embedding similarity search is used to find top-k (default = 5) relevant chunks.
- Adaptive filtering ensures only high-quality matches are used for generation.

### 4. **LLM Inference**
- **Model**: `llama.cpp` compatible `.gguf` file loaded using `llama-cpp-python`.
- Uses strict prompt construction to ensure context-constrained responses.
- No hallucinations due to enforced rule-based prompting.

### 5. **Interfaces**
- **FastAPI** (`fastapiapp.py`): Provides `/upload` and `/query` endpoints.
- **Streamlit UI** (`streamlit.py`): Interactive web interface with:
  - Query input
  - Performance analytics
  - Document upload
  - Index management

---

## 📈 Benchmarking

Run:

```bash
python benchmark_rag.py
```

This:
- Measures response time, CPU usage, memory, and retrieval quality.
- Visualizes:
  - Response time by category
  - CPU and memory usage
  - Retrieval metrics (doc score vs source coverage)

Output saved in `benchmark_results/` with timestamps.

---

## ⚡ Optimization Techniques

- ✅ CPU-only pipeline (no GPU/TPU needed)
- ✅ Uses `torch.set_num_threads` and `faiss-cpu` for parallelism
- ✅ Minimal memory footprint via chunked PDF/text ingestion
- ✅ `gc.collect()` used aggressively in FastAPI for memory cleanup
- ✅ Streamlit/REST API both allow real-time inference

---

## 🛠️ Usage

### ▶️ Run Streamlit App

```bash
streamlit run streamlit.py
```

### ▶️ Run FastAPI Server

```bash
uvicorn fastapiapp:app --host 0.0.0.0 --port 8000
```

### ▶️ Run Indexing Script

```bash
python indexing.py
```

---

## ✏️ Example Query

Use either Streamlit or API to ask:

> "What are the main types of machine learning?"

Output:  
A structured answer only using the indexed document content with source metadata.

---

## 🧩 Extensibility

- Add support for other formats like `.md`, `.docx` in `upload_document()`.
- Plug in other embedding models (e.g., BGE, GTE) via `RAGConfig`.
- Replace `llama.cpp` with another LLM provider by modifying `generate_response()`.
