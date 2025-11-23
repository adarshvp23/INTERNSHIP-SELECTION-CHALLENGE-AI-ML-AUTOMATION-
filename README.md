# INTERNSHIP-SELECTION-CHALLENGE-AI-ML-AUTOMATION-
#  Hybrid RAG Support Bot (Advanced Retrieval-Augmented Generation System)

##  Internship Challenge Submission
This project is submitted for the **AI & Automation Internship Selection Challenge**, under:

### **Option 1 — The "Hybrid" Support Bot (Advanced RAG)**  
A system that performs accurate document-based question answering using hybrid retrieval, metadata filtering, reranking, and local LLM inference.

---

##  Problem Statement
Traditional RAG systems perform **blind retrieval**, often returning irrelevant results and causing hallucinations.  
The challenge requires building a system that:

- Parses a **technical manual PDF**
- Extracts **chapter/page metadata**
- Uses **hybrid search** (semantic + metadata filtering)
- Runs queries through a **locally hosted LLM**
- Tracks **retrieval vs generation latency**
- Responds **“I don’t know”** when context is insufficient



## Solution Overview
This project implements an **Advanced RAG architecture** with:

| Feature |                              | Status |

| Smart PDF ingestion with metadata      | ✔ Completed |
| Chapter-level filtering                | ✔ Completed |
| Semantic & reranking-based retrieval   | ✔ Completed |
| Local LLM with Ollama                  | ✔ Completed |
| FastAPI backend & Streamlit UI         | ✔ Completed |
| PDF upload & dynamic re-indexing       | ✔ Completed |
| Retrieval vs Generation latency logs   | ✔ Completed |
| Hallucination prevention               | ✔ Completed |




## System Architecture
| Component|        | Technology |

| Backend API       | FastAPI |
| Frontend UI       | Streamlit |
| Vector Database   | FAISS |
| Embeddings        | Cohere / FastEmbed |
| LLM               | Ollama (Mistral / LLaMA3) |
| PDF Parsing       | PyMuPDF (fitz) |

[Architecture Diagram]((assets/architecture.png))





##  How to Run
1) Create Virtual Environment
python -m venv venv
source venv/bin/activate       # mac/linux
venv\Scripts\activate.ps1      # windows

2) Install Dependencies
 pip install -r requirements.txt

3) Start Backend API
uvicorn api.app:app --reload --port 8000

4) Start Frontend UI
streamlit run frontend/streamlit_app.py

## Demonstration video liink 
https://youtu.be/0toEaK_rUjs?si=uus7hC5JpHJyXjO7
