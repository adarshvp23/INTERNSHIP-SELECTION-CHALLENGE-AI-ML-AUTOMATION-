# frontend/streamlit_app.py
import streamlit as st
import requests
import json
import time
from pathlib import Path

# ========== CONFIG ==========
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Hybrid RAG Support Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chunk-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 0.5rem;
    }
    .confident {
        color: #28a745;
        font-weight: bold;
    }
    .not-confident {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========== HELPER FUNCTIONS ==========

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_chapters():
    """Get list of chapters from API"""
    try:
        response = requests.get(f"{API_URL}/chapters", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("chapters", [])
    except Exception as e:
        st.error(f"Failed to fetch chapters: {e}")
    return []

def get_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def query_api(query, chapter_hint=None):
    """Send query to API"""
    try:
        payload = {"query": query}
        if chapter_hint and chapter_hint != "All Chapters":
            payload["chapter_hint"] = chapter_hint
        
        response = requests.post(
            f"{API_URL}/query",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try a simpler query.")
        return None
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None

def upload_pdf(file):
    """Upload PDF to API"""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(
            f"{API_URL}/upload_pdf",
            files=files,
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None

# ========== SIDEBAR ==========

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # API Health Check
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.info("Start backend with:\n```\nuvicorn api.app:app --reload\n```")
    
    st.markdown("---")
    
    # System Stats
    st.markdown("## 📊 System Stats")
    stats = get_stats()
    
    if stats:
        st.metric("Index Status", "✅ Loaded" if stats.get("index_loaded") else "❌ Not Loaded")
        st.metric("Total Chunks", stats.get("num_chunks", 0))
        st.metric("Chapters", stats.get("num_chapters", 0))
        
        if "manifest" in stats:
            manifest = stats["manifest"]
            st.markdown("### 📄 Current Document")
            st.info(f"**File:** {manifest.get('filename', 'N/A')}")
            st.info(f"**Pages:** {manifest.get('num_pages', 0)}")
            st.info(f"**Model:** {manifest.get('embed_model', 'N/A').split('/')[-1]}")
    else:
        st.warning("No index loaded")
    
    st.markdown("---")
    
    # PDF Upload
    st.markdown("## 📤 Upload New PDF")
    uploaded_file = st.file_uploader(
        "Choose PDF Manual",
        type=["pdf"],
        help="Upload a technical manual with sections/chapters"
    )
    
    if uploaded_file and st.button("🚀 Ingest PDF", type="primary"):
        with st.spinner("Ingesting PDF... This may take a few minutes."):
            result = upload_pdf(uploaded_file)
            if result:
                st.success(f"✅ Ingested successfully!")
                st.json(result.get("manifest", {}))
                st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown(f"[API Docs]({API_URL}/docs)")
    st.markdown("[GitHub](#)")

# ========== MAIN CONTENT ==========

st.markdown('<div class="main-header">🤖 Hybrid RAG Support Bot</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your technical manual with intelligent metadata filtering.")

# ========== QUERY INTERFACE ==========

# Get available chapters
chapters = get_chapters()
chapter_options = ["All Chapters"] + chapters

# Two columns for input
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "❓ Your Question",
        placeholder="e.g., How do I reset the device?",
        help="Ask any question about the manual"
    )

with col2:
    chapter_filter = st.selectbox(
        "📂 Filter by Chapter",
        options=chapter_options,
        help="Search only within a specific chapter"
    )

# Search button
search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

# ========== QUERY RESULTS ==========

if search_clicked and query:
    chapter_hint = chapter_filter if chapter_filter != "All Chapters" else None
    
    with st.spinner("🔎 Searching and generating answer..."):
        start_time = time.time()
        result = query_api(query, chapter_hint)
        total_time = time.time() - start_time
    
    if result:
        # ========== ANSWER SECTION ==========
        st.markdown("---")
        st.markdown("## 💬 Answer")
        
        answer = result.get("answer", "")
        confident = result.get("confident", False)
        
        # Display answer with confidence indicator
        if confident:
            st.markdown(f'<div class="confident">✅ High Confidence</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="not-confident">⚠️ Low Confidence</div>', unsafe_allow_html=True)
        
        st.markdown(f"### {answer}")
        
        # ========== PERFORMANCE METRICS ==========
        st.markdown("---")
        st.markdown("## ⏱️ Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Retrieval Time",
                f"{result.get('retrieval_time', 0):.3f}s",
                help="Time to search vector database"
            )
        
        with col2:
            st.metric(
                "Generation Time",
                f"{result.get('generation_time', 0):.3f}s",
                help="Time for LLM to generate answer"
            )
        
        with col3:
            st.metric(
                "Total Time",
                f"{total_time:.3f}s",
                help="End-to-end query time"
            )
        
        with col4:
            st.metric(
                "Chunks Retrieved",
                len(result.get('chunks', [])),
                help="Number of relevant text chunks"
            )
        
        # ========== RETRIEVED CHUNKS ==========
        chunks = result.get('chunks', [])
        
        if chunks:
            st.markdown("---")
            st.markdown("## 📚 Retrieved Context")
            
            with st.expander(f"View {len(chunks)} Retrieved Chunks", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"### Chunk {i}")
                    
                    # Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Chapter:** {chunk.get('chapter', 'Unknown')}")
                    with col2:
                        st.markdown(f"**Page:** {chunk.get('page', 'N/A')}")
                    with col3:
                        st.markdown(f"**Source:** {chunk.get('source', 'N/A')}")
                    
                    # Text content
                    st.markdown('<div class="chunk-card">', unsafe_allow_html=True)
                    st.markdown(chunk.get('chunk_text', ''))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if i < len(chunks):
                        st.markdown("---")

elif search_clicked:
    st.warning("Please enter a question")

# ========== EXAMPLE QUERIES ==========

if not search_clicked:
    st.markdown("---")
    st.markdown("## 💡 Example Queries")
    
    example_queries = [
        "How do I install the software?",
        "What are the safety precautions?",
        "How do I troubleshoot connection issues?",
        "What is the warranty policy?",
        "How do I reset to factory settings?"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()

# Handle example query clicks
if hasattr(st.session_state, 'example_query'):
    query = st.session_state.example_query
    del st.session_state.example_query

# ========== FOOTER ==========

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    <p>🤖 Hybrid RAG Support Bot v1.0 | Built with FastAPI, FAISS, Sentence Transformers & Ollama</p>
</div>
""", unsafe_allow_html=True)