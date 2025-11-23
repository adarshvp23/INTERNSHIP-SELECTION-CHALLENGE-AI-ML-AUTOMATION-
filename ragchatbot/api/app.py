from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
import time

from ingest.ingest_pdf import ingest_bytes
from api.retrieval import search, list_chapters, is_confident, reload_index
from api.generate import llm_answer, test_ollama_connection

app = FastAPI(
    title="Hybrid RAG Support Bot",
    description="Advanced RAG system with metadata filtering and reranking",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Load index and test connections on startup"""
    print("\n" + "="*60)
    print("🚀 Starting Hybrid RAG Support Bot v2.0")
    print("="*60 + "\n")
    

    reload_index()
    
    
    print("\n[Startup] Testing Ollama connection...")
    if test_ollama_connection():
        print("[Startup] ✓ Ollama connected\n")
    else:
        print("[Startup] ⚠️  Ollama not available - answers will fail\n")


class QueryRequest(BaseModel):
    query: str
    chapter_hint: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    retrieval_time: float
    generation_time: float
    chunks: list
    confident: bool


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Hybrid RAG Support Bot",
        "version": "2.0.0",
        "features": ["reranking", "metadata_filtering", "semantic_validation"]
    }


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF file.
    
    This will replace the current index.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file
        pdf_bytes = await file.read()
        filename = file.filename or "uploaded.pdf"
        
        print(f"\n[API] Received PDF upload: {filename} ({len(pdf_bytes)} bytes)")
        
        # Ingest
        manifest = ingest_bytes(pdf_bytes, filename)
        
        # Reload index
        reload_index()
        
        return {
            "status": "success",
            "filename": filename,
            "manifest": manifest
        }
    
    except Exception as e:
        print(f"[API ERROR] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/chapters")
def get_chapters():
    """Get list of available chapters/sections"""
    try:
        chapters = list_chapters()
        return {
            "chapters": chapters,
            "count": len(chapters)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    """
    Query the RAG system with optional chapter filtering.
    
    Args:
        query: The user's question
        chapter_hint: Optional chapter name to filter search
    
    Returns:
        Answer with performance metrics
    """
    try:
        # Validate input
        if not body.query or not body.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        print(f"\n[API] Query: '{body.query}'")
        if body.chapter_hint:
            print(f"[API] Chapter filter: '{body.chapter_hint}'")
        
        # Retrieval (now with reranking)
        results, scores, retrieval_time = search(body.query, body.chapter_hint)
        
        # Check confidence (now with enhanced checks)
        confident = is_confident(scores, results, body.query)
        
        # Generation
        if not confident or not results:
            answer = "I don't know — the manual does not contain this information."
            gen_time = 0.0
            print("[API] Low confidence or no results - returning 'I don't know'")
        else:
            # Show what we're sending to LLM
            print(f"[API] Sending {len(results)} chunks to LLM:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['chapter']} (page {r['page']}): {r['chunk_text'][:80]}...")
            
            answer, gen_time = llm_answer(body.query, results, max_tokens=256, temperature=0.0)
        
        # Logging
        log_entry = {
            "timestamp": time.time(),
            "query": body.query,
            "chapter_hint": body.chapter_hint,
            "retrieval_time": retrieval_time,
            "generation_time": gen_time,
            "top_score": float(scores[0]) if scores else None,
            "confident": confident,
            "num_chunks": len(results)
        }
        
        # Append to log file
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "queries.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"[API] ✓ Query completed: retrieval={retrieval_time:.3f}s gen={gen_time:.3f}s")
        
        return QueryResponse(
            answer=answer,
            retrieval_time=retrieval_time,
            generation_time=gen_time,
            chunks=results,
            confident=confident
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API ERROR] Query failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Get system statistics"""
    try:
        from api.retrieval import metadata, faiss_index
        
        stats = {
            "index_loaded": metadata is not None and faiss_index is not None,
            "num_chunks": len(metadata) if metadata is not None else 0,
            "num_chapters": len(list_chapters()),
            "faiss_vectors": faiss_index.ntotal if faiss_index else 0
        }
        
        # Load manifest if exists
        manifest_path = Path("data/ingest_manifest.json")
        if manifest_path.exists():
            with open(manifest_path) as f:
                stats["manifest"] = json.load(f)
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/query")
def debug_query(query: str):
    """
    Debug endpoint to see raw retrieval results.
    
    Usage: GET /debug/query?query=your+question
    """
    try:
        results, scores, retrieval_time = search(query)
        
        return {
            "query": query,
            "retrieval_time": retrieval_time,
            "num_results": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "score": score,
                    "chapter": r["chapter"],
                    "page": r["page"],
                    "text_preview": r["chunk_text"][:200] + "..." if len(r["chunk_text"]) > 200 else r["chunk_text"]
                }
                for i, (r, score) in enumerate(zip(results, scores))
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)