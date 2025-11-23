# ingest/ingest_pdf.py
import os
import json
import time
import shutil
import numpy as np
import pandas as pd
import fitz

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from ingest.parser_utils import extract_sections_from_document, chunk_text

load_dotenv()

# Check which embedding model to use
USE_COHERE = os.getenv("USE_COHERE_EMBEDDINGS", "false").lower() == "true"

if USE_COHERE:
    from utils.embeddings_cohere import embed_texts
    print("[INGESTION] 🚀 Using Cohere embeddings (1024 dim, high quality)")
    EMBED_MODEL_NAME = "cohere/embed-english-v3.0"
else:
    from utils.embeddings_fastembed import embed_texts
    print("[INGESTION] Using FastEmbed embeddings")
    EMBED_MODEL_NAME = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------- CONFIG ----------------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

DATA_DIR = Path("data")
FAISS_DIR = DATA_DIR / "faiss_index"

DATA_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


def safe_clear_directory(directory: Path):
    """Safely clear directory (Windows-compatible)"""
    if directory.exists():
        for item in directory.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"[WARNING] Could not delete {item}: {e}")
    else:
        directory.mkdir(parents=True, exist_ok=True)


def ingest_bytes(pdf_bytes: bytes, filename: str, chunk_size: int = None, overlap: int = None):
    """
    Complete ingestion pipeline with progress tracking and error handling.
    
    Args:
        pdf_bytes: PDF file as bytes
        filename: Name of the PDF file
        chunk_size: Override default chunk size
        overlap: Override default overlap
    
    Returns:
        Manifest dict with ingestion metadata
    """
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    
    print(f"\n{'='*60}")
    print(f"[INGEST] Starting ingestion: {filename}")
    print(f"[INGEST] Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"[INGEST] Embedding model: {EMBED_MODEL_NAME}")
    print(f"{'='*60}\n")
    
    t0 = time.perf_counter()
    
    # Open PDF
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        print(f"[INGEST] ✓ Opened PDF: {len(doc)} pages")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")
    
    # Extract sections
    try:
        print("[INGEST] Extracting sections...")
        sections = extract_sections_from_document(doc)
        doc.close()  # Close ASAP to free memory
        print(f"[INGEST] ✓ Extracted {len(sections)} page sections")
        
        if not sections:
            raise ValueError("No text extracted from PDF. It may be scanned/image-based.")
        
        # Show detected chapters
        unique_chapters = sorted(set(s["chapter"] for s in sections))
        print(f"[INGEST] ✓ Detected {len(unique_chapters)} chapters:")
        for ch in unique_chapters:
            print(f"    - {ch}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to extract sections: {e}")
    
    # Clear old data
    print("[INGEST] Clearing old index data...")
    safe_clear_directory(FAISS_DIR)
    
    # Generate chunks
    print("[INGEST] Generating text chunks...")
    metadata_records = []
    all_chunk_texts = []
    
    for sec in tqdm(sections, desc="Chunking", unit="page"):
        for chunk in chunk_text(sec["text"], chunk_size=chunk_size, overlap=overlap):
            metadata_records.append({
                "chunk_id": f"{filename}::p{sec['page']}::c{len(metadata_records)}",
                "source": filename,
                "chapter": sec["chapter"],
                "page": sec["page"],
                "chunk_text": chunk
            })
            all_chunk_texts.append(chunk)
    
    total_chunks = len(all_chunk_texts)
    print(f"[INGEST] ✓ Generated {total_chunks} chunks")
    
    if total_chunks == 0:
        raise ValueError("No chunks generated. PDF may be empty or unreadable.")
    
    # Embed chunks in batches
    print(f"[INGEST] Embedding {total_chunks} chunks...")
    all_embeddings = []
    
    if USE_COHERE:
        # Cohere can handle larger batches (up to 96)
        COHERE_BATCH = 90
        for i in tqdm(range(0, total_chunks, COHERE_BATCH), desc="Embedding", unit="batch"):
            batch = all_chunk_texts[i:i + COHERE_BATCH]
            try:
                batch_emb = embed_texts(batch, input_type="search_document")
                all_embeddings.append(batch_emb)
            except Exception as e:
                print(f"\n[ERROR] Failed to embed batch {i // COHERE_BATCH}: {e}")
                raise
    else:
        # FastEmbed uses smaller batches
        for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Embedding", unit="batch"):
            batch = all_chunk_texts[i:i + BATCH_SIZE]
            try:
                batch_emb = embed_texts(batch)
                all_embeddings.append(batch_emb)
            except Exception as e:
                print(f"\n[ERROR] Failed to embed batch {i // BATCH_SIZE}: {e}")
                raise
    
    # Combine embeddings
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"[INGEST] ✓ Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("[INGEST] Building FAISS index...")
    try:
        import faiss
        
        dim = embeddings.shape[1]
        
        # Use HNSW for better search performance
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16
        
        index.add(embeddings)
        print(f"[INGEST] ✓ FAISS index built: {index.ntotal} vectors")
        
        # Save index
        index_path = FAISS_DIR / "index.faiss"
        faiss.write_index(index, str(index_path))
        print(f"[INGEST] ✓ Saved index to {index_path}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to build FAISS index: {e}")
    
    # Save embeddings
    try:
        emb_path = FAISS_DIR / "embeddings.npy"
        np.save(str(emb_path), embeddings)
        print(f"[INGEST] ✓ Saved embeddings to {emb_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save embeddings: {e}")
    
    # Save metadata
    try:
        df = pd.DataFrame(metadata_records)
        df["vector_index"] = list(range(len(df)))
        
        meta_path = DATA_DIR / "metadata.parquet"
        df.to_parquet(meta_path, index=False)
        print(f"[INGEST] ✓ Saved metadata to {meta_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save metadata: {e}")
    
    # Save manifest
    total_time = time.perf_counter() - t0
    manifest = {
        "filename": filename,
        "num_chunks": len(df),
        "num_pages": len(sections),
        "num_chapters": len(unique_chapters),
        "embed_model": EMBED_MODEL_NAME,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": dim,
        "total_time_seconds": round(total_time, 2),
        "timestamp": time.time()
    }
    
    manifest_path = DATA_DIR / "ingest_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"[INGEST] ✅ Ingestion completed successfully!")
    print(f"[INGEST] Total time: {total_time:.2f}s")
    print(f"[INGEST] Chunks: {manifest['num_chunks']}")
    print(f"[INGEST] Pages: {manifest['num_pages']}")
    print(f"[INGEST] Chapters: {manifest['num_chapters']}")
    print(f"[INGEST] Embedding dimension: {dim}")
    print(f"{'='*60}\n")
    
    return manifest


def ingest_file_path(path: str, **kwargs):
    """Helper to ingest from file path"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    
    return ingest_bytes(pdf_bytes, path.name, **kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.ingest_pdf <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    manifest = ingest_file_path(pdf_path)
    print("\n✅ Manifest:")
    print(json.dumps(manifest, indent=2))