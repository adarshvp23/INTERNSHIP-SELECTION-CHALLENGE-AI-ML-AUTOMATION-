# api/retrieval.py
"""
Enhanced retrieval with metadata filtering, reranking, and Cohere support
"""
import os
import time
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Check which embedding model to use
USE_COHERE = os.getenv("USE_COHERE_EMBEDDINGS", "false").lower() == "true"

if USE_COHERE:
    from utils.embeddings_cohere import embed_texts
    print("[RETRIEVAL] 🚀 Using Cohere embeddings (search_query mode)")
else:
    from utils.embeddings_fastembed import embed_texts
    print("[RETRIEVAL] Using FastEmbed embeddings")

DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", 1.0))
TOP_K = int(os.getenv("TOP_K", 10))
FINAL_K = 3  # Return top 3 after reranking

DATA_META = Path("data/metadata.parquet")
EMB_PATH = Path("data/faiss_index/embeddings.npy")
IDX_PATH = Path("data/faiss_index/index.faiss")

# Global state
metadata = None
embeddings = None
faiss_index = None


def load_metadata():
    """Load metadata parquet file"""
    try:
        if DATA_META.exists():
            df = pd.read_parquet(DATA_META)
            print(f"[Metadata] Loaded {len(df)} chunks")
            
            # Show chapter distribution
            chapter_counts = df["chapter"].value_counts()
            print(f"[Metadata] Found {len(chapter_counts)} unique chapters:")
            for chapter, count in chapter_counts.items():
                print(f"  - '{chapter}': {count} chunks")
            
            return df
        else:
            print("[WARNING] Metadata file not found")
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}")
    
    return pd.DataFrame(columns=["chunk_id", "source", "chapter", "page", "chunk_text", "vector_index"])


def load_embeddings():
    """Load embeddings numpy file"""
    try:
        if EMB_PATH.exists():
            emb = np.load(EMB_PATH)
            print(f"[Embeddings] Loaded shape {emb.shape}")
            return emb
        else:
            print("[WARNING] Embeddings file not found")
    except Exception as e:
        print(f"[ERROR] Failed to load embeddings: {e}")
    
    return None


def load_faiss_index():
    """Load FAISS index"""
    try:
        if IDX_PATH.exists():
            idx = faiss.read_index(str(IDX_PATH))
            print(f"[FAISS] Loaded index with {idx.ntotal} vectors")
            return idx
        else:
            print("[WARNING] FAISS index not found")
    except Exception as e:
        print(f"[ERROR] Failed to load FAISS index: {e}")
    
    return None


def reload_index():
    """Reload all index data"""
    global metadata, embeddings, faiss_index
    
    print("\n[INDEX] Loading index data...")
    metadata = load_metadata()
    embeddings = load_embeddings()
    faiss_index = load_faiss_index()
    
    if metadata is None or len(metadata) == 0:
        print("[WARNING] No metadata loaded - run ingestion first")
    if embeddings is None:
        print("[WARNING] No embeddings loaded - run ingestion first")
    if faiss_index is None:
        print("[WARNING] No FAISS index loaded - run ingestion first")
    else:
        print(f"[INDEX] ✓ Loaded successfully\n")


# Load on module import
reload_index()


def simple_rerank(query: str, candidates: list, distances: list) -> tuple:
    """
    Simple keyword-based reranking to boost relevant chunks.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
                 'how', 'what', 'when', 'where', 'why', 'do', 'does', 'did', 'can',
                 'could', 'should', 'would', 'i', 'you', 'we', 'they', 'it', 'this', 'that'}
    query_words = query_words - stopwords
    
    print(f"\n[RERANK] Query keywords: {query_words}")
    
    # Score each candidate
    scored_candidates = []
    for candidate, distance in zip(candidates, distances):
        chunk_text = candidate["chunk_text"].lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for word in query_words if word in chunk_text)
        
        # Calculate keyword density
        keyword_density = keyword_matches / max(len(query_words), 1)
        
        # Combined score (lower is better)
        reranked_score = distance * 0.7 + (1 - keyword_density) * 0.3
        
        scored_candidates.append({
            "candidate": candidate,
            "original_distance": distance,
            "keyword_matches": keyword_matches,
            "keyword_density": keyword_density,
            "reranked_score": reranked_score
        })
    
    # Sort by reranked score
    scored_candidates.sort(key=lambda x: x["reranked_score"])
    
    # Log reranking results
    print(f"[RERANK] Results:")
    for i, item in enumerate(scored_candidates[:5], 1):
        print(f"  {i}. Chapter='{item['candidate']['chapter']}' "
              f"Page={item['candidate']['page']} "
              f"Dist={item['original_distance']:.3f} "
              f"Keywords={item['keyword_matches']}/{len(query_words)} "
              f"Score={item['reranked_score']:.3f}")
    
    reranked_candidates = [item["candidate"] for item in scored_candidates]
    reranked_scores = [item["reranked_score"] for item in scored_candidates]
    
    return reranked_candidates, reranked_scores


def find_best_chapter_match(chapter_hint: str) -> str:
    """Find the best matching chapter name"""
    global metadata
    
    if metadata is None or len(metadata) == 0:
        return None
    
    available_chapters = metadata["chapter"].unique()
    chapter_hint_lower = chapter_hint.strip().lower()
    
    # Exact match
    for chapter in available_chapters:
        if chapter.lower() == chapter_hint_lower:
            return chapter
    
    # Partial match
    for chapter in available_chapters:
        if chapter_hint_lower in chapter.lower() or chapter.lower() in chapter_hint_lower:
            print(f"[CHAPTER MATCH] Fuzzy matched '{chapter_hint}' → '{chapter}'")
            return chapter
    
    print(f"[CHAPTER MATCH] No match for '{chapter_hint}'")
    return None


def search(query: str, chapter_filter: str = None, top_k: int = None):
    """
    Enhanced search with reranking and semantic validation.
    """
    global metadata, embeddings, faiss_index
    
    final_k = top_k or FINAL_K
    fetch_k = max(TOP_K, final_k * 3)
    
    if faiss_index is None or embeddings is None or metadata is None or len(metadata) == 0:
        print("[ERROR] Index not loaded")
        return [], [999.0], 0.0
    
    try:
        # Embed query
        t0 = time.perf_counter()
        
        if USE_COHERE:
            q_emb = embed_texts(query, input_type="search_query")
        else:
            q_emb = embed_texts(query)
        
        embed_time = time.perf_counter() - t0
        
        print(f"\n{'='*60}")
        print(f"[SEARCH] Query: '{query}'")
        print(f"[SEARCH] Embedding time: {embed_time:.3f}s")
        
        # Handle chapter filtering
        if chapter_filter:
            print(f"[SEARCH] 🔍 METADATA FILTER: '{chapter_filter}'")
            matched_chapter = find_best_chapter_match(chapter_filter)
            
            if matched_chapter is None:
                print(f"[SEARCH] ⚠️  Chapter not found, using full search")
                chapter_filter = None
            else:
                print(f"[SEARCH] ✓ Using chapter: '{matched_chapter}'")
                chapter_filter = matched_chapter
        
        # Perform search
        t_search_start = time.perf_counter()
        
        if chapter_filter:
            mask = metadata["chapter"] == chapter_filter
            candidate_indices = metadata[mask]["vector_index"].values
            
            print(f"[SEARCH] 📊 Filtered to {len(candidate_indices)} chunks")
            
            if len(candidate_indices) == 0:
                return [], [999.0], time.perf_counter() - t0
            
            filtered_embeddings = embeddings[candidate_indices]
            distances = np.linalg.norm(filtered_embeddings - q_emb[0], axis=1)
            
            k = min(fetch_k, len(distances))
            top_indices = np.argpartition(distances, k-1)[:k]
            top_indices = top_indices[np.argsort(distances[top_indices])]
            
            I = np.array([candidate_indices[top_indices]])
            D = np.array([distances[top_indices]])
        else:
            print(f"[SEARCH] 🌐 Searching all {len(metadata)} chunks")
            D, I = faiss_index.search(q_emb, fetch_k)
        
        search_time = time.perf_counter() - t_search_start
        
        # Build candidates
        candidates = []
        for idx in I[0]:
            if 0 <= idx < len(metadata):
                row = metadata.iloc[idx]
                candidates.append({
                    "chunk_id": str(row.get("chunk_id", "")),
                    "source": str(row.get("source", "")),
                    "chapter": str(row.get("chapter", "Unknown")),
                    "page": int(row.get("page", 0)),
                    "chunk_text": str(row.get("chunk_text", ""))
                })
        
        distances = D[0].tolist()
        
        print(f"[SEARCH] ⏱️  Search: {search_time:.3f}s, found {len(candidates)} candidates")
        
        # Rerank
        t_rerank = time.perf_counter()
        reranked_candidates, reranked_scores = simple_rerank(query, candidates, distances)
        rerank_time = time.perf_counter() - t_rerank
        
        print(f"[SEARCH] ⏱️  Reranking: {rerank_time:.3f}s")
        
        # Take top final_k
        final_results = reranked_candidates[:final_k]
        final_scores = reranked_scores[:final_k]
        
        total_time = time.perf_counter() - t0
        
        print(f"[SEARCH] ✅ Final results (top {final_k}):")
        for i, (result, score) in enumerate(zip(final_results, final_scores), 1):
            print(f"  {i}. Chapter='{result['chapter']}' Page={result['page']} Score={score:.3f}")
            print(f"     Preview: {result['chunk_text'][:100]}...")
        
        print(f"[SEARCH] ⏱️  Total: {total_time:.3f}s")
        print(f"{'='*60}\n")
        
        return final_results, final_scores, total_time
    
    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        import traceback
        traceback.print_exc()
        return [], [999.0], 0.0


def list_chapters():
    """Get list of unique chapters"""
    global metadata
    
    if metadata is None or len(metadata) == 0:
        return []
    
    try:
        chapters = metadata["chapter"].dropna().unique().tolist()
        chapters = [str(ch).strip() for ch in chapters if str(ch).strip()]
        return sorted(chapters)
    except Exception as e:
        print(f"[ERROR] Failed to list chapters: {e}")
        return []


def is_confident(scores: list, results: list, query: str) -> bool:
    """
    Enhanced confidence check with multiple signals.
    
    FIXED: Relaxed margin check from 0.03 to 0.005 for Cohere embeddings
    """
    if not scores or len(scores) == 0 or not results or len(results) == 0:
        print("[CONFIDENCE] ❌ No results")
        return False
    
    top_score = float(scores[0])
    top_chunk = results[0]["chunk_text"].lower()
    query_lower = query.lower()
    
    # Extract keywords
    query_words = set(query_lower.split())
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
                 'how', 'what', 'when', 'where', 'why', 'do', 'does', 'did'}
    query_words = query_words - stopwords
    
    # Check 1: Score threshold
    score_ok = top_score < DISTANCE_THRESHOLD
    
    # Check 2: Keyword presence (at least 20% - relaxed for Cohere)
    keyword_matches = sum(1 for word in query_words if word in top_chunk)
    keyword_ratio = keyword_matches / max(len(query_words), 1)
    keyword_ok = keyword_ratio >= 0.2
    
    # Check 3: Margin (FIXED - relaxed from 0.03 to 0.005)
    if len(scores) > 1:
        margin = scores[1] - scores[0]
        margin_ok = margin > 0.005  # ← CHANGED FROM 0.03
    else:
        margin_ok = True
    
    confident = score_ok and keyword_ok and margin_ok
    
    print(f"\n{'='*60}")
    print(f"[CONFIDENCE CHECK]")
    print(f"  Top score:       {top_score:.4f} (threshold: {DISTANCE_THRESHOLD:.4f}) {'✓' if score_ok else '✗'}")
    print(f"  Keywords:        {keyword_matches}/{len(query_words)} ({keyword_ratio:.1%}) {'✓' if keyword_ok else '✗'}")
    if len(scores) > 1:
        print(f"  Margin:          {margin:.4f} {'✓' if margin_ok else '✗'}")
    print(f"  Final:           {'✓ CONFIDENT' if confident else '✗ NOT CONFIDENT'}")
    print(f"{'='*60}\n")
    
    return confident