"""
Cohere embeddings - High quality embeddings for technical documents
Uses embed-english-v3.0 (1024 dimensions)
"""
import os
import numpy as np
import cohere
from dotenv import load_dotenv

load_dotenv()

# Initialize Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file!")

co = cohere.Client(COHERE_API_KEY)
print(f"[Cohere] ✓ Initialized with API key: {COHERE_API_KEY[:10]}...")


def embed_texts(texts, input_type="search_document"):
    """
    Embed texts using Cohere's embed-english-v3.0 model.
    
    Args:
        texts: str or list of str
        input_type: "search_document" for indexing documents
                   "search_query" for search queries (used in retrieval)
    
    Returns:
        numpy array of embeddings (1024 dimensions)
    """
    # Handle single string
    if isinstance(texts, str):
        texts = [texts]
    
    # Cohere has a limit of 96 texts per request
    MAX_BATCH = 96
    all_embeddings = []
    
    try:
        for i in range(0, len(texts), MAX_BATCH):
            batch = texts[i:i + MAX_BATCH]
            
            response = co.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type=input_type,
                embedding_types=["float"]
            )
            
            batch_embeddings = np.array(response.embeddings.float, dtype=np.float32)
            all_embeddings.append(batch_embeddings)
            
            print(f"[Cohere] Embedded batch {i//MAX_BATCH + 1}: {len(batch)} texts → {batch_embeddings.shape}")
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        return embeddings
    
    except Exception as e:
        print(f"[ERROR] Cohere embedding failed: {e}")
        print(f"[ERROR] First text sample: {texts[0][:100] if texts else 'None'}...")
        raise