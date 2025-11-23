# utils/embeddings_fastembed.py
"""
FREE embeddings using Cohere API (no DLL dependencies!)
- 1000 requests/minute free tier
- No PyTorch or ONNX needed
- Pure Python (just uses requests)
- Fast and reliable
"""
import os
import numpy as np
from typing import List, Union
from dotenv import load_dotenv
import cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError(
        "\n❌ COHERE_API_KEY not found in .env file!\n\n"
        "Get your FREE API key:\n"
        "1. Sign up: https://dashboard.cohere.com/register\n"
        "2. Copy API key\n"
        "3. Add to .env: COHERE_API_KEY=your_key_here\n"
    )

# Create Cohere client (cached)
_cohere_client = None


def get_client():
    """Get or create Cohere client"""
    global _cohere_client
    if _cohere_client is None:
        print("[Cohere] Initializing client...")
        _cohere_client = cohere.Client(COHERE_API_KEY)
        print("[Cohere] ✅ Client ready")
    return _cohere_client


def embed_texts(texts: Union[str, List[str]], use_api: bool = True) -> np.ndarray:
    """
    Embed texts using Cohere API (FREE tier: 1000 req/min).
    
    Args:
        texts: Single text string or list of texts to embed
        use_api: Compatibility parameter (always True)
    
    Returns:
        numpy array of shape (N, 384) containing embeddings
    
    Example:
        >>> embeddings = embed_texts(["Hello world", "Test text"])
        >>> embeddings.shape
        (2, 384)
    """
    # Normalize input to list
    if isinstance(texts, str):
        texts = [texts]
    
    if not texts:
        raise ValueError("No texts provided for embedding")
    
    # Get client
    client = get_client()
    
    # Log progress
    print(f"[Cohere] Embedding {len(texts)} texts via API...")
    
    try:
        # Call Cohere API
        response = client.embed(
            texts=texts,
            model="embed-english-light-v3.0",  # Fast, 384 dims, FREE
            input_type="search_document",      # For indexing documents
            truncate="END"                      # Truncate if too long
        )
        
        # Convert to numpy array
        embeddings = np.array(response.embeddings, dtype=np.float32)
        
        print(f"[Cohere] ✅ Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    except cohere.CohereAPIError as e:
        print(f"[Cohere API ERROR] {e}")
        raise RuntimeError(f"Cohere API failed: {e}")
    except Exception as e:
        print(f"[Cohere ERROR] {e}")
        raise RuntimeError(f"Cohere embedding failed: {e}")


def get_embedding_dim() -> int:
    """
    Get the dimension of embeddings produced by the model.
    
    Returns:
        int: Embedding dimension (384 for embed-english-light-v3.0)
    """
    return 384


# Test function
if __name__ == "__main__":
    print("Testing Cohere embeddings...")
    
    # Test single text
    test_text = "This is a test sentence."
    embedding = embed_texts(test_text)
    print(f"✅ Single text embedding shape: {embedding.shape}")
    
    # Test multiple texts
    test_texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence"
    ]
    embeddings = embed_texts(test_texts)
    print(f"✅ Multiple texts embedding shape: {embeddings.shape}")
    
    print("\n🎉 Cohere is working correctly!")