# api/generate.py
import os
import time
import json
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def build_prompt(query: str, chunks: List[Dict]) -> str:
    """Build prompt with retrieved context"""
    
    if not chunks:
        return f"""You are a technical support assistant. Answer based ONLY on the provided manual context.

CONTEXT: No relevant information found in the manual.

QUESTION: {query}

ANSWER: I don't know — the manual does not contain this information."""
    
    # Build context section
    context_parts = ["CONTEXT FROM MANUAL:\n"]
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"\n[Chunk {i}]")
        context_parts.append(f"Chapter: {chunk.get('chapter', 'Unknown')}")
        context_parts.append(f"Page: {chunk.get('page', 'N/A')}")
        context_parts.append(f"Content:\n{chunk.get('chunk_text', '')}\n")
    
    context = "\n".join(context_parts)
    
    # Build full prompt
    prompt = f"""You are a precise technical support assistant. Answer based ONLY on the provided manual context.

{context}

IMPORTANT RULES:
1. Use ONLY information from the context above
2. If the answer is in the context, provide it clearly and cite the chapter/page
3. If the answer is NOT in the context, respond EXACTLY with: "I don't know — the manual does not contain this information."
4. Do not make assumptions or use external knowledge
5. Be concise and accurate

QUESTION: {query}

ANSWER:"""
    
    return prompt


def llm_answer(query: str, chunks: List[Dict], max_tokens: int = 256, temperature: float = 0.0):
    """
    Generate answer using Ollama LLM.
    
    Args:
        query: User question
        chunks: Retrieved context chunks
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Tuple of (answer, generation_time)
    """
    prompt = build_prompt(query, chunks)
    
    # Build Ollama API payload
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # Disable streaming for simplicity
        "options": {
            "num_predict": max_tokens,  # Correct parameter name for Ollama
            "temperature": temperature,
            "stop": ["\n\nQUESTION:", "\n\nCONTEXT:"]  # Stop sequences
        }
    }
    
    url = f"{OLLAMA_URL}/api/generate"
    
    try:
        t0 = time.perf_counter()
        response = requests.post(url, json=payload, timeout=120)
        t1 = time.perf_counter()
        gen_time = t1 - t0
        
        response.raise_for_status()
        data = response.json()
        
        # Extract answer from response
        answer = data.get("response", "").strip()
        
        if not answer:
            answer = "Error: No response from model"
        
        print(f"[LLM] Generated answer in {gen_time:.2f}s ({len(answer)} chars)")
        
        return answer, gen_time
    
    except requests.exceptions.Timeout:
        print("[LLM ERROR] Request timed out")
        return "Error: LLM request timed out after 120s", 0.0
    
    except requests.exceptions.ConnectionError:
        print(f"[LLM ERROR] Could not connect to Ollama at {OLLAMA_URL}")
        return f"Error: Could not connect to Ollama. Is it running at {OLLAMA_URL}?", 0.0
    
    except requests.exceptions.RequestException as e:
        print(f"[LLM ERROR] Request failed: {e}")
        return f"Error: LLM request failed - {str(e)}", 0.0
    
    except Exception as e:
        print(f"[LLM ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 0.0


def test_ollama_connection():
    """Test if Ollama is accessible and model is available"""
    try:
        # Test basic connection
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        
        print(f"[Ollama] Connected! Available models: {model_names}")
        
        if OLLAMA_MODEL not in model_names:
            print(f"[WARNING] Model '{OLLAMA_MODEL}' not found. Available: {model_names}")
            return False
        
        return True
    
    except Exception as e:
        print(f"[Ollama ERROR] Connection test failed: {e}")
        return False