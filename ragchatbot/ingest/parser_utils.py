# ingest/parser_utils.py - FIXED VERSION

import fitz
import re
from typing import Generator, List, Dict


def detect_heading(block: dict, page_num: int = 0) -> bool:
    """
    STRICT heading detection - only real chapter/section headings.
    
    This fixes the over-detection problem where page numbers, 
    document IDs, and random text were marked as chapters.
    """
    if "lines" not in block:
        return False
    
    text = ""
    for line in block["lines"]:
        for span in line["spans"]:
            text += span.get("text", "") + " "
    
    text = text.strip()
    
    # Basic validation - headings should be 3-100 characters
    if not text or len(text) < 3 or len(text) > 100:
        return False
    
    # Skip common false positives
    # 1. Page numbers like "page 17" or "pg. 5"
    if re.match(r'^(page|pg\.?)\s*\d+', text, re.IGNORECASE):
        return False
    
    # 2. Document IDs like "GI13-5699-00"
    if re.match(r'^[A-Z]{2}\d{2}-\d{4}', text):
        return False
    
    # 3. Common table/list headers
    if text.lower() in ['contents', 'figures', 'tables', 'index', 'preface']:
        return False
    
    # 4. Copyright notices
    if 'copyright' in text.lower() or '©' in text:
        return False
    
    # 5. "Installation Instructions 17" type patterns (section + page number)
    if re.match(r'.+\s+\d{1,2}$', text):
        return False
    
    # STRICT PATTERNS - Only match proper IBM document section numbers
    # These patterns match the actual structure of IBM manuals
    patterns = [
        r'^\d+\.\d+\s+[A-Z]',           # "1.0 Introduction", "5.2 Requirements"
        r'^\d+\.\d+\.\d+\s+[A-Z]',      # "5.2.1 Machine Requirements"
        r'^\d+\.\d+\.\d+\.\d+\s+[A-Z]', # "5.2.2.1 Installation Requisites"
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            print(f"[HEADING DETECTED] Page {page_num + 1}: '{text[:60]}'")
            return True
    
    # Check for unnumbered major sections (only if all caps and short)
    if len(text) < 50 and text.isupper() and len(text.split()) <= 5:
        # But exclude common headers/footers
        exclude_words = ['ibm', 'notice', 'warning', 'caution', 'note']
        if not any(word in text.lower() for word in exclude_words):
            print(f"[HEADING DETECTED - CAPS] Page {page_num + 1}: '{text[:60]}'")
            return True
    
    return False


def normalize_chapter_name(chapter: str) -> str:
    """
    Normalize chapter names for consistent matching.
    
    Converts "1.0 Introduction" → "Introduction"
    Converts "5.2.1 Machine Requirements" → "Machine Requirements"
    """
    # Remove leading section numbers (e.g., "1.0 ", "5.2.1 ")
    chapter = re.sub(r'^\d+(\.\d+)*\s+', '', chapter)
    
    # Remove common prefixes
    chapter = re.sub(r'^(chapter|section|part|appendix)\s*:?\s*', '', chapter, flags=re.IGNORECASE)
    
    # Clean up whitespace
    chapter = ' '.join(chapter.split())
    
    # Title case for consistency
    chapter = chapter.strip().title()
    
    return chapter


def extract_sections_from_document(doc) -> List[Dict]:
    """
    Extract sections from PDF with improved chapter detection.
    
    Now uses strict heading detection to avoid false positives.
    """
    sections = []
    current_chapter = "Introduction"  # Default fallback
    
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
            
            page_text_parts = []
            page_heading = None
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                # Extract text from block
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span.get("text", "") + " "
                
                block_text = block_text.strip()
                
                if not block_text:
                    continue
                
                # Check if this is a heading
                if detect_heading(block, page_num):
                    candidate = block_text.split("\n")[0].strip()
                    
                    # Validate heading quality
                    if 3 < len(candidate) < 100:
                        normalized = normalize_chapter_name(candidate)
                        
                        # Must have substance after normalization
                        if len(normalized) > 2:
                            page_heading = normalized
                            print(f"[CHAPTER UPDATE] Page {page_num + 1}: '{page_heading}'")
                
                # Add text to page content
                page_text_parts.append(block_text)
            
            # Update current chapter if we found a new heading on this page
            if page_heading:
                current_chapter = page_heading
            
            # Combine all text from this page
            full_text = "\n".join(page_text_parts).strip()
            
            # Skip empty pages
            if not full_text:
                continue
            
            # Limit very long pages
            MAX_PAGE_LENGTH = 50000
            if len(full_text) > MAX_PAGE_LENGTH:
                print(f"[WARNING] Page {page_num + 1} has {len(full_text)} chars, truncating")
                full_text = full_text[:MAX_PAGE_LENGTH]
            
            # Add section
            sections.append({
                "page": page_num + 1,
                "chapter": current_chapter,
                "text": full_text
            })
        
        except Exception as e:
            print(f"[ERROR] Failed to process page {page_num + 1}: {e}")
            continue
    
    print(f"\n[EXTRACTION SUMMARY]")
    print(f"  Total pages processed: {len(sections)}")
    print(f"  Unique chapters: {len(set(s['chapter'] for s in sections))}")
    
    return sections


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> Generator[str, None, None]:
    """
    Improved text chunking with better sentence boundary detection.
    
    Key improvements:
    - Respects sentence boundaries
    - Preserves paragraph structure
    - Handles technical content better
    """
    if not text or not text.strip():
        return
    
    # Validate parameters
    if overlap >= chunk_size:
        print(f"[WARNING] Overlap ({overlap}) >= chunk_size ({chunk_size}), adjusting")
        overlap = chunk_size // 4
    
    # Normalize whitespace but preserve paragraphs
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n\n+', '\n\n', text)  # Multiple newlines → double newline
    text = text.strip()
    
    # If text is small enough, return as-is
    if len(text) <= chunk_size:
        yield text
        return
    
    # Split into sentences (improved regex)
    # This handles common sentence endings better
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    
    # Build chunks from sentences
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds chunk_size
        if current_length + sentence_len > chunk_size and current_chunk:
            # Yield current chunk
            chunk_text = ' '.join(current_chunk)
            yield chunk_text
            
            # Start new chunk with overlap
            # Keep last few sentences for context
            overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
            
            # Find sentence boundaries in overlap
            overlap_sentences = re.split(sentence_endings, overlap_text)
            current_chunk = [s.strip() for s in overlap_sentences if s.strip()]
            current_length = sum(len(s) for s in current_chunk)
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_len
    
    # Yield final chunk
    if current_chunk:
        yield ' '.join(current_chunk)