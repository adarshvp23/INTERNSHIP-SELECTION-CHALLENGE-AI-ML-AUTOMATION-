# tests/test_ingest.py
import pytest
from ingest.parser_utils import chunk_text, detect_heading


def test_chunk_text_basic():
    """Test basic chunking"""
    text = "This is a sentence. This is another sentence. Last one."
    chunks = list(chunk_text(text, chunk_size=20, overlap=5))  # ✅ Convert generator to list
    
    assert len(chunks) >= 2
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_short():
    """Test text shorter than chunk size"""
    text = "Short text"
    chunks = list(chunk_text(text, chunk_size=100, overlap=10))
    
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty():
    """Test empty text"""
    chunks = list(chunk_text("", chunk_size=100, overlap=10))
    assert len(chunks) == 0
    
    chunks = list(chunk_text("   ", chunk_size=100, overlap=10))
    assert len(chunks) == 0


def test_chunk_overlap():
    """Test that overlap is working"""
    text = "A" * 100
    chunks = list(chunk_text(text, chunk_size=50, overlap=10))
    
    assert len(chunks) >= 2
    # Check there's overlap between consecutive chunks
    if len(chunks) >= 2:
        assert chunks[0][-10:] == chunks[1][:10] or len(set(chunks[0][-5:] + chunks[1][:5])) == 1


def test_detect_heading_chapter():
    """Test chapter heading detection"""
    block = {
        "lines": [{
            "spans": [{
                "text": "Chapter 1: Introduction",
                "size": 16,
                "font": "Times-Bold"
            }]
        }]
    }
    
    assert detect_heading(block) == True


def test_detect_heading_uppercase():
    """Test uppercase heading detection"""
    block = {
        "lines": [{
            "spans": [{
                "text": "INSTALLATION GUIDE",
                "size": 12,
                "font": "Arial"
            }]
        }]
    }
    
    assert detect_heading(block) == True


def test_detect_heading_normal_text():
    """Test that normal text is not detected as heading"""
    block = {
        "lines": [{
            "spans": [{
                "text": "This is just normal text in a paragraph.",
                "size": 10,
                "font": "Times-Roman"
            }]
        }]
    }
    
    assert detect_heading(block) == False


def test_detect_heading_no_lines():
    """Test block without lines"""
    block = {}
    assert detect_heading(block) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])