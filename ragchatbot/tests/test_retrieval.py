# tests/test_retrieval.py
import pytest
from api.retrieval import list_chapters, is_confident


def test_list_chapters_returns_list():
    """Test that list_chapters returns a list"""
    chapters = list_chapters()
    assert isinstance(chapters, list)


def test_list_chapters_unique():
    """Test that chapters are unique"""
    chapters = list_chapters()
    assert len(chapters) == len(set(chapters))


def test_is_confident_low_distance():
    """Test confidence with low distance"""
    assert is_confident([0.3, 0.5, 0.7]) == True


def test_is_confident_high_distance():
    """Test confidence with high distance"""
    assert is_confident([2.0, 2.5, 3.0]) == False


def test_is_confident_empty():
    """Test confidence with empty list"""
    assert is_confident([]) == False
    assert is_confident(None) == False


def test_is_confident_threshold():
    """Test confidence at threshold boundary"""
    from api.retrieval import DISTANCE_THRESHOLD
    
    assert is_confident([DISTANCE_THRESHOLD - 0.1]) == True
    assert is_confident([DISTANCE_THRESHOLD + 0.1]) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])