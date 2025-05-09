"""
Tests for NLP utilities.
"""

import pytest
from utils.nlp_utils import SentimentAnalyzer

def test_sentiment_analyzer_initialization():
    """Test sentiment analyzer initialization."""
    analyzer = SentimentAnalyzer()
    assert analyzer.device is not None
    assert analyzer.tokenizer is not None
    assert analyzer.model is not None 