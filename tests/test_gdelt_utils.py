"""
Tests for GDELT utilities.
"""

import pytest
from utils.gdelt_utils import GDELTClient

def test_gdelt_client_initialization():
    """Test GDELT client initialization."""
    client = GDELTClient()
    assert client.base_url == "http://data.gdeltproject.org/gdeltv2"
    assert client.last_updated is None 