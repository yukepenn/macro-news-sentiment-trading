"""
Utilities for fetching headlines from URLs.
"""

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def fetch_headline(url: str, timeout: float = 5.0) -> str:
    """
    Download `url` and return its headline.
    First tries <meta property='og:title'>, then <title>. 
    Returns empty string on any error or if the URL is blank.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # 1) try Open Graph
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()
        # 2) fallback to <title>
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception as e:
        logger.debug(f"fetch_headline failed for {url}: {e}")
    return "" 