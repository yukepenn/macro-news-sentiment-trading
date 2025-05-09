"""
Utilities for handling GDELT data.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GDELTClient:
    """Client for interacting with GDELT data."""
    
    def __init__(self):
        """Initialize GDELT client."""
        self.base_url = "http://data.gdeltproject.org/gdeltv2"
        self.last_updated = None
        
    def fetch_events(self, 
                    start_date: datetime,
                    end_date: Optional[datetime] = None,
                    query: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch GDELT events data for a given date range.
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch (defaults to start_date if None)
            query: Optional query string to filter events
            
        Returns:
            DataFrame containing GDELT events
        """
        if end_date is None:
            end_date = start_date
            
        # Convert dates to GDELT format (YYYYMMDDHHMMSS)
        start_str = start_date.strftime("%Y%m%d%H%M%S")
        end_str = end_date.strftime("%Y%m%d%H%M%S")
        
        try:
            # Construct URL for GDELT API
            url = f"{self.base_url}/events/export"
            params = {
                "format": "json",
                "startdatetime": start_str,
                "enddatetime": end_str
            }
            if query:
                params["query"] = query
                
            # Make request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            df = pd.DataFrame(data)
            
            # Update last_updated timestamp
            self.last_updated = datetime.now()
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching GDELT data: {str(e)}")
            raise
            
    def filter_macro_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter GDELT events to only include macro-relevant news.
        
        Args:
            df: DataFrame containing GDELT events
            
        Returns:
            Filtered DataFrame with only macro-relevant events
        """
        # Define macro-relevant keywords and themes
        macro_keywords = [
            "economy", "economic", "inflation", "deflation",
            "interest rate", "monetary policy", "fiscal policy",
            "GDP", "unemployment", "employment", "central bank",
            "Federal Reserve", "ECB", "BOJ", "BOE",
            "currency", "exchange rate", "forex", "FX",
            "bond", "treasury", "yield", "debt"
        ]
        
        # Filter based on themes and keywords
        mask = df['themes'].str.contains('|'.join(macro_keywords), case=False, na=False)
        return df[mask]
        
    def process_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process GDELT events data for sentiment analysis.
        
        Args:
            df: DataFrame containing GDELT events
            
        Returns:
            Processed DataFrame ready for sentiment analysis
        """
        # Select relevant columns
        columns = [
            'date', 'source', 'url', 'title', 'description',
            'themes', 'tone', 'num_mentions', 'num_sources'
        ]
        
        # Ensure all required columns exist
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                df[col] = None
                
        # Process dates
        df['date'] = pd.to_datetime(df['date'])
        
        # Clean text fields
        text_columns = ['title', 'description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
                
        return df[columns] 