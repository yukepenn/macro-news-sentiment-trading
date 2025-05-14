"""
Utilities for handling GDELT data.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import os
import gzip
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        self.base_url = "https://api.gdeltproject.org/api/v2"
        self.api_key = os.getenv('GDELT_API_KEY')
        if not self.api_key:
            raise ValueError("GDELT_API_KEY environment variable is not set")
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
            
        try:
            # Construct URL for GDELT API
            url = f"{self.base_url}/doc/document"
            
            # Set up query parameters
            params = {
                "format": "json",
                "startdatetime": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "enddatetime": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode": "artlist",
                "maxrecords": 250,  # Maximum records per request
                "sort": "hybridrel",
                "apikey": self.api_key
            }
            
            if query:
                params["query"] = query
            else:
                # Default query for macro events
                params["query"] = "(economy OR economic OR inflation OR deflation OR interest rate OR monetary policy OR fiscal policy OR GDP OR unemployment OR employment OR central bank OR Federal Reserve OR ECB OR BOJ OR BOE OR currency OR exchange rate OR forex OR FX OR bond OR treasury OR yield OR debt)"
            
            all_data = []
            
            page = 1
            
            while True:
                params["page"] = page
                
                try:
                    # Make request
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    
                    if not data.get('articles'):
                        break
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data['articles'])
                    all_data.append(df)
                    
                    logger.info(f"Successfully fetched page {page} with {len(df)} articles")
                    
                    # Check if we've reached the end
                    if len(data['articles']) < params['maxrecords']:
                        break
                    
                    page += 1
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error fetching page {page}: {str(e)}")
                    break
            
            if not all_data:
                raise Exception("No data was fetched for the specified date range")
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Update last_updated timestamp
            self.last_updated = datetime.now()
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error in fetch_events: {str(e)}")
            raise
            
    def filter_macro_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter GDELT events to only include macro-relevant news.
        
        Args:
            df: DataFrame containing GDELT events
            
        Returns:
            Filtered DataFrame with only macro-relevant events
        """
        # Define macro-relevant keywords
        macro_keywords = [
            "economy", "economic", "inflation", "deflation",
            "interest rate", "monetary policy", "fiscal policy",
            "GDP", "unemployment", "employment", "central bank",
            "Federal Reserve", "ECB", "BOJ", "BOE",
            "currency", "exchange rate", "forex", "FX",
            "bond", "treasury", "yield", "debt"
        ]
        
        # Create a pattern for matching
        pattern = '|'.join(macro_keywords)
        
        # Filter based on title and description
        mask = (
            df['title'].str.contains(pattern, case=False, na=False) |
            df['description'].str.contains(pattern, case=False, na=False)
        )
        
        return df[mask]
        
    def process_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process GDELT events data for sentiment analysis.
        
        Args:
            df: DataFrame containing GDELT events
            
        Returns:
            Processed DataFrame ready for sentiment analysis
        """
        # Select and rename relevant columns
        columns = {
            'seendate': 'date',
            'domain': 'source',
            'url': 'url',
            'title': 'title',
            'description': 'description',
            'language': 'language',
            'tone': 'tone'
        }
        
        # Process the DataFrame
        processed_df = df[columns.keys()].rename(columns=columns)
        
        # Convert date to datetime
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        # Clean text fields
        text_columns = ['title', 'description', 'source']
        for col in text_columns:
            processed_df[col] = processed_df[col].fillna('').astype(str)
        
        return processed_df 