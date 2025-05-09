"""
Script for fetching and processing GDELT data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.gdelt_utils import GDELTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to fetch and process GDELT data."""
    # Create output directory if it doesn't exist
    output_dir = Path(project_root) / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GDELT client
    client = GDELTClient()
    
    # Set date range (e.g., last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    try:
        # Fetch events
        logger.info(f"Fetching GDELT events from {start_date.date()} to {end_date.date()}")
        events_df = client.fetch_events(start_date, end_date)
        
        # Filter macro events
        logger.info("Filtering macro-relevant events")
        macro_events_df = client.filter_macro_events(events_df)
        
        # Process events
        logger.info("Processing events")
        processed_df = client.process_events(macro_events_df)
        
        # Save to CSV
        output_file = output_dir / "gdelt_macro_events.csv"
        processed_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(processed_df)} events to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 