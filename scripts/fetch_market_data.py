"""
Script for fetching market data using yfinance.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define assets to fetch
ASSETS = {
    "EURUSD": "EURUSD=X",  # EUR/USD exchange rate
    "USDJPY": "USDJPY=X",  # USD/JPY exchange rate
    "TNX": "^TNX",         # 10-Year Treasury Yield
    "TLT": "TLT",          # 20+ Year Treasury Bond ETF
}

def fetch_asset_data(ticker: str,
                    start_date: datetime,
                    end_date: datetime) -> pd.DataFrame:
    """
    Fetch data for a single asset.
    
    Args:
        ticker: Asset ticker
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with asset data
    """
    try:
        # Fetch data
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Reset index to make date a column
        data = data.reset_index()
        
        # Add ticker column
        data["ticker"] = ticker
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {str(e)}")
        return pd.DataFrame()

def main():
    """Main function to fetch market data."""
    # Create output directory if it doesn't exist
    output_dir = Path(project_root) / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set date range (e.g., last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    try:
        # Fetch data for each asset
        all_data = []
        for asset_name, ticker in ASSETS.items():
            logger.info(f"Fetching data for {asset_name} ({ticker})")
            data = fetch_asset_data(ticker, start_date, end_date)
            
            if not data.empty:
                all_data.append(data)
                logger.info(f"Fetched {len(data)} rows for {asset_name}")
            else:
                logger.warning(f"No data fetched for {asset_name}")
        
        if all_data:
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            output_file = output_dir / "market_data.csv"
            combined_data.to_csv(output_file, index=False)
            logger.info(f"Saved market data to {output_file}")
            
            # Also save individual files
            for asset_name, ticker in ASSETS.items():
                asset_data = combined_data[combined_data["ticker"] == ticker]
                if not asset_data.empty:
                    asset_file = output_dir / f"{asset_name.lower()}_data.csv"
                    asset_data.to_csv(asset_file, index=False)
                    logger.info(f"Saved {asset_name} data to {asset_file}")
        else:
            logger.error("No data fetched for any asset")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 