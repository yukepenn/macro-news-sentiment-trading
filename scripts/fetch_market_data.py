"""
Script for fetching market data using yfinance.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np

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
    "ZN": "ZN=F",          # 10 Year Treasury Bond Futures (front month)
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fetch market data')
    parser.add_argument('--start-date', 
                       type=str,
                       default='2015-01-01',
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date',
                       type=str,
                       default='2025-04-30',
                       help='End date in YYYY-MM-DD format')
    return parser.parse_args()

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
        DataFrame with asset data in tidy format
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
        
        # Ensure all required columns are present
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
        for col in required_columns:
            if col not in data.columns:
                data[col] = np.nan
        
        # Select and order columns
        data = data[required_columns]
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {str(e)}")
        return pd.DataFrame()

def save_market_data(data: pd.DataFrame, output_dir: Path):
    """
    Save market data in both combined and individual formats.
    
    Args:
        data: DataFrame containing market data
        output_dir: Directory to save the data
    """
    try:
        # Save combined data
        output_file = output_dir / "market_data.csv"
        data.to_csv(output_file, index=False)
        logger.info(f"Saved market data to {output_file}")
        
        # Save individual files
        for ticker in data['ticker'].unique():
            asset_name = next((name for name, t in ASSETS.items() if t == ticker), ticker)
            asset_data = data[data["ticker"] == ticker]
            if not asset_data.empty:
                asset_file = output_dir / f"{asset_name.lower()}_data.csv"
                asset_data.to_csv(asset_file, index=False)
                logger.info(f"Saved {asset_name} data to {asset_file}")
                
    except Exception as e:
        logger.error(f"Error saving market data: {str(e)}")
        raise

def convert_wide_to_tidy(input_path: Path, output_dir: Path):
    """
    Convert wide market_data.csv to tidy per-ticker files and a long-format file.
    """
    import pandas as pd
    import os
    
    tickers = ["eurusd", "usdjpy", "zn"]
    metrics = ["open", "high", "low", "close", "volume"]
    
    # 1) Read the file, skip the second header row
    with open(input_path, "r") as f:
        lines = f.readlines()
    # Remove the second header row (row 2)
    with open("temp_market_data.csv", "w") as f:
        f.write(lines[0])
        f.writelines(lines[2:])
    # 2) Load as DataFrame
    wide = pd.read_csv("temp_market_data.csv", parse_dates=["Date"])
    os.remove("temp_market_data.csv")
    print("DEBUG: wide.columns before renaming:", list(wide.columns))
    # 3) Rename Date → date
    wide = wide.rename(columns={"Date": "date"})
    # Remove any extra columns (like 'ticker')
    expected_cols = 1 + len(tickers) * len(metrics)
    if len(wide.columns) > expected_cols:
        # Drop extra columns from the end
        wide = wide.iloc[:, :expected_cols]
    elif len(wide.columns) < expected_cols:
        raise ValueError(f"Not enough columns in wide file: expected {expected_cols}, got {len(wide.columns)}")
    # 4) Build MultiIndex columns
    groups = []
    for tk in tickers:
        for m in metrics:
            groups.append((tk, m))
    wide.columns = ["date"] + groups
    print("DEBUG: wide.columns after renaming:", list(wide.columns))
    # 5) Melt to long format
    long = wide.set_index("date").stack(level=0).reset_index()
    long = long.rename(columns={"level_1": "ticker"})
    # 6) Save per-ticker files
    for tk in tickers:
        cols = ["date"] + [(tk, m) for m in metrics]
        sub = wide[cols].copy()
        sub.columns = ["date"] + metrics  # flatten for CSV
        sub.to_csv(output_dir / f"{tk}.csv", index=False)
    # 7) Optionally, save a single long file
    long.to_csv(output_dir / "market_long.csv", index=False)
    print("Conversion complete. Tidy files saved in data/raw/market/")

    # 8) Transform long file into a truly tidy format
    df = pd.read_csv(output_dir / "market_long.csv", parse_dates=["date"])
    print("DEBUG: Intermediate long file head:\n", df.head())
    df[["asset", "metric"]] = df["ticker"].str.extract(r"\('([^']+)',\s*'([^']+)'\)")
    # Check for duplicates
    if df.duplicated(subset=["date", "asset", "metric"]).any():
        print("WARNING: Duplicates found in long file. Dropping duplicates.")
        df = df.drop_duplicates(subset=["date", "asset", "metric"])
    tidy = df.pivot(index=["date", "asset"], columns="metric", values="0").reset_index()
    tidy.columns.name = None
    tidy = tidy.rename(columns=str.lower).sort_values(["asset", "date"])
    tidy.to_csv(output_dir / "market_tidy.csv", index=False)
    print("Truly tidy long file saved as market_tidy.csv")

def main():
    """Main function to fetch market data."""
    # Parse command line arguments
    args = parse_args()
    
    # Convert string dates to datetime objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Create output directory if it doesn't exist
    output_dir = Path(project_root) / "data" / "raw" / "market"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            # Sort by date and ticker
            combined_data = combined_data.sort_values(['Date', 'ticker'])
            
            # Save the data
            save_market_data(combined_data, output_dir)
            # Convert wide file to tidy per-ticker files and long format
            convert_wide_to_tidy(output_dir / "market_data.csv", output_dir)
        else:
            logger.error("No data fetched for any asset")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 