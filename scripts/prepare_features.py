"""
Script for preparing features from sentiment and market data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.finance_utils import calculate_returns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_features(sentiment_df: pd.DataFrame,
                    market_df: pd.DataFrame,
                    asset: str) -> pd.DataFrame:
    """
    Prepare features for modeling.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        market_df: DataFrame with market data
        asset: Asset ticker to prepare features for
        
    Returns:
        DataFrame with features and target
    """
    # Filter market data for the asset
    asset_data = market_df[market_df["ticker"] == asset].copy()
    
    # Calculate returns
    asset_data["return"] = calculate_returns(asset_data["Close"])
    
    # Create target (next day's return)
    asset_data["target"] = asset_data["return"].shift(-1)
    
    # Merge with sentiment data
    merged_df = pd.merge_asof(
        asset_data.sort_values("Date"),
        sentiment_df.sort_values("date"),
        left_on="Date",
        right_on="date",
        direction="backward"
    )
    
    # Create features
    features_df = pd.DataFrame(index=merged_df.index)
    
    # Sentiment features
    features_df["sentiment"] = merged_df["sentiment_mean"]
    features_df["sentiment_std"] = merged_df["sentiment_std"]
    features_df["sentiment_change"] = merged_df["sentiment_mean"].diff()
    features_df["sentiment_ma5"] = merged_df["sentiment_mean"].rolling(5).mean()
    features_df["sentiment_ma10"] = merged_df["sentiment_mean"].rolling(10).mean()
    
    # Market features
    features_df["return"] = merged_df["return"]
    features_df["return_ma5"] = merged_df["return"].rolling(5).mean()
    features_df["return_ma10"] = merged_df["return"].rolling(10).mean()
    features_df["volatility"] = merged_df["return"].rolling(20).std()
    
    # Volume features
    features_df["volume"] = merged_df["Volume"]
    features_df["volume_ma5"] = merged_df["Volume"].rolling(5).mean()
    features_df["volume_ma10"] = merged_df["Volume"].rolling(10).mean()
    
    # Price features
    features_df["high_low_ratio"] = merged_df["High"] / merged_df["Low"]
    features_df["close_open_ratio"] = merged_df["Close"] / merged_df["Open"]
    
    # Add target
    features_df["target"] = merged_df["target"]
    
    # Drop rows with NaN values
    features_df = features_df.dropna()
    
    return features_df

def main():
    """Main function to prepare features."""
    # Create output directory if it doesn't exist
    processed_dir = Path(project_root) / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load sentiment data
        sentiment_file = processed_dir / "news_sentiment_daily.csv"
        if not sentiment_file.exists():
            raise FileNotFoundError(f"Sentiment file not found: {sentiment_file}")
            
        logger.info(f"Loading sentiment data from {sentiment_file}")
        sentiment_df = pd.read_csv(sentiment_file)
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        
        # Load market data
        market_file = Path(project_root) / "data" / "raw" / "market_data.csv"
        if not market_file.exists():
            raise FileNotFoundError(f"Market data file not found: {market_file}")
            
        logger.info(f"Loading market data from {market_file}")
        market_df = pd.read_csv(market_file)
        market_df["Date"] = pd.to_datetime(market_df["Date"])
        
        # Prepare features for each asset
        assets = market_df["ticker"].unique()
        for asset in assets:
            logger.info(f"Preparing features for {asset}")
            features_df = prepare_features(sentiment_df, market_df, asset)
            
            # Save features
            output_file = processed_dir / f"{asset.lower()}_features.csv"
            features_df.to_csv(output_file)
            logger.info(f"Saved features to {output_file}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 