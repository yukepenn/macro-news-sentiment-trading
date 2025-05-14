"""
scripts/prepare_features.py

Prepare modeling features by merging sentiment data with EURUSD prices.
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sentiment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    required = {"date","sentiment_mean","article_count","sentiment_ma5","sentiment_ma20"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sentiment data: {missing}")
    logger.info(f"Loaded {len(df)} sentiment rows")
    return df

def load_market(path: Path, asset: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["asset"] == asset][["date","close"]].sort_values("date")
    logger.info(f"Loaded {len(df)} rows for {asset}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Lagged sentiment
    for lag in (1,2,3):
        df[f"sentiment_lag{lag}"] = df["sentiment_mean"].shift(lag)
    # Article log
    df["log_articles"] = np.log1p(df["article_count"])
    # Momentum & acceleration
    df["sentiment_diff"] = df["sentiment_mean"].diff()
    df["sentiment_accel"] = df["sentiment_diff"].diff()
    # Rolling volatility
    for w in (5,10,20):
        df[f"sentiment_std_{w}d"] = df["sentiment_mean"].rolling(w).std()
        df[f"sentiment_vol_{w}d"] = df["sentiment_diff"].rolling(w).std()
    # Next-day return and market vol
    df["return_t+1"] = df["close"].pct_change().shift(-1)
    df["volatility_20d"] = df["return_t+1"].rolling(20).std()
    # Interaction
    df["sentiment_vol_interaction"] = df["sentiment_std_5d"] * df["volatility_20d"]
    df["article_impact"] = df["sentiment_diff"] * df["log_articles"]
    # Drop incompletes
    df = df.dropna().reset_index(drop=True)
    logger.info(f"After dropna: {len(df)} rows")
    return df

def save_features(df: pd.DataFrame, asset: str):
    out_dir = Path("data/processed/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"model_features_{asset}.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Saved features to {out_file}")
    # Log a quick summary
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Mean return: {df['return_t+1'].mean():.4f}, Vol return: {df['return_t+1'].std():.4f}")

def main():
    try:
        sentiment_path = Path("data/processed/sentiment/news_sentiment_daily.csv")
        market_path    = Path("data/raw/market/market_tidy.csv")
        asset = "eurusd"

        # Load
        sent = load_sentiment(sentiment_path)
        mkt  = load_market(market_path, asset)

        # Merge + features
        df = pd.merge(sent, mkt, on="date", how="inner")
        logger.info(f"Merged data: {len(df)} rows")
        feats = engineer_features(df)

        # Save
        save_features(feats, asset)
        logger.info("Feature preparation completed successfully!")

    except Exception as e:
        logger.error(f"Error in prepare_features.py: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
