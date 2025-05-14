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
    
    # Goldstein momentum (exponential weighted momentum)
    for decay in (0.1, 0.3, 0.5):
        df[f"goldstein_momentum_{decay}"] = df["sentiment_mean"].ewm(alpha=decay).mean()
    
    # Article count spikes (z-score based)
    df["article_zscore"] = (df["article_count"] - df["article_count"].rolling(20).mean()) / df["article_count"].rolling(20).std()
    df["article_spike"] = (df["article_zscore"] > 2).astype(int)  # Spike threshold at 2 standard deviations
    
    # Smoothed sentiment (using Kalman filter-like approach)
    df["sentiment_smoothed"] = df["sentiment_mean"].ewm(span=5, adjust=False).mean()
    df["sentiment_smoothed_std"] = df["sentiment_mean"].ewm(span=5, adjust=False).std()
    
    # Rolling volatility
    for w in (5,10,20):
        df[f"sentiment_std_{w}d"] = df["sentiment_mean"].rolling(w).std()
        df[f"sentiment_vol_{w}d"] = df["sentiment_diff"].rolling(w).std()
    
    # Next-day return and market vol
    df["return_t+1"] = df["close"].pct_change().shift(-1)
    df["volatility_20d"] = df["return_t+1"].rolling(20).std()
    
    # Interaction features
    df["sentiment_vol_interaction"] = df["sentiment_std_5d"] * df["volatility_20d"]
    df["article_impact"] = df["sentiment_diff"] * df["log_articles"]
    df["spike_sentiment"] = df["article_spike"] * df["sentiment_mean"]  # Interaction between spikes and sentiment
    
    # Drop incompletes
    df = df.dropna().reset_index(drop=True)
    logger.info(f"After dropna: {len(df)} rows")
    return df

def save_features(df, asset, out_dir):
    path = out_dir / f"model_features_{asset}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved {asset} features to {path}")

def main():
    sentiment_path = Path("data/processed/sentiment/news_sentiment_daily.csv")
    market_path    = Path("data/raw/market/market_tidy.csv")
    out_dir        = Path("data/processed/features")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load common data
    sent = load_sentiment(sentiment_path)
    mkt  = pd.read_csv(market_path, parse_dates=["date"])

    # Loop each asset
    for asset in mkt["asset"].unique():
        logger.info(f"Preparing features for {asset}")
        df_asset = mkt[mkt["asset"] == asset][["date", "close"]].sort_values("date")
        merged = pd.merge(sent, df_asset, on="date", how="inner")
        feats  = engineer_features(merged)
        save_features(feats, asset, out_dir)

if __name__ == "__main__":
    main()
