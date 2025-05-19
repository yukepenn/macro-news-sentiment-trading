"""
Script for performing sentiment analysis on GDELT data.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.nlp_utils import SentimentAnalyzer
from utils.plot_utils import plot_sentiment_index, save_figure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load GDELT macro events data."""
    logger.info("Loading GDELT macro events...")
    df = pd.read_csv(
        "data/raw/gdelt/gdelt_macro_events_top100_with_headlines.csv",
        parse_dates=["date"]
    )
    logger.info(f"Loaded {len(df):,} events")
    return df

def init_finbert():
    """Initialize FinBERT model and tokenizer."""
    logger.info("Initializing FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model

def score_batch(texts, tokenizer, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Score a batch of texts using FinBERT."""
    if not texts:  # Handle empty batch
        return []
    
    # Tokenize and prepare inputs
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    
    # Calculate sentiment score (positive - negative)
    scores = probs[:, 0] - probs[:, 2]  # positive - negative
    return scores.cpu().numpy()

def process_sentiment(df, tokenizer, model, batch_size=32):
    """Process sentiment for all events in batches."""
    logger.info("Processing sentiment scores...")
    
    # Use the scraped headline for real NLP
    df['text'] = df['headline'].astype(str)
    
    # Process in batches with progress bar
    scores = []
    for i in tqdm(range(0, len(df), batch_size)):
        chunk = df['text'].iloc[i:i+batch_size].tolist()
        batch_scores = score_batch(chunk, tokenizer, model)
        scores.extend(batch_scores)
    
    df['finbert_score'] = scores
    return df

def aggregate_daily(df):
    """Aggregate sentiment scores to daily frequency."""
    logger.info("Aggregating to daily frequency...")
    
    daily = df.groupby('date').agg(
        sentiment_mean=('finbert_score', 'mean'),
        sentiment_std=('finbert_score', 'std'),
        article_count=('finbert_score', 'size'),
        goldstein_mean=('goldstein_scale', 'mean'),
        goldstein_std=('goldstein_scale', 'std')
    ).reset_index()
    
    # Add rolling averages
    daily['sentiment_ma5'] = daily['sentiment_mean'].rolling(5).mean()
    daily['sentiment_ma20'] = daily['sentiment_mean'].rolling(20).mean()
    
    return daily

def save_results(daily_df):
    """Save processed results."""
    # Create directory if it doesn't exist
    os.makedirs("data/processed/sentiment", exist_ok=True)
    
    # Save daily sentiment
    output_path = "data/processed/sentiment/news_sentiment_daily.csv"
    daily_df.to_csv(output_path, index=False)
    logger.info(f"Saved daily sentiment to {output_path}")
    
    # Save summary statistics
    summary = {
        'total_days': len(daily_df),
        'avg_articles_per_day': daily_df['article_count'].mean(),
        'avg_sentiment': daily_df['sentiment_mean'].mean(),
        'sentiment_std': daily_df['sentiment_mean'].std(),
        'date_range': f"{daily_df['date'].min()} to {daily_df['date'].max()}"
    }
    
    logger.info("\nSummary Statistics:")
    for key, value in summary.items():
        logger.info(f"{key}: {value}")

def main():
    try:
        # 1. Load data
        df = load_data()
        
        # 2. Initialize FinBERT
        tokenizer, model = init_finbert()
        
        # 3. Process sentiment
        df = process_sentiment(df, tokenizer, model)
        
        # 4. Aggregate to daily frequency
        daily_df = aggregate_daily(df)
        
        # 5. Save results
        save_results(daily_df)
        
        logger.info("Sentiment analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 