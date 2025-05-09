"""
Utilities for natural language processing and sentiment analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Class for performing sentiment analysis using FinBERT."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer with FinBERT.
        
        Args:
            model_name: Name of the FinBERT model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info("Successfully loaded FinBERT model")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {str(e)}")
            raise
            
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        # Prepare input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
        # Convert to dictionary
        sentiment_scores = {
            "positive": probs[0][0].item(),
            "neutral": probs[0][1].item(),
            "negative": probs[0][2].item()
        }
        
        return sentiment_scores
        
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            batch_size: Size of batches for processing
            
        Returns:
            List of dictionaries with sentiment scores
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i + batch_size]
            
            # Prepare batch input
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                
            # Convert to list of dictionaries
            batch_results = [
                {
                    "positive": probs[j][0].item(),
                    "neutral": probs[j][1].item(),
                    "negative": probs[j][2].item()
                }
                for j in range(len(batch_texts))
            ]
            
            results.extend(batch_results)
            
        return results
        
    def get_sentiment_score(self, sentiment_dict: Dict[str, float]) -> float:
        """
        Convert sentiment dictionary to a single score.
        
        Args:
            sentiment_dict: Dictionary with sentiment scores
            
        Returns:
            Single sentiment score between -1 and 1
        """
        return sentiment_dict["positive"] - sentiment_dict["negative"]
        
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyze sentiment of texts in a DataFrame.
        
        Args:
            df: DataFrame containing texts
            text_column: Name of column containing texts to analyze
            
        Returns:
            DataFrame with added sentiment columns
        """
        # Get texts
        texts = df[text_column].tolist()
        
        # Analyze sentiment
        sentiment_results = self.analyze_batch(texts)
        
        # Add sentiment scores to DataFrame
        df["sentiment_positive"] = [r["positive"] for r in sentiment_results]
        df["sentiment_neutral"] = [r["neutral"] for r in sentiment_results]
        df["sentiment_negative"] = [r["negative"] for r in sentiment_results]
        df["sentiment_score"] = [self.get_sentiment_score(r) for r in sentiment_results]
        
        return df
        
    def create_daily_sentiment_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily sentiment index from sentiment scores.
        
        Args:
            df: DataFrame with sentiment scores and dates
            
        Returns:
            DataFrame with daily sentiment index
        """
        # Group by date and calculate mean sentiment
        daily_sentiment = df.groupby(df["date"].dt.date).agg({
            "sentiment_score": ["mean", "std", "count"],
            "sentiment_positive": "mean",
            "sentiment_negative": "mean",
            "sentiment_neutral": "mean"
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            "date",
            "sentiment_mean",
            "sentiment_std",
            "article_count",
            "positive_mean",
            "negative_mean",
            "neutral_mean"
        ]
        
        return daily_sentiment 