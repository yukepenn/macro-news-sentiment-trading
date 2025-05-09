"""
Script for performing sentiment analysis on GDELT data.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

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

def main():
    """Main function to perform sentiment analysis."""
    # Create output directories if they don't exist
    processed_dir = Path(project_root) / "data" / "processed"
    visuals_dir = Path(project_root) / "visuals"
    processed_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load GDELT data
        input_file = Path(project_root) / "data" / "raw" / "gdelt_macro_events.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        logger.info(f"Loading GDELT data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Initialize sentiment analyzer
        logger.info("Initializing sentiment analyzer")
        analyzer = SentimentAnalyzer()
        
        # Analyze sentiment
        logger.info("Analyzing sentiment")
        df = analyzer.analyze_dataframe(df, "title")  # Use title for sentiment analysis
        
        # Create daily sentiment index
        logger.info("Creating daily sentiment index")
        daily_sentiment = analyzer.create_daily_sentiment_index(df)
        
        # Save processed data
        output_file = processed_dir / "news_sentiment_daily.csv"
        daily_sentiment.to_csv(output_file, index=False)
        logger.info(f"Saved daily sentiment index to {output_file}")
        
        # Create and save visualization
        logger.info("Creating sentiment index visualization")
        fig = plot_sentiment_index(daily_sentiment)
        viz_file = visuals_dir / "sentiment_index.png"
        save_figure(fig, viz_file)
        logger.info(f"Saved visualization to {viz_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 