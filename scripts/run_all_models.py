#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automate training and backtesting for all assets and models.
"""

import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define assets and models
ASSETS = ["eurusd", "usdjpy","zn"]
MODELS = ["logistic", "xgb"]

def run_command(cmd: str) -> None:
    """Run a shell command and log its output."""
    logger.info(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise

def train_and_backtest(asset: str, model_type: str) -> None:
    """Train model and run backtest for a given asset and model type."""
    # Create timestamp for unique file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define paths
    model_path = f"models/{asset}_{model_type}_model.pkl"
    shap_path = f"plots/shap_{asset}_{model_type}.png"
    returns_path = f"backtest/{asset}_{model_type}_daily_returns.csv"
    equity_path = f"visuals/{asset}_{model_type}_equity_curve.png"
    metrics_path = f"backtest/{asset}_{model_type}_performance_metrics.txt"
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("backtest").mkdir(exist_ok=True)
    Path("visuals").mkdir(exist_ok=True)
    
    # Train model
    train_cmd = (
        f"python -m scripts.train_model "
        f"--features data/processed/features/model_features_{asset}.csv "
        f"--model {model_path} "
        f"--shap {shap_path} "
        f"--algo {model_type}"
    )
    run_command(train_cmd)
    
    # Run backtest
    backtest_cmd = (
        f"python -m scripts.run_backtest "
        f"--model {model_path} "
        f"--features data/processed/features/model_features_{asset}.csv "
        f"--returns-out {returns_path} "
        f"--equity-plot {equity_path} "
        f"--metrics-out {metrics_path}"
    )
    run_command(backtest_cmd)

def main():
    parser = argparse.ArgumentParser(description="Train and backtest models for all assets")
    parser.add_argument("--assets", nargs="+", default=ASSETS,
                      help="List of assets to process (default: all)")
    parser.add_argument("--models", nargs="+", default=MODELS,
                      help="List of models to train (default: all)")
    args = parser.parse_args()
    
    try:
        # Process each asset and model combination
        for asset in args.assets:
            for model_type in args.models:
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing {asset.upper()} with {model_type.upper()} model")
                logger.info(f"{'='*50}\n")
                
                train_and_backtest(asset, model_type)
                
        logger.info("\nAll models trained and backtested successfully!")
        
    except Exception as e:
        logger.error(f"Error in run_all_models.py: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 