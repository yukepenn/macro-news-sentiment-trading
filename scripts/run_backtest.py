#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run backtest simulation for trained model.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path: Path) -> Tuple[Pipeline, list]:
    """Load trained model and metadata."""
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle both pipeline and legacy formats
    if isinstance(model_data, dict):
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        # Wrap in pipeline if not already
        if not isinstance(model, Pipeline):
            pipeline = Pipeline([
                ('scaler', scaler if scaler is not None else StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = model
    else:
        pipeline = model
        feature_names = None
    
    return pipeline, feature_names

def load_features(path: Path) -> pd.DataFrame:
    """Load and validate feature data."""
    logger.info(f"Loading features from {path}")
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found at {path}")
    
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info(f"Loaded {len(df):,} rows")
    return df

def simulate_backtest(
    pipeline: Pipeline,
    feature_names: list,
    df: pd.DataFrame,
    cost_lookup: Dict[str, float],
    threshold: float = 0.5,
    spread_lookup: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Simulate backtest with vectorized transaction costs and optional slippage.
    
    Args:
        pipeline: Trained model pipeline (scaler + model)
        feature_names: List of feature names
        df: DataFrame with features and dates
        cost_lookup: Dictionary mapping tickers to round-trip costs
        threshold: Probability threshold for long/short decisions
        spread_lookup: Optional dictionary mapping tickers to typical spreads
    """
    # Prepare features and get predictions
    X = df[feature_names].values
    prob = pipeline.predict_proba(X)[:, 1]
    positions = np.where(prob > threshold, 1, -1)
    
    # Get returns
    y_ret = df["return_t+1"].values
    
    # Strategy P&L = position * market return
    strat_ret = positions * y_ret
    
    # ------------------- TRANSACTION COSTS -------------------
    # Build cost array (requires `df['ticker']`)
    if "ticker" not in df.columns:
        raise ValueError("`ticker` column is required for transaction cost lookup")
    cost_array = df["ticker"].str.lower().map(cost_lookup)
    if cost_array.isna().any():
        missing = cost_array[cost_array.isna()].unique()
        raise KeyError(f"No cost defined for tickers: {missing}")
    cost_array = cost_array.values
    
    # Detect trades (position changes)
    pos_shift = np.empty_like(positions)
    pos_shift[0] = positions[0]
    pos_shift[1:] = positions[:-1]
    trades = (positions != pos_shift).astype(int)
    trades[0] = 0  # never charge on the very first bar
    
    # Calculate costs
    costs = trades * cost_array
    
    # ------------------- OPTIONAL SLIPPAGE -------------------
    if spread_lookup is not None:
        # Build spread array
        spread_array = df["ticker"].str.lower().map(spread_lookup)
        if spread_array.isna().any():
            missing = spread_array[spread_array.isna()].unique()
            raise KeyError(f"No spread defined for tickers: {missing}")
        spread_array = spread_array.values
        
        # Calculate slippage based on position changes
        slippage = spread_array * trades
        costs += slippage
    # ----------------------------------------------------------
    
    # Subtract costs from returns
    strat_ret = strat_ret - costs
    
    # Create backtest DataFrame
    backtest_df = pd.DataFrame({
        "date": df["date"],
        "market_ret": y_ret,
        "position": positions,
        "strategy_ret": strat_ret,
        "cost": costs,  # Track costs separately
        "trades": trades,  # Track trade days
        "prob": prob  # Track probabilities
    }).set_index("date")
    
    return backtest_df

def compute_metrics(bt: pd.DataFrame) -> dict:
    """Compute performance metrics."""
    # Annualized metrics
    days = (bt.index[-1] - bt.index[0]).days
    years = days / 365.25
    
    # Returns
    total_return = (1 + bt["strategy_ret"]).prod() - 1
    cagr = (1 + total_return) ** (1/years) - 1
    
    # Risk metrics
    vol = bt["strategy_ret"].std() * np.sqrt(252)
    sharpe = np.sqrt(252) * bt["strategy_ret"].mean() / bt["strategy_ret"].std()
    
    # Drawdown
    cum_ret = (1 + bt["strategy_ret"]).cumprod()
    rolling_max = cum_ret.expanding().max()
    drawdown = (cum_ret / rolling_max - 1)
    max_dd = drawdown.min()
    
    # Trade metrics
    n_trades = (bt["position"].diff().abs() > 0).sum()
    win_rate = (bt["strategy_ret"] > 0).mean()
    
    # Cost metrics
    total_cost = bt["cost"].sum()
    cost_per_trade = total_cost / n_trades if n_trades > 0 else 0
    
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Volatility": vol,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Total Return": total_return,
        "Number of Trades": n_trades,
        "Total Cost": total_cost,
        "Cost per Trade": cost_per_trade
    }

def save_outputs(backtest_df: pd.DataFrame, metrics: dict, 
                returns_out: Path, equity_plot: Path, metrics_out: Path) -> None:
    """Save backtest results and plots."""
    # Save daily returns
    backtest_df.to_csv(returns_out)
    logger.info(f"Saved daily returns to {returns_out}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Plot equity curve
    cum_ret = (1 + backtest_df["strategy_ret"]).cumprod()
    ax1.plot(cum_ret.index, cum_ret.values, label="Strategy")
    ax1.plot(cum_ret.index, (1 + backtest_df["market_ret"]).cumprod(), 
             label="Market", alpha=0.5)
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative costs
    cum_cost = backtest_df["cost"].cumsum()
    ax2.plot(cum_cost.index, cum_cost.values, label="Cumulative Cost", color='red')
    ax2.set_title("Cumulative Transaction Costs")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cost")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(equity_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved equity curve and costs to {equity_plot}")
    
    # Save metrics
    with open(metrics_out, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    logger.info(f"Saved metrics to {metrics_out}")

def main():
    parser = argparse.ArgumentParser(description="Run backtest simulation")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--features", type=str, required=True, help="Path to feature CSV file")
    parser.add_argument("--asset", type=str, required=True, help="Asset name (e.g. eurusd, usdjpy, zn)")
    parser.add_argument("--returns-out", type=str, required=True, help="Path to save daily returns")
    parser.add_argument("--equity-plot", type=str, required=True, help="Path to save equity curve plot")
    parser.add_argument("--metrics-out", type=str, required=True, help="Path to save performance metrics")
    parser.add_argument("--cost-fx", type=float, default=0.0002, help="Round-trip cost for FX pairs (default: 0.0002)")
    parser.add_argument("--cost-fut", type=float, default=0.0005, help="Round-trip cost for futures (default: 0.0005)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for long/short decisions (default: 0.5)")
    parser.add_argument("--spread-fx", type=float, default=None, help="Typical spread for FX pairs (optional)")
    parser.add_argument("--spread-fut", type=float, default=None, help="Typical spread for futures (optional)")
    args = parser.parse_args()

    try:
        # Load model and features
        model_path = Path(args.model)
        pipeline, feature_names = load_model(model_path)
        features_path = Path(args.features)
        df = load_features(features_path)

        # inject a ticker column if missing
        if "ticker" not in df.columns:
            df["ticker"] = args.asset.lower()

        # Define cost lookup from CLI args
        cost_lookup = {
            "eurusd": args.cost_fx,
            "usdjpy": args.cost_fx,
            "zn": args.cost_fut
        }

        # Define spread lookup if provided
        spread_lookup = None
        if args.spread_fx is not None or args.spread_fut is not None:
            spread_lookup = {
                "eurusd": args.spread_fx or 0.0,
                "usdjpy": args.spread_fx or 0.0,
                "zn": args.spread_fut or 0.0
            }

        # Run backtest
        logger.info("Running backtest simulation...")
        bt = simulate_backtest(
            pipeline,
            feature_names,
            df,
            cost_lookup,
            threshold=args.threshold,
            spread_lookup=spread_lookup
        )

        # Compute metrics
        metrics = compute_metrics(bt)
        logger.info("\nPerformance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Save outputs
        save_outputs(
            bt,
            metrics,
            Path(args.returns_out),
            Path(args.equity_plot),
            Path(args.metrics_out)
        )

        logger.info("Backtest complete!")

    except Exception as e:
        logger.error(f"Error in run_backtest.py: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
