#!/usr/bin/env python
"""
scripts/run_backtest.py

Simulate trading strategy using the trained sentiment model.
Produces daily P&L, equity curve, and summary metrics.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_model(model_path: Path):
    """Load the pickled model, scaler, and metadata."""
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["feature_names"]

def load_features(features_path: Path):
    """Load the feature DataFrame (must include date & return_t+1)."""
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def simulate_backtest(model, scaler, feature_names, df: pd.DataFrame):
    """Run through each date, predict, and compute strategy returns."""
    # Extract feature matrix and true next-day returns
    X = df[feature_names].values
    X_df = pd.DataFrame(X, columns=feature_names)  # Create DataFrame with feature names
    X_scaled = scaler.transform(X_df)  # Now scaler knows feature names
    y_ret = df["return_t+1"].values  # next-day market return

    # Predicted probability of up-move; position = +1 if >0.5 else -1
    probs = model.predict_proba(X_scaled)[:, 1]
    positions = np.where(probs > 0.5, 1, -1)

    # Strategy P&L = position * market return
    strat_ret = positions * y_ret

    backtest_df = pd.DataFrame({
        "date":       df["date"],
        "market_ret": y_ret,
        "position":   positions,
        "strategy_ret": strat_ret
    }).set_index("date")

    # Cumulative returns (equity)
    backtest_df["equity"] = (1 + backtest_df["strategy_ret"]).cumprod()

    return backtest_df

def compute_metrics(bt: pd.DataFrame):
    """Compute annualized Sharpe, CAGR, max drawdown, win rate."""
    # Daily strategy returns
    rets = bt["strategy_ret"]
    # Remove any NaNs
    rets = rets.dropna()

    # Annualization factor (assume 252 trading days)
    ann_fac = 252

    # CAGR: (final equity)^(252/N) - 1
    N = len(rets)
    cagr = bt["equity"].iloc[-1] ** (ann_fac / N) - 1

    # Sharpe: mean(daily) / std(daily) * sqrt(252)
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(ann_fac)

    # Max drawdown
    running_max = bt["equity"].cummax()
    drawdown = bt["equity"] / running_max - 1
    max_dd = drawdown.min()

    # Win rate
    win_rate = (rets > 0).mean()

    return {
        "CAGR":         cagr,
        "Sharpe":       sharpe,
        "Max Drawdown": max_dd,
        "Win Rate":     win_rate
    }

def save_outputs(backtest_df: pd.DataFrame, metrics: dict,
                 returns_out: Path, equity_plot: Path, metrics_out: Path):
    """Write daily returns CSV, equity curve chart, and metrics text."""
    # 1) Daily returns
    returns_out.parent.mkdir(parents=True, exist_ok=True)
    backtest_df[["market_ret", "position", "strategy_ret"]].to_csv(returns_out)
    logger.info(f"Wrote daily returns to {returns_out}")

    # 2) Equity curve plot
    equity_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    backtest_df["equity"].plot(title="Strategy Equity Curve")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(equity_plot, dpi=300)
    plt.close()
    logger.info(f"Saved equity curve to {equity_plot}")

    # 3) Metrics summary
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    logger.info(f"Wrote performance metrics to {metrics_out}")

def main():
    p = argparse.ArgumentParser("Backtest sentiment-driven strategy")
    p.add_argument("--model",      required=True, help="Path to pickle model")
    p.add_argument("--features",   required=True, help="Path to model_features CSV")
    p.add_argument("--returns-out",required=True, help="CSV for daily returns")
    p.add_argument("--equity-plot", required=True, help="Path to equity curve PNG")
    p.add_argument("--metrics-out",required=True, help="Text file for metrics")
    args = p.parse_args()

    # Paths
    model_path    = Path(args.model)
    features_path = Path(args.features)
    returns_out   = Path(args.returns_out)
    equity_plot   = Path(args.equity_plot)
    metrics_out   = Path(args.metrics_out)

    # Load model and data
    model, scaler, feature_names = load_model(model_path)
    df = load_features(features_path)

    # Run backtest
    bt = simulate_backtest(model, scaler, feature_names, df)

    # Compute metrics
    metrics = compute_metrics(bt)

    # Save outputs
    save_outputs(bt, metrics, returns_out, equity_plot, metrics_out)

    logger.info("Backtest complete!")

if __name__ == "__main__":
    main()
