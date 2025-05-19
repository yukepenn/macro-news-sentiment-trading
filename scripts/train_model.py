#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train sentiment-based trading model and generate SHAP analysis.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import shap
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from typing import List, Dict, Optional, Tuple
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_features(path: Path) -> pd.DataFrame:
    """Load and validate feature data."""
    logger.info(f"Loading features from {path}")
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found at {path}")
    
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info(f"Loaded {len(df):,} rows")
    return df

def prepare_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Split data into train/test sets and prepare features/target."""
    # Sort by date
    df = df.sort_values("date")
    
    # Split by date
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Train period: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ["date", "return_t+1"]]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = (train_df["return_t+1"] > 0).astype(int)
    y_test = (test_df["return_t+1"] > 0).astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

def train_model(X_train: np.ndarray, y_train: np.ndarray, algo: str = "logistic") -> object:
    """Train model based on specified algorithm."""
    logger.info(f"Training model: {algo}")
    if algo == "logistic":
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:  # xgb
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    model.fit(X_train, y_train)
    return model

def optimize_threshold(prob: np.ndarray, returns: np.ndarray, 
                      thresholds: np.ndarray = np.linspace(0.3, 0.7, 41)) -> float:
    """
    Find the probability threshold that maximizes Sharpe ratio.
    
    Args:
        prob: Model probabilities
        returns: Actual returns
        thresholds: Array of thresholds to try
        
    Returns:
        Optimal threshold that maximizes Sharpe ratio
    """
    best_sharpe = -np.inf
    best_threshold = 0.5
    
    for threshold in thresholds:
        positions = np.where(prob > threshold, 1, -1)
        # Ensure positions and returns have the same length
        positions = positions[:len(returns)]
        strategy_returns = positions * returns
        
        if len(strategy_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold
    
    return best_threshold

def bootstrap_confidence_intervals(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 20,  # 1-month blocks for daily data
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for performance metrics.
    
    Args:
        returns: Array of strategy returns
        n_bootstrap: Number of bootstrap samples
        block_size: Size of blocks for block bootstrap
        confidence: Confidence level for intervals
        
    Returns:
        Dictionary with confidence intervals for each metric
    """
    n_returns = len(returns)
    n_blocks = n_returns // block_size
    
    # Initialize arrays for bootstrap samples
    sharpe_samples = np.zeros(n_bootstrap)
    cagr_samples = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Generate block bootstrap sample
        block_indices = np.random.randint(0, n_blocks, size=n_blocks)
        sample_indices = np.concatenate([
            np.arange(b * block_size, (b + 1) * block_size)
            for b in block_indices
        ])
        sample_returns = returns[sample_indices]
        
        # Calculate metrics
        sharpe_samples[i] = np.sqrt(252) * np.mean(sample_returns) / np.std(sample_returns)
        cagr_samples[i] = (1 + sample_returns).prod() ** (252/len(sample_returns)) - 1
    
    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    ci_sharpe = np.percentile(sharpe_samples, [alpha * 100, (1 - alpha) * 100])
    ci_cagr = np.percentile(cagr_samples, [alpha * 100, (1 - alpha) * 100])
    
    return {
        "sharpe_ci": tuple(ci_sharpe),
        "cagr_ci": tuple(ci_cagr)
    }

def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "return_t+1",
    algo: str = "logistic",
    n_splits: int = 5,
    max_rolling_window: int = 20,  # Maximum rolling window size used in features
    optimize_thresholds: bool = True,  # Whether to optimize probability thresholds
    cost_lookup: Optional[Dict[str, float]] = None,  # Transaction costs by ticker
    spread_lookup: Optional[Dict[str, float]] = None,  # Spreads by ticker
    n_bootstrap: int = 1000  # Number of bootstrap samples for confidence intervals
) -> pd.DataFrame:
    """
    Perform an expanding-window walk-forward evaluation with transaction costs.
    Returns a DataFrame of fold-level metrics.
    
    Note: Features at time t predict returns at t+1, so we need to align:
    - X[t] -> y[t+1] for training
    - X[t] -> y[t+1] for prediction
    - positions[t] -> returns[t+1] for strategy performance
    
    Also skips initial days in each fold to account for rolling window features.
    """
    X = df[feature_cols].values
    y = (df[target_col] > 0).astype(int).values
    dates = df["date"].values
    returns = df[target_col].values  # These are the t+1 returns

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Skip initial days in train set to account for rolling windows
        train_idx = train_idx[train_idx >= max_rolling_window]
        if len(train_idx) == 0:
            logger.warning(f"Fold {fold+1}: No valid training data after skipping rolling window period")
            continue
            
        # 1) split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        date_train, date_test = dates[train_idx], dates[test_idx]
        
        # Get the actual returns for strategy performance
        # Note: test_idx gives us the indices for features at time t
        # We need the returns at t+1 for these predictions
        strategy_returns_idx = test_idx + 1  # Shift forward by 1 to get t+1 returns
        strategy_returns_idx = strategy_returns_idx[strategy_returns_idx < len(returns)]  # Handle last day
        
        if len(strategy_returns_idx) == 0:
            logger.warning(f"Fold {fold+1}: No returns available for strategy evaluation")
            continue

        # 2) scale
        scaler = StandardScaler().fit(X_train)
        X_tr_s = scaler.transform(X_train)
        X_te_s = scaler.transform(X_test)

        # 3) train
        model = train_model(X_tr_s, y_train, algo=algo)

        # 4) get train & test probabilities
        prob_train = model.predict_proba(X_tr_s)[:, 1]
        prob_test = model.predict_proba(X_te_s)[:, 1]

        # 5) optimize threshold on TRAIN only
        if optimize_thresholds:
            # Align returns_train with next-day returns
            returns_train_idx = train_idx + 1
            returns_train_idx = returns_train_idx[returns_train_idx < len(returns)]
            returns_train = returns[returns_train_idx]
            threshold = optimize_threshold(prob_train, returns_train)
            logger.info(f"Fold {fold+1}: Optimal threshold = {threshold:.3f} (train)")
        else:
            threshold = 0.5
            
        # 6) apply threshold to TEST
        pred_test = (prob_test > threshold).astype(int)
        auc = roc_auc_score(y_test, prob_test)
        acc = accuracy_score(y_test, pred_test)

        # 7) compute strategy returns on TEST
        positions = np.where(prob_test > threshold, 1, -1)
        positions = positions[:len(strategy_returns_idx)]
        strategy_returns = positions * returns[strategy_returns_idx]
        
        # Calculate costs if provided
        total_cost = 0
        if cost_lookup is not None and "ticker" in df.columns:
            # Get tickers for this fold
            tickers = df.iloc[test_idx]["ticker"].values
            tickers = tickers[:len(strategy_returns_idx)]  # Trim to match available returns
            
            # Calculate transaction costs
            cost_array = np.array([cost_lookup[t.lower()] for t in tickers])
            trades = np.abs(np.diff(positions, prepend=positions[0]))
            costs = trades * cost_array
            total_cost = costs.sum()
            
            # Add slippage if provided
            if spread_lookup is not None:
                spread_array = np.array([spread_lookup[t.lower()] for t in tickers])
                slippage = trades * spread_array
                costs += slippage
                total_cost += slippage.sum()
            
            # Subtract costs from returns
            strategy_returns = strategy_returns - costs
        
        # Calculate metrics for this fold
        if len(strategy_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
            cagr = (1 + strategy_returns).prod() ** (252/len(strategy_returns)) - 1
            
            # Calculate bootstrap confidence intervals
            ci = bootstrap_confidence_intervals(
                strategy_returns,
                n_bootstrap=n_bootstrap,
                block_size=20  # 1-month blocks
            )
        else:
            sharpe = np.nan
            cagr = np.nan
            ci = {"sharpe_ci": (np.nan, np.nan), "cagr_ci": (np.nan, np.nan)}

        fold_results.append({
            "fold": fold+1,
            "train_start": date_train.min(),
            "train_end":   date_train.max(),
            "test_start":  date_test.min(),
            "test_end":    date_test.max(),
            "auc": auc,
            "accuracy": acc,
            "sharpe": sharpe,
            "cagr": cagr,
            "sharpe_ci_lower": ci["sharpe_ci"][0],
            "sharpe_ci_upper": ci["sharpe_ci"][1],
            "cagr_ci_lower": ci["cagr_ci"][0],
            "cagr_ci_upper": ci["cagr_ci"][1],
            "n_trades": len(strategy_returns),
            "total_cost": total_cost,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "threshold": threshold
        })

    return pd.DataFrame(fold_results)

def evaluate_model(model, X_train: np.ndarray, X_test: np.ndarray, 
                  y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate model performance."""
    # Training metrics
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_prob)
    
    # Test metrics
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)
    
    logger.info("Model Performance:")
    logger.info(f"Train Accuracy: {train_acc:.3f}, AUC: {train_auc:.3f}")
    logger.info(f"Test Accuracy: {test_acc:.3f}, AUC: {test_auc:.3f}")

def generate_shap_plot(model, X_test: np.ndarray, feature_names: list, 
                      output_path: Path, algo: str) -> None:
    """Generate and save SHAP summary plot."""
    logger.info("Generating SHAP analysis...")
    
    # Select appropriate explainer based on algorithm
    if algo == "logistic":
        explainer = shap.LinearExplainer(model, X_test)
    else:  # xgb
        explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_test)
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                     show=False, plot_size=(10, 6))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP plot saved to {output_path}")

def save_model(model, scaler, feature_names: list, output_path: Path) -> None:
    """Save model and metadata."""
    logger.info(f"Saving model to {output_path}")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'timestamp': datetime.now()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

def main():
    parser = argparse.ArgumentParser(description="Train sentiment-based trading model")
    parser.add_argument("--features", type=str, required=True,
                      help="Path to feature CSV file")
    parser.add_argument("--model", type=str, required=True,
                      help="Path to save trained model")
    parser.add_argument("--shap", type=str, required=True,
                      help="Path to save SHAP summary plot")
    parser.add_argument("--algo", choices=["logistic", "xgb"], default="logistic",
                      help="Which model to train: logistic regression or XGBoost")
    parser.add_argument("--n-splits", type=int, default=5,
                      help="Number of splits for walk-forward evaluation")
    parser.add_argument("--max-rolling-window", type=int, default=20,
                      help="Maximum rolling window size used in features")
    parser.add_argument("--no-threshold-opt", action="store_true",
                      help="Disable threshold optimization")
    parser.add_argument("--cost-fx", type=float, default=0.0002,
                      help="Round-trip cost for FX pairs (default: 0.0002)")
    parser.add_argument("--cost-fut", type=float, default=0.0005,
                      help="Round-trip cost for futures (default: 0.0005)")
    parser.add_argument("--spread-fx", type=float, default=None,
                      help="Typical spread for FX pairs (optional)")
    parser.add_argument("--spread-fut", type=float, default=None,
                      help="Typical spread for futures (optional)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                      help="Number of bootstrap samples for confidence intervals")
    args = parser.parse_args()
    
    try:
        # Load data
        features_path = Path(args.features)
        df = load_features(features_path)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in ["date", "return_t+1"]]
        
        # Define cost and spread lookups
        cost_lookup = {
            "eurusd": args.cost_fx,
            "usdjpy": args.cost_fx,
            "zn": args.cost_fut
        }
        
        spread_lookup = None
        if args.spread_fx is not None or args.spread_fut is not None:
            spread_lookup = {
                "eurusd": args.spread_fx or 0.0,
                "usdjpy": args.spread_fx or 0.0,
                "zn": args.spread_fut or 0.0
            }
        
        # Run walk-forward evaluation
        logger.info("Running walk-forward evaluation...")
        results = walk_forward_evaluate(
            df, 
            feature_cols, 
            algo=args.algo, 
            n_splits=args.n_splits,
            max_rolling_window=args.max_rolling_window,
            optimize_thresholds=not args.no_threshold_opt,
            cost_lookup=cost_lookup,
            spread_lookup=spread_lookup,
            n_bootstrap=args.n_bootstrap
        )
        
        # Print detailed fold-level results
        logger.info("\nFold-level metrics:")
        for _, row in results.iterrows():
            logger.info(f"\nFold {row['fold']}:")
            logger.info(f"Period: {row['test_start']} to {row['test_end']}")
            logger.info(f"AUC: {row['auc']:.3f}")
            logger.info(f"Accuracy: {row['accuracy']:.3f}")
            logger.info(f"Sharpe: {row['sharpe']:.3f} (95% CI: [{row['sharpe_ci_lower']:.3f}, {row['sharpe_ci_upper']:.3f}])")
            logger.info(f"CAGR: {row['cagr']:.3f} (95% CI: [{row['cagr_ci_lower']:.3f}, {row['cagr_ci_upper']:.3f}])")
            logger.info(f"Total Cost: {row['total_cost']:.4f}")
            logger.info(f"Number of Trades: {row['n_trades']}")
            logger.info(f"Threshold: {row['threshold']:.3f}")
        
        logger.info("\nAggregate performance:")
        logger.info(f"Mean AUC: {results.auc.mean():.3f} ± {results.auc.std():.3f}")
        logger.info(f"Mean Accuracy: {results.accuracy.mean():.3f} ± {results.accuracy.std():.3f}")
        logger.info(f"Mean Sharpe: {results.sharpe.mean():.3f} ± {results.sharpe.std():.3f}")
        logger.info(f"Mean CAGR: {results.cagr.mean():.3f} ± {results.cagr.std():.3f}")
        logger.info(f"Mean Threshold: {results.threshold.mean():.3f} ± {results.threshold.std():.3f}")
        logger.info(f"Total Cost: {results.total_cost.sum():.4f}")
        
        # Prepare data for final model
        X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(df)
        
        # Train final model
        model = train_model(X_train, y_train, algo=args.algo)
        
        # Evaluate final model
        evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Generate SHAP plot
        shap_path = Path(args.shap)
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        generate_shap_plot(model, X_test, feature_names, shap_path, args.algo)
        
        # Save model
        model_path = Path(args.model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, scaler, feature_names, model_path)
        
    except Exception as e:
        logger.error(f"Error in train_model.py: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 