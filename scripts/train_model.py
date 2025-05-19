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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

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
    args = parser.parse_args()
    
    try:
        # Load data
        features_path = Path(args.features)
        df = load_features(features_path)
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(df)
        
        # Train model (logistic or xgb)
        model = train_model(X_train, y_train, algo=args.algo)
        
        # Evaluate
        evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Generate SHAP plot
        shap_path = Path(args.shap)
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        generate_shap_plot(model, X_test, feature_names, shap_path, args.algo)
        
        # Save model
        model_path = Path(args.model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, scaler, feature_names, model_path)
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Error in train_model.py: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 