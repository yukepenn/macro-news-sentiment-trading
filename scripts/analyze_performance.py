#!/usr/bin/env python
"""
scripts/analyze_performance.py

Produce confusion matrix, classification report, and list of top errors.
Generate SHAP waterfall plots and case study visualizations.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_model_and_data(model_path: Path, feature_path: Path):
    """Load model, scaler, features and prepare predictions."""
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    # Handle both pipeline and legacy model formats
    if "pipeline" in data:
        pipeline = data["pipeline"]
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["clf"]
        feature_names = getattr(model, "feature_names_in_", None)
    else:
        model = data["model"]
        scaler = data["scaler"]
        feature_names = data["feature_names"]

    logger.info(f"Loading features from {feature_path}")
    df = pd.read_csv(feature_path, parse_dates=["date"])
    
    # Fallback: if feature_names is None, use all columns except 'date' and 'return_t+1'
    if feature_names is None:
        feature_names = [col for col in df.columns if col not in ["date", "return_t+1"]]

    # Prepare feature matrix and predictions
    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    y_true = (df["return_t+1"] > 0).astype(int).values
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    return df, model, scaler, feature_names, y_true, preds, probs

def error_analysis(df: pd.DataFrame, y_true: np.ndarray, preds: np.ndarray, 
                  probs: np.ndarray, top_n: int = 10):
    """Compute confusion matrix, classification report, and identify top errors."""
    df = df.copy()
    df["true"] = y_true
    df["pred"] = preds
    df["prob_up"] = probs
    df["ret"] = df["return_t+1"]
    
    # Identify error types
    df["error_type"] = np.where(
        (df["true"] == 1) & (df["pred"] == 0), "FN",
        np.where((df["true"] == 0) & (df["pred"] == 1), "FP", "OK")
    )
    
    # Compute metrics
    cm = confusion_matrix(y_true, preds)
    cr = classification_report(y_true, preds, output_dict=True)
    
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"Classification report:\n{classification_report(y_true, preds)}")
    
    # Get top errors
    fps = df[df["error_type"] == "FP"].nlargest(top_n, "prob_up")
    fns = df[df["error_type"] == "FN"].nsmallest(top_n, "prob_up")
    
    # Save error cases
    fps.to_csv("backtest/top_false_positives.csv", index=False)
    fns.to_csv("backtest/top_false_negatives.csv", index=False)
    logger.info(f"Wrote top {top_n} false positives/negatives to backtest/")
    
    return df, cm, cr

def generate_case_study_plot(df: pd.DataFrame, date: pd.Timestamp, 
                            window: int = 5, save_path: Path = None):
    """Generate case study plot showing sentiment and returns around a date."""
    # Fallback for missing sentiment column
    if "sentiment" not in df.columns:
        logger.warning("No 'sentiment' column found in features. Skipping plot.")
        return
    # Get window of dates
    date_idx = df[df["date"] == date].index[0]
    start_idx = max(0, date_idx - window)
    end_idx = min(len(df), date_idx + window + 1)
    
    # Prepare data
    plot_df = df.iloc[start_idx:end_idx].copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot sentiment
    ax1.plot(plot_df["date"], plot_df["sentiment"], "b-", label="Sentiment")
    ax1.axvline(date, color="r", linestyle="--", label="Event Date")
    ax1.set_ylabel("Sentiment")
    ax1.legend()
    
    # Plot returns
    ax2.plot(plot_df["date"], plot_df["return_t+1"], "g-", label="Returns")
    ax2.axvline(date, color="r", linestyle="--", label="Event Date")
    ax2.set_ylabel("Returns")
    ax2.legend()
    
    plt.title(f"Case Study: {date.strftime('%Y-%m-%d')}")
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved case study plot to {save_path}")
    
    plt.close()

def generate_shap_waterfall(df: pd.DataFrame, model, scaler, feature_names: list,
                          date: pd.Timestamp, save_path: Path = None):
    """Generate SHAP waterfall plot for a specific date."""
    # Fallback: if feature_names is None, use all columns except 'date' and 'return_t+1'
    if feature_names is None:
        feature_names = [col for col in df.columns if col not in ["date", "return_t+1"]]
    # Find row for date
    row = df[df["date"] == date]
    if row.empty:
        logger.warning(f"No data found for date {date}")
        return
    # Prepare data
    X = row[feature_names].values
    X_scaled = scaler.transform(X)
    # Select SHAP explainer based on model type
    model_class = type(model).__name__
    try:
        if "XGB" in model_class or "Booster" in model_class:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
    except Exception as e:
        logger.error(f"SHAP explainer failed: {e}")
        return
    # Limit to top 10 features by absolute SHAP value
    abs_shap = np.abs(shap_values[0])
    top_idx = np.argsort(abs_shap)[-10:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_shap = shap_values[0][top_idx]
    top_x = X_scaled[0][top_idx] if "XGB" not in model_class else X[0][top_idx]
    try:
        plt.figure(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value if hasattr(explainer, 'expected_value') else explainer.expected_values,
            top_shap,
            top_x,
            feature_names=top_features,
            show=False
        )
        plt.title(f"SHAP Waterfall Plot - {date.strftime('%Y-%m-%d')}")
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SHAP waterfall plot to {save_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Legacy SHAP waterfall plot failed: {e}")
        return

def main():
    # Paths
    model_paths = {
        "logistic": Path("models/sentiment_model_tuned_logistic.pkl"),
        "xgb": Path("models/sentiment_model_tuned_xgb.pkl")
    }
    feature_path = Path("data/processed/features/model_features_eurusd.csv")
    
    for model_name, model_path in model_paths.items():
        logger.info(f"\nAnalyzing {model_name} model...")
        
        # Load data and model
        df, model, scaler, feature_names, y_true, preds, probs = load_model_and_data(
            model_path, feature_path
        )
        
        # Error analysis
        df, cm, cr = error_analysis(df, y_true, preds, probs)
        
        # Generate case studies for top errors
        fps = df[df["error_type"] == "FP"].nlargest(3, "prob_up")
        fns = df[df["error_type"] == "FN"].nsmallest(3, "prob_up")
        
        for date in fps["date"]:
            generate_case_study_plot(
                df, date,
                save_path=Path(f"visuals/case_study_fp_{model_name}_{date.strftime('%Y%m%d')}.png")
            )
            generate_shap_waterfall(
                df, model, scaler, feature_names, date,
                save_path=Path(f"visuals/waterfall_fp_{model_name}_{date.strftime('%Y%m%d')}.png")
            )
        
        for date in fns["date"]:
            generate_case_study_plot(
                df, date,
                save_path=Path(f"visuals/case_study_fn_{model_name}_{date.strftime('%Y%m%d')}.png")
            )
            generate_shap_waterfall(
                df, model, scaler, feature_names, date,
                save_path=Path(f"visuals/waterfall_fn_{model_name}_{date.strftime('%Y%m%d')}.png")
            )
        
        logger.info(f"Analysis complete for {model_name} model! Check backtest/ and visuals/ directories for outputs.")

if __name__ == "__main__":
    main() 