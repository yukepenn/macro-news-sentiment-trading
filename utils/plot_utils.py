"""
Utilities for creating visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")

def plot_equity_curve(returns: pd.Series,
                     title: str = "Strategy Equity Curve",
                     figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot equity curve from returns.
    
    Args:
        returns: Series of returns
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate cumulative returns
    equity_curve = (1 + returns).cumprod()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, label="Strategy")
    
    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Customize plot
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_drawdown(returns: pd.Series,
                 title: str = "Strategy Drawdown",
                 figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot drawdown from returns.
    
    Args:
        returns: Series of returns
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot drawdown
    ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, color='red', label="Drawdown")
    
    # Customize plot
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_rolling_metrics(returns: pd.Series,
                        window: int = 252,
                        title: str = "Rolling Performance Metrics",
                        figsize: tuple = (12, 8)) -> plt.Figure:
    """
    Plot rolling performance metrics.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate rolling metrics
    rolling_metrics = pd.DataFrame(index=returns.index)
    rolling_metrics["rolling_return"] = returns.rolling(window).mean()
    rolling_metrics["rolling_vol"] = returns.rolling(window).std()
    rolling_metrics["rolling_sharpe"] = (
        rolling_metrics["rolling_return"] / rolling_metrics["rolling_vol"]
    )
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot rolling return
    ax1.plot(rolling_metrics.index, rolling_metrics["rolling_return"], label="Rolling Return")
    ax1.set_title("Rolling Return")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot rolling volatility
    ax2.plot(rolling_metrics.index, rolling_metrics["rolling_vol"], label="Rolling Volatility")
    ax2.set_title("Rolling Volatility")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot rolling Sharpe
    ax3.plot(rolling_metrics.index, rolling_metrics["rolling_sharpe"], label="Rolling Sharpe")
    ax3.set_title("Rolling Sharpe Ratio")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Customize plot
    fig.suptitle(title)
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_sentiment_index(sentiment_df: pd.DataFrame,
                        title: str = "News Sentiment Index",
                        figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot sentiment index over time.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot sentiment index
    ax.plot(sentiment_df["date"], sentiment_df["sentiment_mean"], label="Sentiment Index")
    
    # Add confidence interval
    ax.fill_between(
        sentiment_df["date"],
        sentiment_df["sentiment_mean"] - sentiment_df["sentiment_std"],
        sentiment_df["sentiment_mean"] + sentiment_df["sentiment_std"],
        alpha=0.2
    )
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Customize plot
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_correlation_matrix(returns_df: pd.DataFrame,
                          title: str = "Correlation Matrix",
                          figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Args:
        returns_df: DataFrame of returns
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        square=True,
        ax=ax
    )
    
    # Customize plot
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_monthly_returns(returns: pd.Series,
                        title: str = "Monthly Returns",
                        figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot monthly returns heatmap.
    
    Args:
        returns: Series of returns
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Reshape for heatmap
    monthly_returns = monthly_returns.to_frame()
    monthly_returns["year"] = monthly_returns.index.year
    monthly_returns["month"] = monthly_returns.index.month
    monthly_returns = monthly_returns.pivot(
        index="year",
        columns="month",
        values=returns.name
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        monthly_returns,
        annot=True,
        cmap="RdYlGn",
        center=0,
        fmt=".1%",
        ax=ax
    )
    
    # Customize plot
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance(importance_df: pd.DataFrame,
                          title: str = "Feature Importance",
                          figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    importance_df.sort_values("importance").plot(
        kind="barh",
        x="feature",
        y="importance",
        ax=ax
    )
    
    # Customize plot
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def save_figure(fig: plt.Figure,
               filename: str,
               dpi: int = 300) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: DPI for saved figure
    """
    try:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {filename}")
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        raise 