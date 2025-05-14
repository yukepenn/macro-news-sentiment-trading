"""
Utilities for financial calculations and analysis.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_market_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load market data from CSV file.
    
    Args:
        file_path: Path to market data CSV file
        
    Returns:
        DataFrame with market data in tidy format
    """
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        raise

def pivot_market_data(data: pd.DataFrame, 
                     value_col: str = 'Close',
                     index_col: str = 'Date',
                     columns_col: str = 'ticker') -> pd.DataFrame:
    """
    Pivot market data from tidy format to wide format.
    
    Args:
        data: DataFrame with market data in tidy format
        value_col: Column to use as values
        index_col: Column to use as index
        columns_col: Column to use as columns
        
    Returns:
        Pivoted DataFrame
    """
    return data.pivot(index=index_col, columns=columns_col, values=value_col)

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of returns
    """
    return prices.pct_change().dropna()

def calculate_returns_by_ticker(data: pd.DataFrame,
                              price_col: str = 'Close',
                              date_col: str = 'Date',
                              ticker_col: str = 'ticker') -> pd.DataFrame:
    """
    Calculate returns for each ticker in the market data.
    
    Args:
        data: DataFrame with market data in tidy format
        price_col: Column containing price data
        date_col: Column containing dates
        ticker_col: Column containing ticker symbols
        
    Returns:
        DataFrame with returns for each ticker
    """
    # Pivot data to wide format
    prices = pivot_market_data(data, value_col=price_col)
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Convert back to tidy format
    returns = returns.reset_index().melt(
        id_vars=[date_col],
        var_name=ticker_col,
        value_name='return'
    )
    
    return returns

def calculate_sharpe_ratio(returns: pd.Series, 
                         risk_free_rate: float = 0.0,
                         periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate/periods_per_year
    if len(excess_returns) < 2:
        return 0.0
        
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown as a negative number
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    return drawdown.min()

def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate from returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Win rate as a decimal
    """
    return (returns > 0).mean()

def calculate_annualized_return(returns: pd.Series,
                              periods_per_year: int = 252) -> float:
    """
    Calculate annualized return from returns.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized return
    """
    return (1 + returns.mean()) ** periods_per_year - 1

def calculate_volatility(returns: pd.Series,
                        periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility from returns.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)

def calculate_calmar_ratio(returns: pd.Series,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio
    """
    annualized_return = calculate_annualized_return(returns, periods_per_year)
    max_drawdown = calculate_max_drawdown(returns)
    
    if max_drawdown == 0:
        return 0.0
        
    return annualized_return / abs(max_drawdown)

def calculate_sortino_ratio(returns: pd.Series,
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio from returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate/periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) < 2:
        return 0.0
        
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
        
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

def calculate_performance_metrics(returns: pd.Series,
                                risk_free_rate: float = 0.0,
                                periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {
        "total_return": (1 + returns).prod() - 1,
        "annualized_return": calculate_annualized_return(returns, periods_per_year),
        "volatility": calculate_volatility(returns, periods_per_year),
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": calculate_max_drawdown(returns),
        "calmar_ratio": calculate_calmar_ratio(returns, periods_per_year),
        "win_rate": calculate_win_rate(returns)
    }
    
    return metrics

def calculate_rolling_metrics(returns: pd.Series,
                            window: int = 252,
                            risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with rolling metrics
    """
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    rolling_metrics["rolling_return"] = returns.rolling(window).mean()
    
    # Rolling volatility
    rolling_metrics["rolling_vol"] = returns.rolling(window).std()
    
    # Rolling Sharpe
    rolling_metrics["rolling_sharpe"] = (
        (rolling_metrics["rolling_return"] - risk_free_rate/252) /
        rolling_metrics["rolling_vol"]
    )
    
    # Rolling drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.rolling(window).max()
    rolling_metrics["rolling_drawdown"] = (cumulative_returns / rolling_max) - 1
    
    return rolling_metrics

def calculate_correlation_matrix(returns_df: pd.DataFrame,
                               date_col: str = 'Date',
                               ticker_col: str = 'ticker',
                               value_col: str = 'return') -> pd.DataFrame:
    """
    Calculate correlation matrix from returns DataFrame in tidy format.
    
    Args:
        returns_df: DataFrame of returns in tidy format
        date_col: Column containing dates
        ticker_col: Column containing ticker symbols
        value_col: Column containing return values
        
    Returns:
        Correlation matrix
    """
    # Pivot data to wide format
    returns_wide = pivot_market_data(returns_df, value_col=value_col)
    return returns_wide.corr()

def calculate_beta(returns: pd.Series,
                  market_returns: pd.Series) -> float:
    """
    Calculate beta relative to market returns.
    
    Args:
        returns: Series of strategy returns
        market_returns: Series of market returns
        
    Returns:
        Beta coefficient
    """
    covariance = returns.cov(market_returns)
    market_variance = market_returns.var()
    
    if market_variance == 0:
        return 0.0
        
    return covariance / market_variance

def calculate_betas(returns_df: pd.DataFrame,
                   market_ticker: str,
                   date_col: str = 'Date',
                   ticker_col: str = 'ticker',
                   value_col: str = 'return') -> pd.Series:
    """
    Calculate betas for all assets relative to a market ticker.
    
    Args:
        returns_df: DataFrame of returns in tidy format
        market_ticker: Ticker to use as market
        date_col: Column containing dates
        ticker_col: Column containing ticker symbols
        value_col: Column containing return values
        
    Returns:
        Series of beta values for each ticker
    """
    # Pivot data to wide format
    returns_wide = pivot_market_data(returns_df, value_col=value_col)
    
    # Get market returns
    market_returns = returns_wide[market_ticker]
    
    # Calculate betas
    betas = {}
    for ticker in returns_wide.columns:
        if ticker != market_ticker:
            betas[ticker] = calculate_beta(returns_wide[ticker], market_returns)
    
    return pd.Series(betas)

def calculate_rolling_metrics_by_ticker(data: pd.DataFrame,
                                      window: int = 252,
                                      risk_free_rate: float = 0.0,
                                      price_col: str = 'Close',
                                      date_col: str = 'Date',
                                      ticker_col: str = 'ticker') -> pd.DataFrame:
    """
    Calculate rolling metrics for each ticker in the market data.
    
    Args:
        data: DataFrame with market data in tidy format
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        price_col: Column containing price data
        date_col: Column containing dates
        ticker_col: Column containing ticker symbols
        
    Returns:
        DataFrame with rolling metrics for each ticker
    """
    # Calculate returns
    returns = calculate_returns_by_ticker(data, price_col, date_col, ticker_col)
    
    # Pivot returns to wide format
    returns_wide = pivot_market_data(returns, value_col='return')
    
    # Calculate rolling metrics
    rolling_metrics = pd.DataFrame(index=returns_wide.index)
    
    for ticker in returns_wide.columns:
        ticker_returns = returns_wide[ticker]
        
        # Rolling returns
        rolling_metrics[f"{ticker}_rolling_return"] = ticker_returns.rolling(window).mean()
        
        # Rolling volatility
        rolling_metrics[f"{ticker}_rolling_vol"] = ticker_returns.rolling(window).std()
        
        # Rolling Sharpe
        rolling_metrics[f"{ticker}_rolling_sharpe"] = (
            (rolling_metrics[f"{ticker}_rolling_return"] - risk_free_rate/252) /
            rolling_metrics[f"{ticker}_rolling_vol"]
        )
        
        # Rolling drawdown
        cumulative_returns = (1 + ticker_returns).cumprod()
        rolling_max = cumulative_returns.rolling(window).max()
        rolling_metrics[f"{ticker}_rolling_drawdown"] = (cumulative_returns / rolling_max) - 1
    
    return rolling_metrics 