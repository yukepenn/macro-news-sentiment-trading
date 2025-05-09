# API Documentation

## GDELT Utilities

### GDELTClient

A client for interacting with GDELT data.

#### Methods

- `fetch_events(start_date, end_date=None, query=None)`: Fetch GDELT events data for a given date range.
- `filter_macro_events(df)`: Filter GDELT events to only include macro-relevant news.
- `process_events(df)`: Process GDELT events data for sentiment analysis.

## NLP Utilities

### SentimentAnalyzer

A class for performing sentiment analysis using FinBERT.

#### Methods

- `analyze_text(text)`: Analyze sentiment of a single text.
- `analyze_batch(texts, batch_size=32)`: Analyze sentiment of multiple texts in batches.
- `get_sentiment_score(sentiment_dict)`: Convert sentiment dictionary to a single score.
- `analyze_dataframe(df, text_column)`: Analyze sentiment of texts in a DataFrame.
- `create_daily_sentiment_index(df)`: Create daily sentiment index from sentiment scores.

## Finance Utilities

### Financial Calculations

- `calculate_returns(prices)`: Calculate returns from price series.
- `calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)`: Calculate Sharpe ratio.
- `calculate_max_drawdown(returns)`: Calculate maximum drawdown.
- `calculate_win_rate(returns)`: Calculate win rate.
- `calculate_annualized_return(returns, periods_per_year=252)`: Calculate annualized return.
- `calculate_volatility(returns, periods_per_year=252)`: Calculate annualized volatility.
- `calculate_calmar_ratio(returns, periods_per_year=252)`: Calculate Calmar ratio.
- `calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)`: Calculate Sortino ratio.
- `calculate_performance_metrics(returns, risk_free_rate=0.0, periods_per_year=252)`: Calculate comprehensive performance metrics.
- `calculate_rolling_metrics(returns, window=252, risk_free_rate=0.0)`: Calculate rolling performance metrics.
- `calculate_correlation_matrix(returns_df)`: Calculate correlation matrix.
- `calculate_beta(returns, market_returns)`: Calculate beta relative to market returns.

## Plotting Utilities

### Visualization Functions

- `plot_equity_curve(returns, title="Strategy Equity Curve", figsize=(12, 6))`: Plot equity curve.
- `plot_drawdown(returns, title="Strategy Drawdown", figsize=(12, 6))`: Plot drawdown.
- `plot_rolling_metrics(returns, window=252, title="Rolling Performance Metrics", figsize=(12, 8))`: Plot rolling performance metrics.
- `plot_sentiment_index(sentiment_df, title="News Sentiment Index", figsize=(12, 6))`: Plot sentiment index.
- `plot_correlation_matrix(returns_df, title="Correlation Matrix", figsize=(10, 8))`: Plot correlation matrix.
- `plot_monthly_returns(returns, title="Monthly Returns", figsize=(12, 6))`: Plot monthly returns.
- `plot_feature_importance(importance_df, title="Feature Importance", figsize=(10, 6))`: Plot feature importance.
- `save_figure(fig, filename, dpi=300)`: Save figure to file. 