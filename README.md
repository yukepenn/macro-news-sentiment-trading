# News & Macro Sentiment Alpha

An end-to-end trading system that leverages unstructured news data and NLP to generate trading signals for macro assets. This project demonstrates the integration of alternative data (news sentiment) into a quantitative trading framework.

## Project Overview

The system processes global news data from GDELT, applies FinBERT for sentiment analysis, and generates trading signals for macro assets (FX rates and Treasury yields). Key components include:

- News data ingestion from GDELT
- Sentiment analysis using FinBERT
- Feature engineering and signal generation
- Machine learning model training
- Strategy backtesting and evaluation

## Project Structure

```
.
├── data/
│   ├── raw/           # Raw data from GDELT and market sources
│   │   ├── gdelt/     # GDELT news data
│   │   └── market/    # Market data for macro assets
│   └── processed/     # Processed datasets and features
│       ├── sentiment/ # Processed sentiment data
│       └── features/  # Engineered features for trading
├── scripts/           # Main pipeline scripts
│   ├── fetch_gdelt_data.py    # Script to fetch news data from GDELT
│   ├── fetch_market_data.py   # Script to fetch market data
│   ├── sentiment_analysis.py  # Script for sentiment analysis
│   ├── prepare_features.py    # Script for feature engineering
│   ├── train_model.py        # Script for model training
│   └── run_backtest.py       # Script for strategy backtesting
├── utils/            # Utility functions and helpers
│   ├── gdelt_utils.py    # GDELT data handling utilities
│   ├── nlp_utils.py      # NLP and sentiment analysis utilities
│   ├── finance_utils.py  # Financial calculations and metrics
│   └── plot_utils.py     # Visualization utilities
├── models/           # Saved models and model artifacts
│   └── finbert/     # FinBERT model and configurations
├── backtest/         # Backtesting results and metrics
│   ├── results/     # Backtest performance results
│   └── metrics/     # Detailed performance metrics
├── visuals/          # Generated plots and visualizations
│   ├── sentiment/   # Sentiment analysis visualizations
│   ├── performance/ # Trading performance charts
│   └── features/    # Feature analysis plots
├── tests/            # Unit tests
│   ├── test_gdelt_utils.py
│   ├── test_nlp_utils.py
│   └── test_finance_utils.py
└── docs/             # Additional documentation
    ├── api/         # API documentation
    └── guides/      # User guides and tutorials
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/macro-news-sentiment-trading.git
cd macro-news-sentiment-trading
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The pipeline can be run in sequence or as individual components:

1. Data Collection:
```bash
python scripts/fetch_gdelt_data.py
python scripts/fetch_market_data.py
```

2. Sentiment Analysis:
```bash
python scripts/sentiment_analysis.py
```

3. Feature Engineering:
```bash
python scripts/prepare_features.py
```

4. Model Training:
```bash
python scripts/train_model.py
```

5. Backtesting:
```bash
python scripts/run_backtest.py
```

## Results

The backtesting results and visualizations can be found in the `backtest/` and `visuals/` directories. Key metrics include:

- **Sharpe Ratio**: Measures the risk-adjusted return of the trading strategy.
- **Maximum Drawdown**: Indicates the largest drop in portfolio value from a peak to a trough.
- **Win Rate**: The percentage of trades that result in a profit.
- **Annualized Returns**: The average annual return of the strategy.

### Performance Metrics Summary

#### EURUSD | LOGISTIC
- **CAGR**: 0.0657
- **Sharpe Ratio**: 0.8257
- **Volatility**: 0.0790
- **Max Drawdown**: -0.1976
- **Win Rate**: 0.5307
- **Total Return**: 0.9167
- **Number of Trades**: 1186
- **Total Cost**: 0.2372
- **Cost per Trade**: 0.0002

#### EURUSD | XGB
- **CAGR**: 0.5543
- **Sharpe Ratio**: 5.8722
- **Volatility**: 0.0738
- **Max Drawdown**: -0.1555
- **Win Rate**: 0.7265
- **Total Return**: 89.8934
- **Number of Trades**: 1158
- **Total Cost**: 0.2316
- **Cost per Trade**: 0.0002

#### USDJPY | LOGISTIC
- **CAGR**: 0.0617
- **Sharpe Ratio**: 0.6649
- **Volatility**: 0.0946
- **Max Drawdown**: -0.1493
- **Win Rate**: 0.5693
- **Total Return**: 0.8443
- **Number of Trades**: 215
- **Total Cost**: 0.0430
- **Cost per Trade**: 0.0002

#### USDJPY | XGB
- **CAGR**: 0.5322
- **Sharpe Ratio**: 4.6479
- **Volatility**: 0.0906
- **Max Drawdown**: -0.2289
- **Win Rate**: 0.7212
- **Total Return**: 77.5373
- **Number of Trades**: 917
- **Total Cost**: 0.1834
- **Cost per Trade**: 0.0002

#### ZN | LOGISTIC
- **CAGR**: -0.0080
- **Sharpe Ratio**: -0.1521
- **Volatility**: 0.0463
- **Max Drawdown**: -0.2042
- **Win Rate**: 0.4712
- **Total Return**: -0.0788
- **Number of Trades**: 778
- **Total Cost**: 0.3890
- **Cost per Trade**: 0.0005

#### ZN | XGB
- **CAGR**: 0.2207
- **Sharpe Ratio**: 4.6529
- **Volatility**: 0.0436
- **Max Drawdown**: -0.0928
- **Win Rate**: 0.6610
- **Total Return**: 6.6851
- **Number of Trades**: 1043
- **Total Cost**: 0.5215
- **Cost per Trade**: 0.0005

### Key Observations
- **EURUSD XGBoost** shows the best performance with a high CAGR and Sharpe ratio.
- **USDJPY XGBoost** also performs well, with a strong CAGR and Sharpe ratio.
- **EURUSD Logistic** and **USDJPY Logistic** show moderate performance with lower Sharpe ratios and CAGRs.
- **ZN Logistic** shows poor performance with a negative CAGR and Sharpe ratio.
- The number of trades and costs vary significantly across models and assets.

## Implementation Logic
- **Data Acquisition**: The system fetches news data from GDELT and market data from Yahoo Finance. The data is processed to extract relevant features for sentiment analysis.
- **Sentiment Analysis**: FinBERT is used to analyze the sentiment of news articles, providing a sentiment score that is integrated into the feature set.
- **Feature Engineering**: Advanced features are created, including Goldstein momentum, article count spike detection, and smoothed sentiment, to enhance the predictive power of the models.
- **Model Training**: Both Logistic Regression and XGBoost models are trained on the engineered features. The models are evaluated using cross-validation to ensure robustness.
- **Backtesting**: The trained models are backtested using historical data to evaluate their performance in a simulated trading environment. Performance metrics such as Sharpe ratio, maximum drawdown, and win rate are calculated to assess the effectiveness of the strategy.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GDELT Project for news data
- ProsusAI for FinBERT model
- Yahoo Finance for market data

## Data Sources

### Market Data
- **Foreign Exchange (FX)**
  - EUR/USD and USD/JPY exchange rates from Yahoo Finance
  - Daily OHLCV data from 2015-01-01 to present

- **Treasury Futures**
  - We attempt to download the back-adjusted continuous 10-year note series (CHRIS/CME_ZN1) via Nasdaq Data Link
  - If access is unavailable (403 error), we fall back to the front-month contract (ZN=F) via Yahoo Finance
  - Note: The front-month contract introduces small roll discontinuities at contract expiration dates
  - Daily OHLCV data from 2015-01-01 to present 