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

- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Annualized Returns

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