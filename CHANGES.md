# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced model tuning with threshold optimization:
  - Added confidence-based trading thresholds (0.55/0.45)
  - Implemented grid search for optimal threshold pairs
  - Added strategy evaluation metrics (Sharpe, win rate, turnover)
  - Enhanced model saving to include optimal thresholds
- Enhanced feature engineering with new advanced features:
  - Goldstein momentum with multiple decay rates (0.1, 0.3, 0.5)
  - Article count spike detection using z-scores
  - Smoothed sentiment using exponential weighted moving averages
  - New interaction features combining spikes and sentiment
- Initial project setup with basic structure
- README.md with project overview and setup instructions
- Basic data fetching scripts for GDELT and market data
- Requirements.txt with necessary dependencies
- Initial project structure
- Basic utility modules for data handling and analysis
- Sentiment analysis script using FinBERT
- Feature preparation script for combining sentiment and market data
- Project directory structure (data/, models/, backtest/, visuals/, tests/, docs/)
- Basic test files for utilities
- API documentation

### Changed
- Updated GDELT data fetching to use DOC API for full article content
  - Switched from events API to DOC API for richer content
  - Added support for article titles and descriptions
  - Improved error handling and logging
  - Added pagination support for large date ranges
- Added Google Cloud dependencies to requirements.txt
- Updated project structure to include processed data directory
- Enhanced data processing pipeline with feature engineering

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

## [0.1.0] - 2024-03-XX

### Added
- Project initialization
- Basic project structure
- Essential configuration files

## Latest Changes

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