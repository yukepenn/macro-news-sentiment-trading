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
- Updated GDELT data fetching to use BigQuery instead of GDELT API
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