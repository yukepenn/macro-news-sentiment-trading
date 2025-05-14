# Error Analysis Report

This document provides a detailed analysis of the sentiment-based trading model's performance, focusing on error cases and their underlying causes.

## Model Performance Overview

The model's performance is evaluated using standard classification metrics:

- Confusion Matrix: Shows the distribution of true positives, false positives, true negatives, and false negatives
- Classification Report: Provides precision, recall, and F1-score for both up and down predictions

## Error Analysis

### False Positives (Predicted Up, Market Down)

The top false positive cases are documented in `backtest/top_false_positives.csv`. These represent cases where the model predicted a market up-move but the market actually declined. For each case, we analyze:

1. The sentiment features that led to the prediction
2. The market context around the prediction
3. Potential reasons for the misclassification

### False Negatives (Predicted Down, Market Up)

The top false negative cases are documented in `backtest/top_false_negatives.csv`. These represent missed opportunities where the model predicted a down-move but the market rallied. For each case, we analyze:

1. Why the model failed to capture the positive sentiment
2. The market conditions that led to the rally
3. Potential improvements to capture similar cases

## Case Studies

### Case Study 1: [Date] - False Positive

**Context:**
- Market Return: [X]%
- Model Prediction: Up (Probability: [Y]%)
- Actual Outcome: Down

**Key Features:**
- Sentiment Score: [Z]
- Article Count: [N]
- Goldstein Score: [G]

**Analysis:**
[Detailed analysis of why the model misclassified this case]

### Case Study 2: [Date] - False Negative

**Context:**
- Market Return: [X]%
- Model Prediction: Down (Probability: [Y]%)
- Actual Outcome: Up

**Key Features:**
- Sentiment Score: [Z]
- Article Count: [N]
- Goldstein Score: [G]

**Analysis:**
[Detailed analysis of why the model missed this opportunity]

## SHAP Analysis

SHAP (SHapley Additive exPlanations) waterfall plots are generated for selected cases to understand feature contributions:

1. `visuals/waterfall_fp_[DATE].png`: SHAP analysis for false positive cases
2. `visuals/waterfall_fn_[DATE].png`: SHAP analysis for false negative cases

These plots show how each feature contributed to the model's prediction, helping identify potential areas for improvement.

## Recommendations

Based on the error analysis, we recommend the following improvements:

1. **Feature Engineering:**
   - [Specific feature engineering suggestions]
   - [Potential new features to consider]

2. **Model Adjustments:**
   - [Suggested model parameter adjustments]
   - [Potential alternative model architectures]

3. **Data Quality:**
   - [Suggestions for improving data quality]
   - [Potential data sources to consider]

## Next Steps

1. Implement the recommended improvements
2. Retrain the model with the new features/parameters
3. Conduct another round of error analysis
4. Document the impact of changes on model performance

## Visualizations

The following visualizations are available in the `visuals/` directory:

1. Case study plots showing sentiment and returns around error dates
2. SHAP waterfall plots for selected cases
3. Confusion matrix visualization
4. Feature importance plots

## Conclusion

This error analysis provides valuable insights into the model's strengths and weaknesses. By addressing the identified issues and implementing the recommended improvements, we can enhance the model's predictive power and trading performance. 