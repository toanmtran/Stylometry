# MLP Authorship Classification - With Outliers

## Data Split

| Set | Passages | Proportion |
|-----|----------|------------|
| Train     | 433    | 60% |
| Dev       | 145      | 20%   |
| Test      | 145     | 20%  |
| **Total** | **723**| 100%      |

## Dev Set - Model Selection

Dev accuracy for every feature-subset x architecture combination (patience=15, batch_size=32). Best cell marked with checkmark.

| Feature Subset | Depth 1 (64,) | Depth 3 (64,64,64) | Depth 10 | Depth 50 |
|---|---|---|---|---|
| All 107 features | 0.9034 ✓ | 0.8690 | 0.8414 | 0.2069 |
| Top 50 features | 0.8414 | 0.8621 | 0.8759 | 0.2069 |
| Top 30 features | 0.8345 | 0.8069 | 0.8621 | 0.2069 |

**Best model:** All 107 features x Depth 1 (64,) - Dev accuracy: **0.9034**

## Final Test Set Results

Retrained on train+dev (578 passages) using **All 107 features**, **Depth 1 (64,)**.

### Key Metrics

| Metric | Value |
|--------|-------|
| Accuracy            | 0.9310 |
| Weighted F1         | 0.9297 |
| ROC-AUC (macro OvR) | 0.9951 |

### Per-Class Report

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.909091 | 1        |   0.952381 |        30 |
| Johnswentworth    |    0.933333 | 0.933333 |   0.933333 |        30 |
| Raemon            |    0.884615 | 0.92     |   0.901961 |        25 |
| Scottalexander    |    0.96     | 0.8      |   0.872727 |        30 |
| Zvi               |    0.967742 | 1        |   0.983607 |        30 |
| macro avg         |    0.930956 | 0.930667 |   0.928802 |       145 |
| weighted avg      |    0.932554 | 0.931034 |   0.929727 |       145 |

### Confusion Matrix

_Rows = actual, Columns = predicted._

| Actual \ Pred | **Eliezer Yudkow** | **Johnswentworth** | **Raemon** | **Scottalexander** | **Zvi** |
|---|---|---|---|---|---|
| **Eliezer Yudkow** | 30 | 0 | 0 | 0 | 0 |
| **Johnswentworth** | 1 | 28 | 0 | 1 | 0 |
| **Raemon** | 1 | 1 | 23 | 0 | 0 |
| **Scottalexander** | 1 | 1 | 3 | 24 | 1 |
| **Zvi** | 0 | 0 | 0 | 0 | 30 |

## ROC Curves

![ROC Curves](roc_with_outliers.png)
