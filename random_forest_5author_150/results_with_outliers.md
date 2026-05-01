# Random Forest Authorship Classification — With Outliers

## Configuration

- **Classifier:** Random Forest
- **Outer folds:** 5 (performance estimation)
- **Inner folds:** 3 (hyperparameter tuning via GridSearchCV)
- **Param combinations:** 36
- **Passages:** 723
- **Features:** all 107

**Search grid:**

| Hyperparameter | Values |
|----------------|--------|
| `n_estimators` | [100, 200, 300] |
| `max_depth` | [None, 10, 20] |
| `min_samples_split` | [2, 5] |
| `max_features` | ['sqrt', 'log2'] |

## Per-Fold Results

| Fold | Accuracy | Precision (macro) | Recall (macro) | Weighted F1 | Best Params |
|------|----------|-------------------|----------------|-------------|-------------|
| 1 | 0.8690 | 0.8777 | 0.8671 | 0.8701 | `max_depth=10, max_features=sqrt, min_samples_split=2, n_estimators=200` |
| 2 | 0.9172 | 0.9202 | 0.9187 | 0.9181 | `max_depth=None, max_features=log2, min_samples_split=5, n_estimators=200` |
| 3 | 0.9241 | 0.9309 | 0.9227 | 0.9252 | `max_depth=None, max_features=log2, min_samples_split=2, n_estimators=200` |
| 4 | 0.9583 | 0.9586 | 0.9595 | 0.9584 | `max_depth=None, max_features=sqrt, min_samples_split=2, n_estimators=200` |
| 5 | 0.9444 | 0.9479 | 0.9435 | 0.9450 | `max_depth=None, max_features=log2, min_samples_split=2, n_estimators=100` |

## Summary

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy           | 0.9226  | 0.0341  |
| Precision (macro)  | 0.9271 | 0.0313 |
| Recall (macro)     | 0.9223    | 0.0350    |
| Weighted F1        | 0.9234      | 0.0338      |

## Average Classification Report

_Per-class metrics averaged across all outer folds._

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.959297 | 0.906667 |   0.930899 |      30   |
| Johnswentworth    |    0.941853 | 0.933333 |   0.93619  |      30   |
| Raemon            |    0.931305 | 0.913538 |   0.919838 |      25.2 |
| Scottalexander    |    0.82227  | 0.891264 |   0.854808 |      29.4 |
| Zvi               |    0.980645 | 0.966667 |   0.97332  |      30   |
| macro avg         |    0.927074 | 0.922294 |   0.923011 |     144.6 |
| weighted avg      |    0.927401 | 0.922625 |   0.923372 |     144.6 |
