# MLP Authorship Classification — Without Outliers

## Data Split

| Set | Passages | Proportion |
|-----|----------|------------|
| Train     | 411    | 60% |
| Dev       | 137      | 20%   |
| Test      | 138     | 20%  |
| **Total** | **686**| 100%      |

## Dev Set — Model Selection

All feature-subset × architecture combinations ranked by dev accuracy. Best configuration is retrained on train+dev and evaluated on the test set.

| Rank | Feature Subset | Architecture | Patience | Train Acc | Dev Acc |
|------|----------------|-------------|----------|-----------|----------|
| 1 | All 107 features | Depth 2 (64, 32) | 10 | 0.9878 | 0.8978 ✓ |
| 2 | All 107 features | Depth 2 (64, 32) | 15 | 0.9878 | 0.8978 |
| 3 | Top 50 features | Depth 2 (64, 32) | 15 | 0.9781 | 0.8759 |
| 4 | Top 74 features | Depth 2 (64, 32) | 5 | 0.9319 | 0.8759 |
| 5 | Top 74 features | Depth 2 (64, 32) | 10 | 0.9319 | 0.8759 |
| 6 | Top 74 features | Depth 2 (64, 32) | 15 | 0.9319 | 0.8759 |
| 7 | Top 74 features | Depth 1 (64,) | 10 | 0.9586 | 0.8686 |
| 8 | Top 74 features | Depth 1 (64,) | 15 | 0.9586 | 0.8686 |
| 9 | Top 74 features | Depth 3 (64,64,64) | 5 | 0.9270 | 0.8686 |
| 10 | Top 74 features | Depth 3 (64,64,64) | 10 | 0.9270 | 0.8686 |
| 11 | Top 74 features | Depth 3 (64,64,64) | 15 | 0.9270 | 0.8686 |
| 12 | Top 30 features | Depth 10 | 5 | 0.9343 | 0.8613 |
| 13 | Top 30 features | Depth 10 | 10 | 0.9343 | 0.8613 |
| 14 | Top 30 features | Depth 10 | 15 | 0.9343 | 0.8613 |
| 15 | All 107 features | Depth 3 (64,64,64) | 10 | 0.9513 | 0.8540 |
| 16 | All 107 features | Depth 3 (64,64,64) | 15 | 0.9513 | 0.8540 |
| 17 | All 107 features | Depth 2 (64, 32) | 5 | 0.9148 | 0.8467 |
| 18 | Top 30 features | Depth 3 (64,64,64) | 10 | 0.9294 | 0.8394 |
| 19 | Top 30 features | Depth 3 (64,64,64) | 15 | 0.9294 | 0.8394 |
| 20 | Top 50 features | Depth 1 (64,) | 15 | 0.9708 | 0.8394 |
| 21 | Top 50 features | Depth 3 (64,64,64) | 5 | 0.9173 | 0.8394 |
| 22 | Top 50 features | Depth 3 (64,64,64) | 10 | 0.9173 | 0.8394 |
| 23 | Top 50 features | Depth 3 (64,64,64) | 15 | 0.9173 | 0.8394 |
| 24 | Top 30 features | Depth 1 (64,) | 10 | 0.9075 | 0.8321 |
| 25 | Top 30 features | Depth 1 (64,) | 15 | 0.9075 | 0.8321 |
| 26 | Top 50 features | Depth 2 (64, 32) | 5 | 0.9246 | 0.8321 |
| 27 | Top 50 features | Depth 2 (64, 32) | 10 | 0.9246 | 0.8321 |
| 28 | Top 50 features | Depth 10 | 5 | 0.9562 | 0.8175 |
| 29 | Top 50 features | Depth 10 | 10 | 0.9562 | 0.8175 |
| 30 | Top 50 features | Depth 10 | 15 | 0.9562 | 0.8175 |
| 31 | Top 74 features | Depth 1 (64,) | 5 | 0.8978 | 0.8175 |
| 32 | Top 30 features | Depth 2 (64, 32) | 5 | 0.8540 | 0.8102 |
| 33 | Top 30 features | Depth 2 (64, 32) | 10 | 0.8540 | 0.8102 |
| 34 | Top 30 features | Depth 2 (64, 32) | 15 | 0.8540 | 0.8102 |
| 35 | Top 50 features | Depth 1 (64,) | 5 | 0.9148 | 0.8102 |
| 36 | Top 50 features | Depth 1 (64,) | 10 | 0.9148 | 0.8102 |
| 37 | All 107 features | Depth 3 (64,64,64) | 5 | 0.8735 | 0.8102 |
| 38 | All 107 features | Depth 1 (64,) | 5 | 0.9027 | 0.8029 |
| 39 | All 107 features | Depth 1 (64,) | 10 | 0.9027 | 0.8029 |
| 40 | All 107 features | Depth 1 (64,) | 15 | 0.9027 | 0.8029 |
| 41 | All 107 features | Depth 10 | 10 | 0.9051 | 0.8029 |
| 42 | All 107 features | Depth 10 | 15 | 0.9051 | 0.8029 |
| 43 | Top 74 features | Depth 10 | 5 | 0.9538 | 0.7956 |
| 44 | Top 74 features | Depth 10 | 10 | 0.9538 | 0.7956 |
| 45 | Top 74 features | Depth 10 | 15 | 0.9538 | 0.7956 |
| 46 | Top 15 features | Depth 2 (64, 32) | 15 | 0.8491 | 0.7883 |
| 47 | Top 30 features | Depth 3 (64,64,64) | 5 | 0.8832 | 0.7883 |
| 48 | Top 15 features | Depth 1 (64,) | 10 | 0.8418 | 0.7591 |
| 49 | Top 15 features | Depth 1 (64,) | 15 | 0.8418 | 0.7591 |
| 50 | Top 30 features | Depth 1 (64,) | 5 | 0.8175 | 0.7518 |
| 51 | Top 15 features | Depth 3 (64,64,64) | 5 | 0.7616 | 0.7445 |
| 52 | Top 15 features | Depth 3 (64,64,64) | 10 | 0.7616 | 0.7445 |
| 53 | Top 15 features | Depth 3 (64,64,64) | 15 | 0.7616 | 0.7445 |
| 54 | Top 15 features | Depth 2 (64, 32) | 10 | 0.8005 | 0.7372 |
| 55 | Top 15 features | Depth 10 | 5 | 0.7956 | 0.7372 |
| 56 | Top 15 features | Depth 10 | 10 | 0.7956 | 0.7372 |
| 57 | Top 15 features | Depth 10 | 15 | 0.7956 | 0.7372 |
| 58 | Top 15 features | Depth 2 (64, 32) | 5 | 0.7275 | 0.6861 |
| 59 | Top 15 features | Depth 1 (64,) | 5 | 0.7226 | 0.6715 |
| 60 | All 107 features | Depth 10 | 5 | 0.5839 | 0.6058 |
| 61 | Top 15 features | Depth 50 | 5 | 0.2141 | 0.2117 |
| 62 | Top 15 features | Depth 50 | 10 | 0.2141 | 0.2117 |
| 63 | Top 15 features | Depth 50 | 15 | 0.2141 | 0.2117 |
| 64 | Top 30 features | Depth 50 | 5 | 0.2141 | 0.2117 |
| 65 | Top 30 features | Depth 50 | 10 | 0.2141 | 0.2117 |
| 66 | Top 30 features | Depth 50 | 15 | 0.2141 | 0.2117 |
| 67 | Top 50 features | Depth 50 | 5 | 0.2141 | 0.2117 |
| 68 | Top 50 features | Depth 50 | 10 | 0.2141 | 0.2117 |
| 69 | Top 50 features | Depth 50 | 15 | 0.2141 | 0.2117 |
| 70 | All 107 features | Depth 50 | 5 | 0.2141 | 0.2117 |
| 71 | All 107 features | Depth 50 | 10 | 0.2141 | 0.2117 |
| 72 | All 107 features | Depth 50 | 15 | 0.2141 | 0.2117 |
| 73 | Top 74 features | Depth 50 | 5 | 0.2068 | 0.2044 |
| 74 | Top 74 features | Depth 50 | 10 | 0.2068 | 0.2044 |
| 75 | Top 74 features | Depth 50 | 15 | 0.2068 | 0.2044 |

**Best model:** All 107 features · Depth 2 (64, 32) · patience=10 — Dev accuracy: **0.8978**

## Final Test Set Results

Retrained on train+dev (548 passages) using **All 107 features**, **Depth 2 (64, 32)**.

### Key Metrics

| Metric | Value |
|--------|-------|
| Accuracy       | 0.8913 |
| Weighted F1    | 0.8913 |
| ROC-AUC (macro OvR) | 0.9894 |
| ECE            | 0.0452 |

### Per-Class Report

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.931034 | 0.964286 |   0.947368 |        28 |
| Johnswentworth    |    0.931034 | 0.9      |   0.915254 |        30 |
| Raemon            |    0.769231 | 0.833333 |   0.8      |        24 |
| Scottalexander    |    0.846154 | 0.785714 |   0.814815 |        28 |
| Zvi               |    0.964286 | 0.964286 |   0.964286 |        28 |
| macro avg         |    0.888348 | 0.889524 |   0.888345 |       138 |
| weighted avg      |    0.892419 | 0.891304 |   0.891295 |       138 |

### Confusion Matrix

_Rows = actual, Columns = predicted._

| Actual \ Pred | **Eliezer Yudkow** | **Johnswentworth** | **Raemon** | **Scottalexander** | **Zvi** |
|---|---|---|---|---|---|
| **Eliezer Yudkow** | 27 | 0 | 0 | 0 | 1 |
| **Johnswentworth** | 0 | 27 | 1 | 2 | 0 |
| **Raemon** | 1 | 1 | 20 | 2 | 0 |
| **Scottalexander** | 1 | 1 | 4 | 22 | 0 |
| **Zvi** | 0 | 0 | 1 | 0 | 27 |

## ROC Curves

![ROC Curves](roc_without_outliers.png)
