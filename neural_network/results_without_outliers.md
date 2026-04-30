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

| Rank | Feature Subset | Architecture | Train Acc | Dev Acc |
|------|----------------|-------------|-----------|----------|
| 1 | Top 50 features | Depth 1 (64,) | 1.0000 | 0.9343 ✓ |
| 2 | Top 50 features | Depth 2 (64, 32) | 1.0000 | 0.9270 |
| 3 | Top 74 features | Depth 1 (64,) | 1.0000 | 0.9197 |
| 4 | All 107 features | Depth 1 (64,) | 1.0000 | 0.9197 |
| 5 | All 107 features | Depth 2 (64, 32) | 1.0000 | 0.9197 |
| 6 | Top 74 features | Depth 2 (64, 32) | 1.0000 | 0.9124 |
| 7 | Top 74 features | Depth 3 (64,64,64) | 1.0000 | 0.9124 |
| 8 | All 107 features | Depth 3 (64,64,64) | 1.0000 | 0.9124 |
| 9 | Top 50 features | Depth 3 (64,64,64) | 1.0000 | 0.8905 |
| 10 | All 107 features | Depth 10 | 1.0000 | 0.8905 |
| 11 | Top 30 features | Depth 2 (64, 32) | 1.0000 | 0.8832 |
| 12 | Top 74 features | Depth 10 | 1.0000 | 0.8759 |
| 13 | Top 50 features | Depth 10 | 1.0000 | 0.8686 |
| 14 | Top 30 features | Depth 3 (64,64,64) | 1.0000 | 0.8613 |
| 15 | Top 30 features | Depth 10 | 1.0000 | 0.8613 |
| 16 | Top 30 features | Depth 1 (64,) | 1.0000 | 0.8540 |
| 17 | Top 15 features | Depth 2 (64, 32) | 0.9951 | 0.8394 |
| 18 | Top 15 features | Depth 3 (64,64,64) | 1.0000 | 0.8394 |
| 19 | Top 15 features | Depth 1 (64,) | 0.9659 | 0.8248 |
| 20 | Top 15 features | Depth 10 | 0.9538 | 0.8102 |
| 21 | Top 15 features | Depth 50 | 0.2044 | 0.2117 |
| 22 | Top 74 features | Depth 50 | 0.2044 | 0.2117 |
| 23 | All 107 features | Depth 50 | 0.2141 | 0.2117 |
| 24 | Top 30 features | Depth 50 | 0.2068 | 0.2044 |
| 25 | Top 50 features | Depth 50 | 0.1995 | 0.1971 |

**Best model:** Top 50 features · Depth 1 (64,) — Dev accuracy: **0.9343**

## Final Test Set Results

Retrained on train+dev (548 passages) using **Top 50 features**, **Depth 1 (64,)**.

**Test Accuracy: 0.9638**

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.965517 | 1        |   0.982456 |        28 |
| Johnswentworth    |    0.967742 | 1        |   0.983607 |        30 |
| Raemon            |    0.954545 | 0.875    |   0.913043 |        24 |
| Scottalexander    |    0.931034 | 0.964286 |   0.947368 |        28 |
| Zvi               |    1        | 0.964286 |   0.981818 |        28 |
| macro avg         |    0.963768 | 0.960714 |   0.961659 |       138 |
| weighted avg      |    0.964093 | 0.963768 |   0.963386 |       138 |
