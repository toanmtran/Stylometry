# MLP Authorship Classification — With Outliers

## Data Split

| Set | Passages | Proportion |
|-----|----------|------------|
| Train     | 433    | 60% |
| Dev       | 145      | 20%   |
| Test      | 145     | 20%  |
| **Total** | **723**| 100%      |

## Dev Set — Model Selection

All feature-subset × architecture combinations ranked by dev accuracy. Best configuration is retrained on train+dev and evaluated on the test set.

| Rank | Feature Subset | Architecture | Train Acc | Dev Acc |
|------|----------------|-------------|-----------|----------|
| 1 | Top 74 features | Depth 1 (64,) | 1.0000 | 0.9448 ✓ |
| 2 | Top 50 features | Depth 2 (64, 32) | 1.0000 | 0.9379 |
| 3 | Top 50 features | Depth 1 (64,) | 1.0000 | 0.9310 |
| 4 | Top 74 features | Depth 2 (64, 32) | 1.0000 | 0.9241 |
| 5 | All 107 features | Depth 3 (64,64,64) | 1.0000 | 0.9241 |
| 6 | Top 30 features | Depth 3 (64,64,64) | 1.0000 | 0.9103 |
| 7 | Top 74 features | Depth 3 (64,64,64) | 1.0000 | 0.9103 |
| 8 | All 107 features | Depth 1 (64,) | 1.0000 | 0.9103 |
| 9 | Top 50 features | Depth 3 (64,64,64) | 1.0000 | 0.9034 |
| 10 | All 107 features | Depth 2 (64, 32) | 1.0000 | 0.9034 |
| 11 | Top 30 features | Depth 1 (64,) | 1.0000 | 0.8966 |
| 12 | Top 30 features | Depth 2 (64, 32) | 1.0000 | 0.8966 |
| 13 | Top 50 features | Depth 10 | 1.0000 | 0.8897 |
| 14 | Top 30 features | Depth 10 | 1.0000 | 0.8690 |
| 15 | Top 74 features | Depth 10 | 1.0000 | 0.8690 |
| 16 | All 107 features | Depth 10 | 1.0000 | 0.8690 |
| 17 | Top 15 features | Depth 1 (64,) | 1.0000 | 0.8138 |
| 18 | Top 15 features | Depth 2 (64, 32) | 1.0000 | 0.8069 |
| 19 | Top 15 features | Depth 3 (64,64,64) | 1.0000 | 0.8069 |
| 20 | Top 15 features | Depth 10 | 1.0000 | 0.8069 |
| 21 | Top 15 features | Depth 50 | 0.2079 | 0.2069 |
| 22 | Top 30 features | Depth 50 | 0.2079 | 0.2069 |
| 23 | Top 50 features | Depth 50 | 0.2079 | 0.2069 |
| 24 | Top 74 features | Depth 50 | 0.2079 | 0.2069 |
| 25 | All 107 features | Depth 50 | 0.2079 | 0.2069 |

**Best model:** Top 74 features · Depth 1 (64,) — Dev accuracy: **0.9448**

## Final Test Set Results

Retrained on train+dev (578 passages) using **Top 74 features**, **Depth 1 (64,)**.

**Test Accuracy: 0.9379**

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    1        | 0.966667 |   0.983051 |        30 |
| Johnswentworth    |    0.878788 | 0.966667 |   0.920635 |        30 |
| Raemon            |    0.923077 | 0.96     |   0.941176 |        25 |
| Scottalexander    |    0.925926 | 0.833333 |   0.877193 |        30 |
| Zvi               |    0.966667 | 0.966667 |   0.966667 |        30 |
| macro avg         |    0.938891 | 0.938667 |   0.937744 |       145 |
| weighted avg      |    0.939437 | 0.937931 |   0.937626 |       145 |
