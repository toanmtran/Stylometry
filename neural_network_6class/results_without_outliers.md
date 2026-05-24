# MLP 6-Class Authorship Classification — Without Outliers

## None-of-the-5-Authors Class Construction

**15 authors × 10 articles = 150 total passages** labelled `none_of_the_5_authors`.

Authors used for the none class:

- `ricraz`
- `benquo`
- `abramdemski`
- `sarahconstantin`
- `holdenkarnofsky`
- `gordon-seidoh-worley`
- `screwtape`
- `buck`
- `turntrout`
- `nunosempere`
- `benito`
- `petermccluskey`
- `joe-carlsmith`
- `adamshimi`
- `tsvibt`

## Data Split

| Set | Passages | Proportion |
|-----|----------|------------|
| Train     | 501    | 60% |
| Dev       | 168      | 20%   |
| Test      | 168     | 20%  |
| **Total** | **837**| 100%      |

## Dev Set — Model Selection

All feature-subset × architecture combinations ranked by dev accuracy. Best configuration is retrained on train+dev and evaluated on the test set.

| Rank | Feature Subset | Architecture | Patience | Train Acc | Dev Acc |
|------|----------------|-------------|----------|-----------|----------|
| 1 | Top 74 features | Depth 3 (64,64,64) | 10 | 0.9840 | 0.9345 ✓ |
| 2 | Top 74 features | Depth 3 (64,64,64) | 15 | 0.9840 | 0.9345 |
| 3 | Top 74 features | Depth 2 (64, 32) | 15 | 0.9820 | 0.9286 |
| 4 | Top 50 features | Depth 3 (64,64,64) | 10 | 0.9721 | 0.9167 |
| 5 | Top 50 features | Depth 3 (64,64,64) | 15 | 0.9721 | 0.9167 |
| 6 | Top 74 features | Depth 1 (64,) | 15 | 0.9760 | 0.9167 |
| 7 | All 107 features | Depth 2 (64, 32) | 10 | 0.9760 | 0.9107 |
| 8 | All 107 features | Depth 2 (64, 32) | 15 | 0.9760 | 0.9107 |
| 9 | All 107 features | Depth 10 | 5 | 0.9860 | 0.9107 |
| 10 | All 107 features | Depth 10 | 10 | 0.9860 | 0.9107 |
| 11 | All 107 features | Depth 10 | 15 | 0.9860 | 0.9107 |
| 12 | All 107 features | Depth 1 (64,) | 10 | 0.9581 | 0.9048 |
| 13 | All 107 features | Depth 1 (64,) | 15 | 0.9581 | 0.9048 |
| 14 | Top 74 features | Depth 1 (64,) | 10 | 0.9661 | 0.8988 |
| 15 | Top 74 features | Depth 3 (64,64,64) | 5 | 0.9561 | 0.8988 |
| 16 | Top 30 features | Depth 1 (64,) | 10 | 0.9421 | 0.8929 |
| 17 | Top 30 features | Depth 1 (64,) | 15 | 0.9421 | 0.8929 |
| 18 | Top 50 features | Depth 3 (64,64,64) | 5 | 0.9281 | 0.8929 |
| 19 | Top 30 features | Depth 10 | 5 | 0.9741 | 0.8869 |
| 20 | Top 30 features | Depth 10 | 10 | 0.9741 | 0.8869 |
| 21 | Top 30 features | Depth 10 | 15 | 0.9741 | 0.8869 |
| 22 | Top 50 features | Depth 10 | 10 | 0.9900 | 0.8869 |
| 23 | Top 50 features | Depth 10 | 15 | 0.9900 | 0.8869 |
| 24 | All 107 features | Depth 3 (64,64,64) | 10 | 0.9701 | 0.8869 |
| 25 | All 107 features | Depth 3 (64,64,64) | 15 | 0.9701 | 0.8869 |
| 26 | Top 30 features | Depth 3 (64,64,64) | 5 | 0.9421 | 0.8810 |
| 27 | Top 30 features | Depth 3 (64,64,64) | 10 | 0.9421 | 0.8810 |
| 28 | Top 30 features | Depth 3 (64,64,64) | 15 | 0.9421 | 0.8810 |
| 29 | Top 50 features | Depth 1 (64,) | 10 | 0.9401 | 0.8810 |
| 30 | Top 50 features | Depth 1 (64,) | 15 | 0.9401 | 0.8810 |
| 31 | Top 74 features | Depth 2 (64, 32) | 5 | 0.9481 | 0.8810 |
| 32 | Top 74 features | Depth 2 (64, 32) | 10 | 0.9481 | 0.8810 |
| 33 | All 107 features | Depth 2 (64, 32) | 5 | 0.9222 | 0.8810 |
| 34 | Top 15 features | Depth 3 (64,64,64) | 10 | 0.9321 | 0.8750 |
| 35 | Top 15 features | Depth 3 (64,64,64) | 15 | 0.9321 | 0.8750 |
| 36 | Top 15 features | Depth 10 | 5 | 0.9182 | 0.8750 |
| 37 | Top 15 features | Depth 10 | 10 | 0.9182 | 0.8750 |
| 38 | Top 15 features | Depth 10 | 15 | 0.9182 | 0.8750 |
| 39 | Top 30 features | Depth 2 (64, 32) | 5 | 0.9301 | 0.8750 |
| 40 | Top 30 features | Depth 2 (64, 32) | 10 | 0.9301 | 0.8750 |
| 41 | Top 30 features | Depth 2 (64, 32) | 15 | 0.9301 | 0.8750 |
| 42 | Top 30 features | Depth 1 (64,) | 5 | 0.8882 | 0.8690 |
| 43 | Top 50 features | Depth 1 (64,) | 5 | 0.9162 | 0.8690 |
| 44 | Top 15 features | Depth 3 (64,64,64) | 5 | 0.8982 | 0.8571 |
| 45 | Top 50 features | Depth 2 (64, 32) | 5 | 0.9321 | 0.8571 |
| 46 | Top 50 features | Depth 2 (64, 32) | 10 | 0.9321 | 0.8571 |
| 47 | Top 50 features | Depth 2 (64, 32) | 15 | 0.9321 | 0.8571 |
| 48 | Top 50 features | Depth 10 | 5 | 0.9381 | 0.8512 |
| 49 | Top 74 features | Depth 1 (64,) | 5 | 0.9301 | 0.8512 |
| 50 | All 107 features | Depth 1 (64,) | 5 | 0.9062 | 0.8512 |
| 51 | All 107 features | Depth 3 (64,64,64) | 5 | 0.8982 | 0.8452 |
| 52 | Top 15 features | Depth 2 (64, 32) | 10 | 0.8862 | 0.8214 |
| 53 | Top 15 features | Depth 2 (64, 32) | 15 | 0.8862 | 0.8214 |
| 54 | Top 15 features | Depth 2 (64, 32) | 5 | 0.8303 | 0.8036 |
| 55 | Top 74 features | Depth 10 | 5 | 0.8184 | 0.8036 |
| 56 | Top 74 features | Depth 10 | 10 | 0.8184 | 0.8036 |
| 57 | Top 74 features | Depth 10 | 15 | 0.8184 | 0.8036 |
| 58 | Top 15 features | Depth 1 (64,) | 5 | 0.7565 | 0.7321 |
| 59 | Top 15 features | Depth 1 (64,) | 10 | 0.7565 | 0.7321 |
| 60 | Top 15 features | Depth 1 (64,) | 15 | 0.7565 | 0.7321 |
| 61 | Top 15 features | Depth 50 | 10 | 0.1756 | 0.1786 |
| 62 | Top 15 features | Depth 50 | 15 | 0.1756 | 0.1786 |
| 63 | Top 30 features | Depth 50 | 5 | 0.1756 | 0.1786 |
| 64 | Top 30 features | Depth 50 | 10 | 0.1756 | 0.1786 |
| 65 | Top 30 features | Depth 50 | 15 | 0.1756 | 0.1786 |
| 66 | Top 50 features | Depth 50 | 5 | 0.1756 | 0.1786 |
| 67 | Top 50 features | Depth 50 | 10 | 0.1756 | 0.1786 |
| 68 | Top 50 features | Depth 50 | 15 | 0.1756 | 0.1786 |
| 69 | Top 74 features | Depth 50 | 5 | 0.1756 | 0.1786 |
| 70 | All 107 features | Depth 50 | 5 | 0.1737 | 0.1726 |
| 71 | All 107 features | Depth 50 | 10 | 0.1737 | 0.1726 |
| 72 | All 107 features | Depth 50 | 15 | 0.1737 | 0.1726 |
| 73 | Top 15 features | Depth 50 | 5 | 0.1677 | 0.1667 |
| 74 | Top 74 features | Depth 50 | 10 | 0.1697 | 0.1667 |
| 75 | Top 74 features | Depth 50 | 15 | 0.1697 | 0.1667 |

**Best model:** Top 74 features · Depth 3 (64,64,64) · patience=10 — Dev accuracy: **0.9345**

## Final Test Set Results

Retrained on train+dev (669 passages) using **Top 74 features**, **Depth 3 (64,64,64)**.

### Key Metrics

| Metric | Value |
|--------|-------|
| Accuracy            | 0.9167 |
| Weighted F1         | 0.9176 |
| ROC-AUC (macro OvR) | 0.9939 |
| ECE                 | 0.0368 |

### Per-Class Report

|                       |   precision |   recall |   f1-score |   support |
|:----------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky     |    0.9      | 0.931034 |   0.915254 |        29 |
| Johnswentworth        |    0.961538 | 0.862069 |   0.909091 |        29 |
| Raemon                |    0.913043 | 0.875    |   0.893617 |        24 |
| Scottalexander        |    0.78125  | 0.892857 |   0.833333 |        28 |
| Zvi                   |    0.962963 | 0.928571 |   0.945455 |        28 |
| none_of_the_5_authors |    1        | 1        |   1        |        30 |
| macro avg             |    0.919799 | 0.914922 |   0.916125 |       168 |
| weighted avg          |    0.921045 | 0.916667 |   0.917612 |       168 |

### Confusion Matrix

_Rows = actual, Columns = predicted._

| Actual \ Pred | **Eliezer Yudkow** | **Johnswentworth** | **Raemon** | **Scottalexander** | **Zvi** | **none_of_the_5_** |
|---|---|---|---|---|---|---|
| **Eliezer Yudkow** | 27 | 0 | 1 | 1 | 0 | 0 |
| **Johnswentworth** | 1 | 25 | 0 | 3 | 0 | 0 |
| **Raemon** | 1 | 1 | 21 | 1 | 0 | 0 |
| **Scottalexander** | 1 | 0 | 1 | 25 | 1 | 0 |
| **Zvi** | 0 | 0 | 0 | 2 | 26 | 0 |
| **none_of_the_5_** | 0 | 0 | 0 | 0 | 0 | 30 |

## ROC Curves

![ROC Curves](roc_without_outliers.png)
