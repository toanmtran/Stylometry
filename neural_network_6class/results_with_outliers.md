# MLP 6-Class Authorship Classification - With Outliers

## None-of-the-5-Authors Class Construction

**15 authors x 10 articles = 150 total passages** labelled `none_of_the_5_authors`.

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
| Train     | 523    | 60% |
| Dev       | 175      | 20%   |
| Test      | 175     | 20%  |
| **Total** | **873**| 100%      |

## Dev Set - Model Selection

Dev accuracy for every feature-subset x architecture combination (patience=15, batch_size=32). Best cell marked with checkmark.

| Feature Subset | Depth 1 (64,) | Depth 3 (64,64,64) | Depth 10 | Depth 50 |
|---|---|---|---|---|
| All 107 features | 0.9429 ✓ | 0.9257 | 0.8971 | 0.1714 |
| Top 50 features | 0.9086 | 0.8286 | 0.9257 | 0.1714 |
| Top 30 features | 0.9143 | 0.9029 | 0.8629 | 0.1714 |

**Best model:** All 107 features x Depth 1 (64,) - Dev accuracy: **0.9429**

## Final Test Set Results

Retrained on train+dev (698 passages) using **All 107 features**, **Depth 1 (64,)**.

### Key Metrics

| Metric | Value |
|--------|-------|
| Accuracy            | 0.9486 |
| Weighted F1         | 0.9485 |
| ROC-AUC (macro OvR) | 0.9960 |

### Per-Class Report

|                       |   precision |   recall |   f1-score |   support |
|:----------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky     |    0.90625  | 0.966667 |   0.935484 |        30 |
| Johnswentworth        |    0.90625  | 0.966667 |   0.935484 |        30 |
| Raemon                |    0.956522 | 0.88     |   0.916667 |        25 |
| Scottalexander        |    0.931034 | 0.9      |   0.915254 |        30 |
| Zvi                   |    1        | 0.966667 |   0.983051 |        30 |
| none_of_the_5_authors |    1        | 1        |   1        |        30 |
| macro avg             |    0.950009 | 0.946667 |   0.947657 |       175 |
| weighted avg          |    0.949823 | 0.948571 |   0.948542 |       175 |

### Confusion Matrix

_Rows = actual, Columns = predicted._

| Actual \ Pred | **Eliezer Yudkow** | **Johnswentworth** | **Raemon** | **Scottalexander** | **Zvi** | **none_of_the_5_** |
|---|---|---|---|---|---|---|
| **Eliezer Yudkow** | 29 | 0 | 0 | 1 | 0 | 0 |
| **Johnswentworth** | 0 | 29 | 0 | 1 | 0 | 0 |
| **Raemon** | 2 | 1 | 22 | 0 | 0 | 0 |
| **Scottalexander** | 1 | 2 | 0 | 27 | 0 | 0 |
| **Zvi** | 0 | 0 | 1 | 0 | 29 | 0 |
| **none_of_the_5_** | 0 | 0 | 0 | 0 | 0 | 30 |

## ROC Curves

![ROC Curves](roc_with_outliers.png)
