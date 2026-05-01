# XGBoost — Stylometric Author Classification

XGBoost authorship classifier evaluated with nested cross-validation on stylometric features extracted from LessWrong blog posts. Five target authors: Eliezer Yudkowsky, Johnswentworth, Raemon, Scottalexander, and Zvi.

---

## Evaluation Design

**Nested cross-validation** produces an unbiased generalisation estimate while tuning hyperparameters:

```
Outer fold k  (×5)  ─► performance estimate on held-out fold
  └─ Inner fold j (×3) ─► GridSearchCV selects best hyperparameters
                           on the outer training split
```

Both cases run independently from a single command:

| Case | Input |
|------|-------|
| With outliers | `../neural_network/author_features_extracted_full.csv` |
| Without outliers | `../neural_network/feature_extracted_without_outliers.csv` |

---

## Quick Start

```bash
python xgboost_code.py
```

**Outputs:** `results_with_outliers.md`, `results_without_outliers.md`

---

## Files

### `xgboost_code.py`
Main script. Runs nested CV for both dataset variants and writes one Markdown result file per case.

### `results_with_outliers.md`
Nested CV results on the full dataset (723 passages).

### `results_without_outliers.md`
Nested CV results on the outlier-removed dataset (686 passages).

---

## Model Configuration

| Setting | Value |
|---------|-------|
| Classifier | XGBoost (`XGBClassifier`) |
| Outer folds | 5 (performance estimation) |
| Inner folds | 3 (hyperparameter tuning) |
| Features | All 107 stylometric features |
| Preprocessing | None (tree-based; scale-invariant) |
| Scoring | Accuracy (inner CV), macro precision/recall + weighted F1 (reported) |

**Hyperparameter search grid:**

| Parameter | Values |
|-----------|--------|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 3, 5, 7 |
| `learning_rate` | 0.05, 0.1, 0.2 |

Total combinations: 27 per outer fold → 135 inner fits per case.

> **Note:** `GridSearchCV` runs single-threaded (`n_jobs=1`) to avoid joblib's loky multiprocessing backend crashing XGBoost on Windows. XGBoost itself uses all cores via `n_jobs=-1`.

---

## Results Summary

| Case | Acc (mean±std) | Precision (macro) | Recall (macro) | Weighted F1 |
|------|---------------|-------------------|----------------|-------------|
| With outliers    | 0.9447 ± 0.0257 | 0.9474 ± 0.0253 | 0.9440 ± 0.0267 | 0.9444 ± 0.0257 |
| Without outliers | 0.9461 ± 0.0160 | 0.9501 ± 0.0140 | 0.9451 ± 0.0172 | 0.9461 ± 0.0161 |

Best hyperparameters consistently favour `max_depth=3–5` and `learning_rate=0.1–0.2`. Removing outliers tightens variance (std drops from 0.026 to 0.016) without a large accuracy shift.
