# Random Forest — Stylometric Author Classification

Random Forest authorship classifier evaluated with nested cross-validation on stylometric features extracted from LessWrong blog posts. Five target authors: Eliezer Yudkowsky, Johnswentworth, Raemon, Scottalexander, and Zvi.

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
python random_forest_code.py
```

**Outputs:** `results_with_outliers.md`, `results_without_outliers.md`

---

## Files

### `random_forest_code.py`
Main script. Runs nested CV for both dataset variants and writes one Markdown result file per case.

### `results_with_outliers.md`
Nested CV results on the full dataset (723 passages).

### `results_without_outliers.md`
Nested CV results on the outlier-removed dataset (686 passages).

---

## Model Configuration

| Setting | Value |
|---------|-------|
| Classifier | Random Forest (`RandomForestClassifier`) |
| Outer folds | 5 (performance estimation) |
| Inner folds | 3 (hyperparameter tuning) |
| Features | All 107 stylometric features |
| Preprocessing | None (tree-based; scale-invariant) |
| Scoring | Accuracy (inner CV), macro precision/recall + weighted F1 (reported) |

**Hyperparameter search grid:**

| Parameter | Values |
|-----------|--------|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | None, 10, 20 |
| `min_samples_split` | 2, 5 |
| `max_features` | `sqrt`, `log2` |

Total combinations: 36 per outer fold → 180 inner fits per case.

> **Note:** `GridSearchCV` runs single-threaded (`n_jobs=1`) to avoid joblib's loky backend OOM-killing worker processes on Windows. Random Forest itself uses all cores via `n_jobs=-1`.

---

## Results Summary

| Case | Acc (mean±std) | Precision (macro) | Recall (macro) | Weighted F1 |
|------|---------------|-------------------|----------------|-------------|
| With outliers    | 0.9226 ± 0.0341 | 0.9271 ± 0.0313 | 0.9223 ± 0.0350 | 0.9234 ± 0.0338 |
| Without outliers | 0.9345 ± 0.0312 | 0.9398 ± 0.0263 | 0.9339 ± 0.0318 | 0.9349 ± 0.0306 |

Best hyperparameters consistently select `max_depth=None` (fully grown trees), `max_features=log2`, and `min_samples_split=2`. Outlier removal improves mean accuracy by ~1.2% and slightly reduces variance.
