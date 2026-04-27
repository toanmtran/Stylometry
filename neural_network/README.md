# Neural Network — Stylometric Author Classification

This directory contains feature datasets and evaluation results for an MLP-based authorship classifier trained on stylometric features extracted from LessWrong blog posts. The target authors are Eliezer Yudkowsky, Johnswentworth, Raemon, Scottalexander, and Zvi.

---

## Files

### `author_features_extracted_full.csv`
Full feature matrix containing 107 stylometric features per passage, including all authors and all passages (with outliers). Each row represents one passage identified by `author` and `passage_id`. Feature categories cover lexical richness, word-length distributions, sentence statistics, function-word frequencies, punctuation rates, POS-tag proportions, and character-level rates. This is the primary input used for model training and evaluation.

### `feature_extracted_without_outliers.csv`
Same structure as `author_features_extracted_full.csv` but with statistical outlier passages removed per author. Outliers were identified and filtered to reduce noise before training. Used to assess whether removing extreme passages improves generalisation.

### `evaluation_results_full_and_subsets (1).md`
5-fold cross-validation results for the MLP trained on the **full dataset** (with outliers). Evaluates five feature-count subsets — top 15, 30, 50, 74, and all 107 features — crossed with five network depths: depth 1 (64 units), depth 2 (64→32), depth 3 (64→64→64), depth 10, and depth 50. Reports mean training/testing accuracy and per-author precision, recall, and F1-score. Key finding: top-74-features + depth-1 achieves the best test accuracy (~95.2%).

### `Results_without_outlier.md`
Same experimental grid as above but trained and evaluated on the **outlier-removed dataset**. Generally shows higher test accuracy and lower overfitting at shallow depths. Best result: top-74-features + depth-1 reaches ~96.7% test accuracy, the highest across all experiments.

### `with_early_stopping.md`
Evaluation of the MLP on the **outlier-removed dataset with early stopping** enabled during training. Included to examine whether early stopping helps deeper networks (depth 10, 50) recover from vanishing-gradient / dead-neuron collapse. Depth-50 networks still collapse to near-random accuracy (~21%), confirming that architecture choice (shallow networks) matters more than early stopping for this task.

---

## Summary of Best Results

| Dataset          | Features | Depth    | Test Accuracy |
|------------------|----------|----------|---------------|
| Full             | Top 74   | 1 (64,)  | 95.16%        |
| Without outliers | Top 74   | 1 (64,)  | **96.65%**    |
| With early stop  | Top 74   | 1 (64,)  | ~96.5%        |

Shallow networks (depth 1–2) consistently outperform deeper ones. Depth ≥ 10 degrades accuracy; depth 50 collapses entirely to predicting a single class.
