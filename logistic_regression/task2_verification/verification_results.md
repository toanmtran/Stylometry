# Authorship Verification Results

## Summary Table

| Case | Model | Passages | Authors | Best Params | CV Acc | Test Acc | Precision | Recall | F1 | AUC |
|------|-------|----------|---------|-------------|--------|----------|-----------|--------|----|-----|
| 35 Authors | Logistic Regression | 1484 | 35 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` | 0.7812+/-0.0012 | 0.6999 | 0.6786 | 0.7593 | 0.7167 | 0.7759 |

## Detailed Results

### 35 Authors — Logistic Regression

- **Passages:** 1484
- **Authors:** 35
- **Best Params:** `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs`
- **CV Accuracy:** 0.7812 +/- 0.0012
- **Test Accuracy:** 0.6999
- **Precision:** 0.6786
- **Recall:** 0.7593
- **F1:** 0.7167
- **AUC:** 0.7759

**Confusion Matrix:**

| | Predicted Diff | Predicted Same |
|---|---:|---:|
| Actual Different | 2086 | 1171 |
| Actual Same | 784 | 2473 |

