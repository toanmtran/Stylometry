# Authorship Verification Results

## Summary Table

| Case | Model | Passages | Authors | CV Acc | Test Acc | Precision | Recall | F1 | AUC |
|------|-------|----------|---------|--------|----------|-----------|--------|----|-----|
| 35 Authors | Logistic Regression | 1484 | 35 | 0.7181+/-0.0327 | 0.7002 | 0.6783 | 0.7614 | 0.7175 | 0.7748 |

## Detailed Results

### 35 Authors — Logistic Regression

- **Passages:** 1484
- **Authors:** 35
- **CV Accuracy:** 0.7181 +/- 0.0327
- **Test Accuracy:** 0.7002
- **Precision:** 0.6783
- **Recall:** 0.7614
- **F1:** 0.7175
- **AUC:** 0.7748

**Confusion Matrix:**

| | Predicted Diff | Predicted Same |
|---|---:|---:|
| Actual Different | 2081 | 1176 |
| Actual Same | 777 | 2480 |

