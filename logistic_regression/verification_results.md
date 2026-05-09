# Authorship Verification Results

## Summary Table

| Case | Model | Passages | Authors | CV Acc | Test Acc | Precision | Recall | F1 | AUC |
|------|-------|----------|---------|--------|----------|-----------|--------|----|-----|
| 5 Authors | lr | 723 | 5 | 0.5511+/-0.0997 | 0.9600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5 Authors (no outliers) | lr | 686 | 5 | 0.4987+/-0.0177 | 0.9660 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 35 Authors | lr | 1484 | 35 | 0.7181+/-0.0327 | 0.7002 | 0.6783 | 0.7614 | 0.7175 | 0.7748 |

## Detailed Results

### 5 Authors — lr

- **Passages:** 723
- **Authors:** 5
- **CV Accuracy:** 0.5511 +/- 0.0997
- **Test Accuracy:** 0.9600
- **Precision:** 0.0000
- **Recall:** 0.0000
- **F1:** 0.0000
- **AUC:** 0.0000

**Confusion Matrix:**

| | Predicted Diff | Predicted Same |
|---|---:|---:|
| Actual Different | 0 | 0 |
| Actual Same | 20 | 480 |

### 5 Authors (no outliers) — lr

- **Passages:** 686
- **Authors:** 5
- **CV Accuracy:** 0.4987 +/- 0.0177
- **Test Accuracy:** 0.9660
- **Precision:** 0.0000
- **Recall:** 0.0000
- **F1:** 0.0000
- **AUC:** 0.0000

**Confusion Matrix:**

| | Predicted Diff | Predicted Same |
|---|---:|---:|
| Actual Different | 0 | 0 |
| Actual Same | 17 | 483 |

### 35 Authors — lr

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

