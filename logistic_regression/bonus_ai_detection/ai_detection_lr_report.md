# AI Detection Report — Logistic Regression

## Best Params

```
{'C': 0.01, 'class_weight': 'balanced', 'max_iter': 1000, 'penalty': 'l2', 'solver': 'lbfgs'}
```

## Results on Test Set (20%)

- **Accuracy:** 0.6567
- **AUC-ROC:** 0.7420

### Confusion Matrix

| | Pred HUMAN | Pred AI |
|---|---|---|
| Actual HUMAN | 44 | 23 |
| Actual AI | 23 | 44 |

### Cross-Validation (5-fold, on Train)

- **Mean:** 0.6711
- **Std:** 0.0444

### Top Features -> HUMAN (<=2022)

| Feature | Coefficient |
|---|---|
| fw_this | -0.1785 |
| fw_which | -0.1519 |
| fw_the | -0.1407 |
| fw_of | -0.1278 |
| punct_excl_rate | -0.1256 |
| cat_hedge_rate | -0.1143 |
| fw_there | -0.1110 |
| char_o_rate | -0.1048 |
| word_len_4_frac | -0.1027 |
| cat_discourse_rate | -0.1022 |

### Top Features -> AI (>=2024)

| Feature | Coefficient |
|---|---|
| uppercase_ratio | 0.1769 |
| pronoun_start_rate | 0.1503 |
| std_word_len | 0.1343 |
| punct_semicolon_rate | 0.1067 |
| fw_so | 0.1066 |
| char_a_rate | 0.0987 |
| fw_with | 0.0941 |
| median_word_len | 0.0936 |
| fw_me | 0.0646 |
| fw_that | 0.0642 |
