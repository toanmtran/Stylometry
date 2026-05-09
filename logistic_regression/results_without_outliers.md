# Logistic Regression Authorship Classification — Without Outliers

## Configuration

- **Classifier:** Logistic Regression (multinomial / softmax)
- **Scaler:** StandardScaler
- **Outer folds:** 5 (performance estimation)
- **Inner folds:** 3 (hyperparameter tuning via GridSearchCV)
- **Param combinations:** 15
- **Passages:** 686
- **Features:** all 107

**Search grid:**

| Hyperparameter | Values |
|----------------|--------|
| `C` | [0.01, 0.1, 1, 10, 100] |
| `penalty` | ['l2'] |
| `solver` | ['lbfgs'] |
| `max_iter` | [1000, 2000, 5000] |

## Per-Fold Results

| Fold | Accuracy | Precision (macro) | Recall (macro) | Weighted F1 | Best Params |
|------|----------|-------------------|----------------|-------------|-------------|
| 1 | 0.9275 | 0.9303 | 0.9239 | 0.9263 | `C=10, max_iter=1000, penalty=l2, solver=lbfgs` |
| 2 | 0.9635 | 0.9651 | 0.9619 | 0.9631 | `C=1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 3 | 0.9708 | 0.9695 | 0.9693 | 0.9709 | `C=1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 4 | 0.9635 | 0.9645 | 0.9631 | 0.9635 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 5 | 0.9635 | 0.9626 | 0.9640 | 0.9637 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |

## Summary

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy           | 0.9578  | 0.0172  |
| Precision (macro)  | 0.9584 | 0.0159 |
| Recall (macro)     | 0.9564    | 0.0184    |
| Weighted F1        | 0.9575      | 0.0178      |

## Average Classification Report

_Per-class metrics averaged across all outer folds._

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.959029 | 0.978818 |   0.968516 |      28.2 |
| Johnswentworth    |    0.968347 | 0.97977  |   0.973092 |      29.4 |
| Raemon            |    0.941561 | 0.908333 |   0.923039 |      24   |
| Scottalexander    |    0.936928 | 0.915271 |   0.925372 |      28.2 |
| Zvi               |    0.986207 | 1        |   0.992857 |      27.4 |
| macro avg         |    0.958414 | 0.956438 |   0.956575 |     137.2 |
| weighted avg      |    0.958885 | 0.95777  |   0.957509 |     137.2 |

## Feature Importance (Coefficient Analysis)

Trained on full dataset with best hyperparameters found via inner CV.

### Top 5 Positive Features Per Author

_Features with highest positive coefficient → strongly associated with this author._

**Eliezer Yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | 0.4479 |
| `punct_semicolon_rate` | 0.3222 |
| `punct_dquote_rate` | 0.2913 |
| `fw_his` | 0.2731 |
| `punct_excl_rate` | 0.2319 |

**Johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `fw_which` | 0.4581 |
| `punct_dash_rate` | 0.4269 |
| `brunet_w` | 0.3442 |
| `pos_adv` | 0.3140 |
| `fw_we` | 0.1913 |

**Raemon:**

| Feature | Coefficient |
|---------|-------------|
| `fw_but` | 0.3383 |
| `fw_to` | 0.3157 |
| `pronoun_start_rate` | 0.2693 |
| `punct_apost_rate` | 0.2564 |
| `fw_about` | 0.2451 |

**Scottalexander:**

| Feature | Coefficient |
|---------|-------------|
| `cat_amplifier_rate` | 0.4837 |
| `fw_one` | 0.3078 |
| `char_e_rate` | 0.2962 |
| `punct_semicolon_rate` | 0.2639 |
| `fw_by` | 0.2332 |

**Zvi:**

| Feature | Coefficient |
|---------|-------------|
| `fw_not` | 0.3858 |
| `fw_this` | 0.2830 |
| `n_vocab` | 0.2333 |
| `pos_adv` | 0.2220 |
| `uppercase_ratio` | 0.2060 |

### Top 5 Negative Features Per Author

_Features with most negative coefficient → strongly disassociated from this author._

**Eliezer Yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | -0.3078 |
| `fw_but` | -0.3038 |
| `min_sent_len` | -0.2095 |
| `fw_we` | -0.2063 |
| `cat_conj_rate` | -0.2062 |

**Johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `honore_r` | -0.4389 |
| `fw_you` | -0.2710 |
| `pronoun_start_rate` | -0.2258 |
| `fw_this` | -0.2209 |
| `std_sent_len` | -0.2102 |

**Raemon:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | -0.4302 |
| `fw_so` | -0.3224 |
| `cat_amplifier_rate` | -0.3094 |
| `fw_would` | -0.2929 |
| `fw_say` | -0.2531 |

**Scottalexander:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | -0.5004 |
| `fw_not` | -0.3606 |
| `fw_what` | -0.3160 |
| `pos_adv` | -0.2709 |
| `fw_to` | -0.2474 |

**Zvi:**

| Feature | Coefficient |
|---------|-------------|
| `fw_of` | -0.2832 |
| `punct_semicolon_rate` | -0.2703 |
| `punct_dash_rate` | -0.2455 |
| `contraction_rate` | -0.2400 |
| `punct_apost_rate` | -0.2387 |

### Most Discriminative Features Overall

_Features with highest absolute coefficient range across authors._

| Feature | Max Coef | Min Coef | Range |
|---------|----------|----------|-------|
| `fw_that` | 0.4479 | -0.5004 | 0.9483 |
| `cat_amplifier_rate` | 0.4837 | -0.3094 | 0.7930 |
| `punct_semicolon_rate` | 0.3222 | -0.4302 | 0.7524 |
| `fw_not` | 0.3858 | -0.3606 | 0.7463 |
| `punct_dash_rate` | 0.4269 | -0.2455 | 0.6724 |
| `fw_which` | 0.4581 | -0.1999 | 0.6580 |
| `fw_but` | 0.3383 | -0.3038 | 0.6421 |
| `honore_r` | 0.1895 | -0.4389 | 0.6284 |
| `fw_this` | 0.2830 | -0.3078 | 0.5908 |
| `pos_adv` | 0.3140 | -0.2709 | 0.5849 |
