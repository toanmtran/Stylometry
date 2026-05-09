# Logistic Regression Authorship Classification — With Outliers

## Configuration

- **Classifier:** Logistic Regression (multinomial / softmax)
- **Scaler:** StandardScaler
- **Outer folds:** 5 (performance estimation)
- **Inner folds:** 3 (hyperparameter tuning via GridSearchCV)
- **Param combinations:** 15
- **Passages:** 723
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
| 1 | 0.9241 | 0.9250 | 0.9227 | 0.9241 | `C=1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 2 | 0.9379 | 0.9418 | 0.9373 | 0.9392 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 3 | 0.9241 | 0.9283 | 0.9213 | 0.9246 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 4 | 0.9583 | 0.9596 | 0.9577 | 0.9583 | `C=100, max_iter=1000, penalty=l2, solver=lbfgs` |
| 5 | 0.9375 | 0.9407 | 0.9380 | 0.9385 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |

## Summary

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy           | 0.9364  | 0.0140  |
| Precision (macro)  | 0.9391 | 0.0136 |
| Recall (macro)     | 0.9354    | 0.0147    |
| Weighted F1        | 0.9369      | 0.0139      |

## Average Classification Report

_Per-class metrics averaged across all outer folds._

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| Eliezer Yudkowsky |    0.955582 | 0.953333 |   0.953514 |      30   |
| Johnswentworth    |    0.955528 | 0.94     |   0.946708 |      30   |
| Raemon            |    0.921302 | 0.912923 |   0.915966 |      25.2 |
| Scottalexander    |    0.869414 | 0.890805 |   0.877544 |      29.4 |
| Zvi               |    0.993548 | 0.98     |   0.986552 |      30   |
| macro avg         |    0.939075 | 0.935412 |   0.936057 |     144.6 |
| weighted avg      |    0.939851 | 0.936408 |   0.936947 |     144.6 |

## Feature Importance (Coefficient Analysis)

Trained on full dataset with best hyperparameters found via inner CV.

### Top 5 Positive Features Per Author

_Features with highest positive coefficient → strongly associated with this author._

**Eliezer Yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | 0.4794 |
| `punct_semicolon_rate` | 0.4421 |
| `punct_dquote_rate` | 0.3247 |
| `fw_his` | 0.3024 |
| `pos_prep` | 0.2694 |

**Johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `fw_which` | 0.5081 |
| `punct_dash_rate` | 0.4245 |
| `brunet_w` | 0.3649 |
| `pos_adv` | 0.3179 |
| `pos_det` | 0.1993 |

**Raemon:**

| Feature | Coefficient |
|---------|-------------|
| `fw_but` | 0.3768 |
| `fw_to` | 0.3020 |
| `punct_apost_rate` | 0.2861 |
| `pronoun_start_rate` | 0.2804 |
| `contraction_rate` | 0.2757 |

**Scottalexander:**

| Feature | Coefficient |
|---------|-------------|
| `cat_amplifier_rate` | 0.5063 |
| `fw_one` | 0.3019 |
| `char_e_rate` | 0.2784 |
| `punct_semicolon_rate` | 0.2637 |
| `fw_but` | 0.2607 |

**Zvi:**

| Feature | Coefficient |
|---------|-------------|
| `fw_not` | 0.3501 |
| `fw_this` | 0.3117 |
| `fw_get` | 0.2434 |
| `pos_adv` | 0.2424 |
| `fw_for` | 0.2349 |

### Top 5 Negative Features Per Author

_Features with most negative coefficient → strongly disassociated from this author._

**Eliezer Yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `fw_but` | -0.3444 |
| `fw_for` | -0.2865 |
| `fw_this` | -0.2594 |
| `punct_paren_rate` | -0.2457 |
| `fw_we` | -0.2310 |

**Johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `honore_r` | -0.4337 |
| `fw_this` | -0.2835 |
| `fw_you` | -0.2458 |
| `char_i_rate` | -0.2347 |
| `pronoun_start_rate` | -0.2308 |

**Raemon:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | -0.4738 |
| `fw_so` | -0.3381 |
| `fw_would` | -0.3263 |
| `fw_he` | -0.3122 |
| `cat_amplifier_rate` | -0.3008 |

**Scottalexander:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | -0.4780 |
| `fw_not` | -0.4112 |
| `fw_what` | -0.3651 |
| `fw_to` | -0.3108 |
| `pos_adv` | -0.2989 |

**Zvi:**

| Feature | Coefficient |
|---------|-------------|
| `fw_but` | -0.2919 |
| `punct_apost_rate` | -0.2817 |
| `contraction_rate` | -0.2816 |
| `fw_of` | -0.2736 |
| `median_sent_len` | -0.2687 |

### Most Discriminative Features Overall

_Features with highest absolute coefficient range across authors._

| Feature | Max Coef | Min Coef | Range |
|---------|----------|----------|-------|
| `fw_that` | 0.4794 | -0.4780 | 0.9575 |
| `punct_semicolon_rate` | 0.4421 | -0.4738 | 0.9159 |
| `cat_amplifier_rate` | 0.5063 | -0.3008 | 0.8071 |
| `fw_not` | 0.3501 | -0.4112 | 0.7613 |
| `fw_but` | 0.3768 | -0.3444 | 0.7212 |
| `fw_which` | 0.5081 | -0.1978 | 0.7058 |
| `punct_dash_rate` | 0.4245 | -0.2618 | 0.6863 |
| `honore_r` | 0.1972 | -0.4337 | 0.6309 |
| `pos_adv` | 0.3179 | -0.2989 | 0.6168 |
| `fw_to` | 0.3020 | -0.3108 | 0.6128 |
