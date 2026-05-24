# Logistic Regression Authorship Classification — Without Outliers

## Configuration

- **Classifier:** Logistic Regression (multinomial / softmax)
- **Scaler:** StandardScaler
- **Outer folds:** 5 (performance estimation)
- **Inner folds:** 3 (hyperparameter tuning via GridSearchCV)
- **Param combinations:** 15
- **Passages:** 783
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
| 1 | 0.7261 | 0.7277 | 0.7207 | 0.7191 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 2 | 0.7707 | 0.7866 | 0.7664 | 0.7697 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 3 | 0.7134 | 0.7379 | 0.7119 | 0.7129 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 4 | 0.7821 | 0.7947 | 0.7881 | 0.7785 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 5 | 0.7628 | 0.7829 | 0.7709 | 0.7543 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |

## Summary

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy           | 0.7510  | 0.0297  |
| Precision (macro)  | 0.7660 | 0.0308 |
| Recall (macro)     | 0.7516    | 0.0334    |
| Weighted F1        | 0.7469      | 0.0296      |

## Average Classification Report

_Per-class metrics averaged across all outer folds._

|                      |   precision |   recall |   f1-score |   support |
|:---------------------|------------:|---------:|-----------:|----------:|
| abramdemski          |    0.901984 | 0.909524 |   0.894767 |       6.6 |
| adamshimi            |    0.709091 | 0.780952 |   0.733333 |       6.8 |
| benquo               |    0.610952 | 0.633333 |   0.620346 |       6.6 |
| buck                 |    0.842857 | 0.8      |   0.805556 |       4.8 |
| david-gross          |    0.829762 | 0.857143 |   0.841172 |       7   |
| eliezer_yudkowsky    |    0.635714 | 0.635714 |   0.634524 |       7.6 |
| elizabeth-1          |    0.67     | 0.62     |   0.636667 |       4.4 |
| gordon-seidoh-worley |    0.81119  | 0.675    |   0.732143 |       7.4 |
| holdenkarnofsky      |    0.833333 | 0.74     |   0.763636 |       4.8 |
| jacob-falkovich      |    0.734848 | 0.835714 |   0.773992 |       7.4 |
| jasoncrawford        |    0.661667 | 0.660714 |   0.653539 |       7.6 |
| joe-carlsmith        |    0.867857 | 0.933333 |   0.891342 |       5.6 |
| johnswentworth       |    0.893333 | 0.866667 |   0.869091 |       6   |
| kaj_sotala           |    0.59     | 0.514286 |   0.510222 |       7.6 |
| katjagrace           |    0.750397 | 0.764286 |   0.753333 |       7.6 |
| petermccluskey       |    1        | 1        |   1        |       5.4 |
| raemon               |    0.813333 | 0.766667 |   0.787879 |       6   |
| ruby                 |    0.485    | 0.453333 |   0.458368 |       5.4 |
| ryan_greenblatt      |    0.884524 | 0.938095 |   0.909377 |       6.6 |
| sarahconstantin      |    0.713968 | 0.652381 |   0.673312 |       6.4 |
| screwtape            |    0.742857 | 0.761905 |   0.74381  |       6.8 |
| steve2152            |    0.782857 | 0.871429 |   0.809981 |       6.4 |
| tsvibt               |    0.653333 | 0.506667 |   0.566667 |       5.2 |
| zack_m_davis         |    0.803333 | 0.693333 |   0.738788 |       5.4 |
| zvi                  |    0.926667 | 0.92     |   0.919596 |       5.2 |
| macro avg            |    0.765954 | 0.751619 |   0.748858 |     156.6 |
| weighted avg         |    0.762656 | 0.751013 |   0.746889 |     156.6 |

## Feature Importance (Coefficient Analysis)

Trained on full dataset with best hyperparameters found via inner CV.

### Top 5 Positive Features Per Author

_Features with highest positive coefficient → strongly associated with this author._

**abramdemski:**

| Feature | Coefficient |
|---------|-------------|
| `cat_discourse_rate` | 0.3847 |
| `fw_which` | 0.3578 |
| `fw_this` | 0.3212 |
| `punct_semicolon_rate` | 0.3104 |
| `fw_we` | 0.2684 |

**adamshimi:**

| Feature | Coefficient |
|---------|-------------|
| `cat_conj_rate` | 0.4133 |
| `fw_what` | 0.3846 |
| `n_tokens` | 0.2822 |
| `avg_paragraph_len` | 0.2822 |
| `fw_the` | 0.2771 |

**benquo:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | 0.3599 |
| `fw_do` | 0.2937 |
| `fw_the` | 0.2511 |
| `punct_comma_rate` | 0.2496 |
| `median_word_len` | 0.2239 |

**buck:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | 0.3033 |
| `char_u_rate` | 0.2423 |
| `fw_this` | 0.2149 |
| `fw_for` | 0.2103 |
| `median_word_len` | 0.1831 |

**david-gross:**

| Feature | Coefficient |
|---------|-------------|
| `fw_or` | 0.4610 |
| `fw_as` | 0.3056 |
| `fw_who` | 0.2923 |
| `n_tokens` | 0.2845 |
| `avg_paragraph_len` | 0.2845 |

**eliezer_yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | 0.5393 |
| `fw_go` | 0.3986 |
| `fw_not` | 0.3427 |
| `punct_colon_rate` | 0.3161 |
| `fw_that` | 0.3040 |

**elizabeth-1:**

| Feature | Coefficient |
|---------|-------------|
| `fw_but` | 0.5480 |
| `cat_amplifier_rate` | 0.3213 |
| `fw_so` | 0.2646 |
| `fw_me` | 0.2433 |
| `fw_which` | 0.2376 |

**gordon-seidoh-worley:**

| Feature | Coefficient |
|---------|-------------|
| `fw_we` | 0.4650 |
| `fw_so` | 0.4395 |
| `cat_discourse_rate` | 0.3458 |
| `cat_conj_rate` | 0.3120 |
| `fw_it` | 0.2859 |

**holdenkarnofsky:**

| Feature | Coefficient |
|---------|-------------|
| `cat_hedge_rate` | 0.5795 |
| `fw_be` | 0.4028 |
| `punct_paren_rate` | 0.3740 |
| `fw_will` | 0.2919 |
| `fw_do` | 0.2608 |

**jacob-falkovich:**

| Feature | Coefficient |
|---------|-------------|
| `fw_who` | 0.4239 |
| `fw_by` | 0.3823 |
| `fw_and` | 0.3030 |
| `fw_her` | 0.2866 |
| `min_sent_len` | 0.2678 |

**jasoncrawford:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | 0.3720 |
| `fw_and` | 0.3584 |
| `honore_r` | 0.3526 |
| `fw_would` | 0.2987 |
| `hapax_ratio` | 0.2689 |

**joe-carlsmith:**

| Feature | Coefficient |
|---------|-------------|
| `fw_in` | 0.4909 |
| `punct_colon_rate` | 0.3428 |
| `punct_dash_rate` | 0.3339 |
| `char_i_rate` | 0.3031 |
| `punct_comma_rate` | 0.2994 |

**johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `fw_which` | 0.5919 |
| `brunet_w` | 0.2391 |
| `fw_all` | 0.2297 |
| `word_len_5_frac` | 0.2166 |
| `fw_the` | 0.1881 |

**kaj_sotala:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | 0.4556 |
| `fw_as` | 0.3773 |
| `cat_discourse_rate` | 0.3293 |
| `fw_it` | 0.3243 |
| `fw_me` | 0.2893 |

**katjagrace:**

| Feature | Coefficient |
|---------|-------------|
| `punct_apost_rate` | 0.4615 |
| `word_len_2_frac` | 0.3438 |
| `fw_so` | 0.3185 |
| `fw_there` | 0.3157 |
| `fw_would` | 0.2828 |

**petermccluskey:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | 0.4274 |
| `fw_will` | 0.2198 |
| `char_a_rate` | 0.1592 |
| `fw_to` | 0.1447 |
| `fw_as` | 0.1435 |

**raemon:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | 0.4818 |
| `fw_get` | 0.3593 |
| `fw_but` | 0.3087 |
| `fw_about` | 0.2987 |
| `punct_period_rate` | 0.2785 |

**ruby:**

| Feature | Coefficient |
|---------|-------------|
| `fw_we` | 0.2750 |
| `punct_period_rate` | 0.2421 |
| `fw_and` | 0.1984 |
| `cat_conj_rate` | 0.1974 |
| `fw_my` | 0.1912 |

**ryan_greenblatt:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | 0.3954 |
| `punct_paren_rate` | 0.3468 |
| `fw_on` | 0.2805 |
| `brunet_w` | 0.2783 |
| `cat_hedge_rate` | 0.2423 |

**sarahconstantin:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | 0.4699 |
| `punct_comma_rate` | 0.3453 |
| `punct_semicolon_rate` | 0.3436 |
| `fw_or` | 0.3426 |
| `fw_in` | 0.2563 |

**screwtape:**

| Feature | Coefficient |
|---------|-------------|
| `fw_a` | 0.3267 |
| `fw_say` | 0.3224 |
| `fw_if` | 0.2493 |
| `fw_i` | 0.2365 |
| `fw_or` | 0.2354 |

**steve2152:**

| Feature | Coefficient |
|---------|-------------|
| `punct_excl_rate` | 0.5430 |
| `punct_paren_rate` | 0.4256 |
| `punct_dash_rate` | 0.3928 |
| `punct_dquote_rate` | 0.3811 |
| `punct_comma_rate` | 0.3255 |

**tsvibt:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | 0.5353 |
| `punct_comma_rate` | 0.4790 |
| `cat_amplifier_rate` | 0.3016 |
| `punct_period_rate` | 0.2276 |
| `fw_up` | 0.2215 |

**zack_m_davis:**

| Feature | Coefficient |
|---------|-------------|
| `std_sent_len` | 0.3571 |
| `punct_excl_rate` | 0.3232 |
| `fw_would` | 0.3003 |
| `fw_the` | 0.2928 |
| `avg_sent_len` | 0.2799 |

**zvi:**

| Feature | Coefficient |
|---------|-------------|
| `uppercase_ratio` | 0.4963 |
| `n_sentences` | 0.3195 |
| `punct_period_rate` | 0.3043 |
| `fw_not` | 0.2398 |
| `fw_all` | 0.2359 |

### Top 5 Negative Features Per Author

_Features with most negative coefficient → strongly disassociated from this author._

**abramdemski:**

| Feature | Coefficient |
|---------|-------------|
| `fw_and` | -0.5430 |
| `cat_conj_rate` | -0.4003 |
| `fw_that` | -0.3099 |
| `std_sent_len` | -0.2119 |
| `max_sent_len` | -0.1937 |

**adamshimi:**

| Feature | Coefficient |
|---------|-------------|
| `punct_period_rate` | -0.3622 |
| `fw_to` | -0.3485 |
| `punct_dquote_rate` | -0.3437 |
| `cat_hedge_rate` | -0.2471 |
| `fw_who` | -0.2094 |

**benquo:**

| Feature | Coefficient |
|---------|-------------|
| `punct_paren_rate` | -0.3709 |
| `punct_dquote_rate` | -0.3678 |
| `fw_will` | -0.2700 |
| `fw_we` | -0.2360 |
| `fw_go` | -0.2328 |

**buck:**

| Feature | Coefficient |
|---------|-------------|
| `fw_what` | -0.3269 |
| `char_i_rate` | -0.3167 |
| `fw_at` | -0.2722 |
| `punct_dquote_rate` | -0.2571 |
| `fw_not` | -0.2536 |

**david-gross:**

| Feature | Coefficient |
|---------|-------------|
| `uppercase_ratio` | -0.2971 |
| `word_len_5_frac` | -0.2736 |
| `cat_amplifier_rate` | -0.2501 |
| `fw_on` | -0.2379 |
| `punct_apost_rate` | -0.2294 |

**eliezer_yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `punct_paren_rate` | -0.5724 |
| `fw_for` | -0.4644 |
| `cat_conj_rate` | -0.3295 |
| `char_i_rate` | -0.2650 |
| `fw_do` | -0.2200 |

**elizabeth-1:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | -0.3245 |
| `fw_be` | -0.2666 |
| `punct_comma_rate` | -0.2479 |
| `n_tokens` | -0.2078 |
| `avg_paragraph_len` | -0.2078 |

**gordon-seidoh-worley:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dash_rate` | -0.4058 |
| `fw_who` | -0.3432 |
| `punct_paren_rate` | -0.3254 |
| `fw_be` | -0.3152 |
| `hapax_dis_ratio` | -0.3090 |

**holdenkarnofsky:**

| Feature | Coefficient |
|---------|-------------|
| `fw_you` | -0.2424 |
| `fw_in` | -0.2344 |
| `punct_comma_rate` | -0.2272 |
| `fw_but` | -0.1918 |
| `cat_discourse_rate` | -0.1865 |

**jacob-falkovich:**

| Feature | Coefficient |
|---------|-------------|
| `std_sent_len` | -0.3794 |
| `cat_discourse_rate` | -0.3602 |
| `punct_paren_rate` | -0.3319 |
| `punct_semicolon_rate` | -0.3096 |
| `cat_hedge_rate` | -0.3077 |

**jasoncrawford:**

| Feature | Coefficient |
|---------|-------------|
| `punct_apost_rate` | -0.4683 |
| `contraction_rate` | -0.3565 |
| `fw_about` | -0.3071 |
| `fw_with` | -0.3016 |
| `fw_i` | -0.2801 |

**joe-carlsmith:**

| Feature | Coefficient |
|---------|-------------|
| `fw_there` | -0.2151 |
| `fw_which` | -0.2091 |
| `punct_excl_rate` | -0.2003 |
| `fw_a` | -0.1977 |
| `fw_all` | -0.1969 |

**johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | -0.3713 |
| `word_len_2_frac` | -0.2806 |
| `fw_you` | -0.2696 |
| `honore_r` | -0.2559 |
| `hapax_ratio` | -0.2484 |

**kaj_sotala:**

| Feature | Coefficient |
|---------|-------------|
| `fw_on` | -0.4377 |
| `fw_we` | -0.2954 |
| `word_len_2_frac` | -0.2837 |
| `punct_paren_rate` | -0.2780 |
| `fw_who` | -0.2586 |

**katjagrace:**

| Feature | Coefficient |
|---------|-------------|
| `contraction_rate` | -0.4936 |
| `punct_dquote_rate` | -0.3141 |
| `min_sent_len` | -0.2960 |
| `fw_but` | -0.2919 |
| `punct_comma_rate` | -0.2594 |

**petermccluskey:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | -0.3770 |
| `fw_and` | -0.2880 |
| `cat_conj_rate` | -0.2806 |
| `cat_amplifier_rate` | -0.2350 |
| `std_sent_len` | -0.2141 |

**raemon:**

| Feature | Coefficient |
|---------|-------------|
| `fw_in` | -0.3260 |
| `fw_from` | -0.2840 |
| `char_a_rate` | -0.2784 |
| `punct_semicolon_rate` | -0.2534 |
| `fw_be` | -0.2140 |

**ruby:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | -0.3921 |
| `fw_an` | -0.3741 |
| `max_word_len` | -0.3411 |
| `punct_dquote_rate` | -0.2907 |
| `fw_by` | -0.2744 |

**ryan_greenblatt:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | -0.3102 |
| `punct_dquote_rate` | -0.2763 |
| `fw_not` | -0.2703 |
| `n_vocab` | -0.2604 |
| `hapax_ratio` | -0.2493 |

**sarahconstantin:**

| Feature | Coefficient |
|---------|-------------|
| `punct_colon_rate` | -0.3535 |
| `fw_as` | -0.2837 |
| `punct_period_rate` | -0.2624 |
| `fw_if` | -0.2582 |
| `fw_would` | -0.2489 |

**screwtape:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dash_rate` | -0.3468 |
| `std_word_len` | -0.3412 |
| `punct_colon_rate` | -0.3105 |
| `avg_word_len` | -0.3096 |
| `fw_we` | -0.2814 |

**steve2152:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | -0.2697 |
| `fw_it` | -0.2570 |
| `min_sent_len` | -0.2075 |
| `fw_say` | -0.1948 |
| `honore_r` | -0.1901 |

**tsvibt:**

| Feature | Coefficient |
|---------|-------------|
| `fw_i` | -0.2827 |
| `punct_excl_rate` | -0.2751 |
| `fw_which` | -0.2637 |
| `hapax_dis_ratio` | -0.2467 |
| `fw_this` | -0.2423 |

**zack_m_davis:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | -0.5388 |
| `fw_will` | -0.4246 |
| `word_len_4_frac` | -0.2403 |
| `fw_so` | -0.2324 |
| `cat_conj_rate` | -0.2206 |

**zvi:**

| Feature | Coefficient |
|---------|-------------|
| `median_sent_len` | -0.3221 |
| `punct_dquote_rate` | -0.2485 |
| `fw_by` | -0.2129 |
| `avg_sent_len` | -0.1965 |
| `fw_of` | -0.1850 |

### Most Discriminative Features Overall

_Features with highest absolute coefficient range across authors._

| Feature | Max Coef | Min Coef | Range |
|---------|----------|----------|-------|
| `punct_paren_rate` | 0.4256 | -0.5724 | 0.9980 |
| `fw_this` | 0.3954 | -0.5388 | 0.9343 |
| `punct_semicolon_rate` | 0.5393 | -0.3921 | 0.9314 |
| `punct_apost_rate` | 0.4615 | -0.4683 | 0.9298 |
| `fw_and` | 0.3584 | -0.5430 | 0.9014 |
| `cat_hedge_rate` | 0.5795 | -0.3077 | 0.8872 |
| `fw_which` | 0.5919 | -0.2673 | 0.8592 |
| `punct_comma_rate` | 0.4790 | -0.3770 | 0.8560 |
| `punct_dquote_rate` | 0.4818 | -0.3678 | 0.8496 |
| `fw_but` | 0.5480 | -0.2919 | 0.8400 |
