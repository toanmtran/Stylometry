# Logistic Regression Authorship Classification — Without Outliers

## Configuration

- **Classifier:** Logistic Regression (multinomial / softmax)
- **Scaler:** StandardScaler (re-fit per outer fold — no leakage)
- **Outlier removal:** per-fold IsolationForest (contamination=0.05)
- **Outer folds:** 5 (performance estimation)
- **Inner folds:** 3 (hyperparameter tuning via GridSearchCV)
- **Param combinations:** 15
- **Passages:** 868
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
| 1 | 0.6839 | 0.6896 | 0.6820 | 0.6775 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 2 | 0.7471 | 0.7757 | 0.7529 | 0.7408 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 3 | 0.7529 | 0.7647 | 0.7580 | 0.7430 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 4 | 0.6879 | 0.7082 | 0.6927 | 0.6794 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 5 | 0.6647 | 0.6972 | 0.6630 | 0.6588 | `C=1, max_iter=1000, penalty=l2, solver=lbfgs` |

## Summary

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy           | 0.7073  | 0.0400  |
| Precision (macro)  | 0.7271 | 0.0401 |
| Recall (macro)     | 0.7097    | 0.0432    |
| Weighted F1        | 0.6999      | 0.0392      |

## Average Classification Report

_Per-class metrics averaged across all outer folds._

|                      |   precision |   recall |   f1-score |   support |
|:---------------------|------------:|---------:|-----------:|----------:|
| abramdemski          |    0.941667 | 0.864286 |   0.896557 |       7.4 |
| adamshimi            |    0.727698 | 0.789286 |   0.753175 |       7.6 |
| benquo               |    0.439603 | 0.525    |   0.463007 |       8   |
| buck                 |    0.789524 | 0.846667 |   0.815664 |       5.2 |
| david-gross          |    0.791162 | 0.792857 |   0.783778 |       7.8 |
| eliezer_yudkowsky    |    0.577576 | 0.461111 |   0.502456 |       8.2 |
| elizabeth-1          |    0.707143 | 0.786667 |   0.739261 |       5.8 |
| gordon-seidoh-worley |    0.716558 | 0.65     |   0.662368 |       8   |
| holdenkarnofsky      |    0.866667 | 0.606667 |   0.671111 |       5.6 |
| jacob-falkovich      |    0.644372 | 0.764286 |   0.693683 |       7.6 |
| jasoncrawford        |    0.712857 | 0.661111 |   0.673974 |       8.2 |
| joe-carlsmith        |    0.761905 | 0.819048 |   0.777216 |       6.4 |
| johnswentworth       |    0.95     | 0.857143 |   0.889377 |       6.8 |
| kaj_sotala           |    0.471905 | 0.508333 |   0.476249 |       8.2 |
| katjagrace           |    0.738413 | 0.730556 |   0.728478 |       8.2 |
| petermccluskey       |    0.942857 | 0.971429 |   0.953846 |       6.2 |
| raemon               |    0.672619 | 0.666667 |   0.661086 |       6   |
| ruby                 |    0.433333 | 0.346667 |   0.377172 |       5.8 |
| ryan_greenblatt      |    0.843651 | 0.889286 |   0.863314 |       7.2 |
| sarahconstantin      |    0.681746 | 0.714286 |   0.693242 |       7   |
| screwtape            |    0.622222 | 0.696429 |   0.640267 |       7.4 |
| steve2152            |    0.729524 | 0.795238 |   0.753846 |       6.8 |
| tsvibt               |    0.758333 | 0.6      |   0.626984 |       6.2 |
| zack_m_davis         |    0.713333 | 0.566667 |   0.627879 |       6   |
| zvi                  |    0.942857 | 0.833333 |   0.851049 |       6   |
| macro avg            |    0.727101 | 0.709721 |   0.703002 |     173.6 |
| weighted avg         |    0.722335 | 0.707302 |   0.6999   |     173.6 |

## Feature Importance (Coefficient Analysis)

Trained on full dataset with best hyperparameters found via inner CV.

### Top 5 Positive Features Per Author

_Features with highest positive coefficient — strongly associated with this author._

**abramdemski:**

| Feature | Coefficient |
|---------|-------------|
| `cat_discourse_rate` | 0.3932 |
| `fw_which` | 0.3892 |
| `fw_this` | 0.3223 |
| `punct_semicolon_rate` | 0.2939 |
| `fw_a` | 0.2878 |

**adamshimi:**

| Feature | Coefficient |
|---------|-------------|
| `fw_what` | 0.4383 |
| `cat_conj_rate` | 0.4058 |
| `fw_one` | 0.2925 |
| `n_tokens` | 0.2867 |
| `avg_paragraph_len` | 0.2867 |

**benquo:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | 0.3258 |
| `fw_do` | 0.3031 |
| `fw_the` | 0.2964 |
| `fw_if` | 0.2838 |
| `fw_who` | 0.2594 |

**buck:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | 0.2909 |
| `char_u_rate` | 0.2503 |
| `fw_for` | 0.2246 |
| `fw_this` | 0.2196 |
| `median_word_len` | 0.1976 |

**david-gross:**

| Feature | Coefficient |
|---------|-------------|
| `fw_or` | 0.4543 |
| `fw_as` | 0.3676 |
| `fw_there` | 0.3112 |
| `fw_who` | 0.2888 |
| `char_u_rate` | 0.2873 |

**eliezer_yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | 0.4995 |
| `fw_his` | 0.3704 |
| `fw_not` | 0.3473 |
| `fw_go` | 0.3456 |
| `punct_colon_rate` | 0.3305 |

**elizabeth-1:**

| Feature | Coefficient |
|---------|-------------|
| `fw_but` | 0.6670 |
| `cat_amplifier_rate` | 0.3427 |
| `cat_conj_rate` | 0.2902 |
| `char_u_rate` | 0.2638 |
| `fw_me` | 0.2598 |

**gordon-seidoh-worley:**

| Feature | Coefficient |
|---------|-------------|
| `fw_we` | 0.4736 |
| `fw_so` | 0.4413 |
| `cat_discourse_rate` | 0.3592 |
| `cat_conj_rate` | 0.2975 |
| `fw_what` | 0.2680 |

**holdenkarnofsky:**

| Feature | Coefficient |
|---------|-------------|
| `cat_hedge_rate` | 0.6302 |
| `punct_paren_rate` | 0.3895 |
| `fw_be` | 0.3277 |
| `fw_will` | 0.2925 |
| `fw_this` | 0.2671 |

**jacob-falkovich:**

| Feature | Coefficient |
|---------|-------------|
| `fw_who` | 0.4210 |
| `fw_by` | 0.3407 |
| `fw_and` | 0.3058 |
| `fw_her` | 0.2887 |
| `min_sent_len` | 0.2764 |

**jasoncrawford:**

| Feature | Coefficient |
|---------|-------------|
| `fw_and` | 0.3909 |
| `honore_r` | 0.3670 |
| `punct_comma_rate` | 0.3351 |
| `fw_would` | 0.3114 |
| `hapax_ratio` | 0.2774 |

**joe-carlsmith:**

| Feature | Coefficient |
|---------|-------------|
| `fw_in` | 0.4915 |
| `punct_comma_rate` | 0.4084 |
| `punct_colon_rate` | 0.3440 |
| `punct_dash_rate` | 0.3314 |
| `char_i_rate` | 0.3195 |

**johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `fw_which` | 0.6242 |
| `brunet_w` | 0.2530 |
| `word_len_5_frac` | 0.2178 |
| `fw_all` | 0.2174 |
| `word_len_1_frac` | 0.1917 |

**kaj_sotala:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | 0.4851 |
| `fw_as` | 0.3796 |
| `cat_discourse_rate` | 0.3485 |
| `fw_it` | 0.3131 |
| `fw_you` | 0.2946 |

**katjagrace:**

| Feature | Coefficient |
|---------|-------------|
| `punct_apost_rate` | 0.4929 |
| `word_len_2_frac` | 0.3546 |
| `fw_so` | 0.3320 |
| `fw_there` | 0.3142 |
| `fw_would` | 0.3048 |

**petermccluskey:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | 0.4314 |
| `fw_will` | 0.2057 |
| `fw_as` | 0.1616 |
| `type_token_ratio` | 0.1509 |
| `hapax_ratio` | 0.1503 |

**raemon:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | 0.4794 |
| `fw_get` | 0.3606 |
| `fw_but` | 0.3318 |
| `pronoun_start_rate` | 0.2889 |
| `fw_about` | 0.2724 |

**ruby:**

| Feature | Coefficient |
|---------|-------------|
| `fw_we` | 0.2725 |
| `punct_period_rate` | 0.2046 |
| `fw_my` | 0.1925 |
| `fw_and` | 0.1895 |
| `n_sentences` | 0.1848 |

**ryan_greenblatt:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | 0.4296 |
| `punct_paren_rate` | 0.3344 |
| `cat_hedge_rate` | 0.2794 |
| `fw_which` | 0.2546 |
| `min_sent_len` | 0.2539 |

**sarahconstantin:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | 0.4669 |
| `fw_or` | 0.3739 |
| `punct_semicolon_rate` | 0.3648 |
| `punct_comma_rate` | 0.3464 |
| `fw_in` | 0.2753 |

**screwtape:**

| Feature | Coefficient |
|---------|-------------|
| `fw_a` | 0.3598 |
| `fw_say` | 0.3089 |
| `fw_if` | 0.2485 |
| `fw_or` | 0.2381 |
| `fw_i` | 0.2273 |

**steve2152:**

| Feature | Coefficient |
|---------|-------------|
| `punct_excl_rate` | 0.5496 |
| `punct_paren_rate` | 0.4161 |
| `punct_dash_rate` | 0.4096 |
| `punct_dquote_rate` | 0.3760 |
| `punct_comma_rate` | 0.3093 |

**tsvibt:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | 0.5806 |
| `punct_comma_rate` | 0.4792 |
| `cat_amplifier_rate` | 0.3402 |
| `punct_period_rate` | 0.2693 |
| `fw_up` | 0.2165 |

**zack_m_davis:**

| Feature | Coefficient |
|---------|-------------|
| `std_sent_len` | 0.3487 |
| `punct_excl_rate` | 0.3294 |
| `fw_would` | 0.2863 |
| `avg_sent_len` | 0.2820 |
| `fw_who` | 0.2783 |

**zvi:**

| Feature | Coefficient |
|---------|-------------|
| `uppercase_ratio` | 0.5019 |
| `punct_period_rate` | 0.3297 |
| `n_sentences` | 0.3068 |
| `fw_not` | 0.2861 |
| `punct_apost_rate` | 0.2774 |

### Top 5 Negative Features Per Author

_Features with most negative coefficient — strongly disassociated from this author._

**abramdemski:**

| Feature | Coefficient |
|---------|-------------|
| `fw_and` | -0.5841 |
| `cat_conj_rate` | -0.3972 |
| `fw_that` | -0.3542 |
| `std_sent_len` | -0.1956 |
| `max_sent_len` | -0.1936 |

**adamshimi:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | -0.3919 |
| `punct_period_rate` | -0.3624 |
| `fw_to` | -0.3069 |
| `fw_if` | -0.2217 |
| `word_len_6_frac` | -0.2192 |

**benquo:**

| Feature | Coefficient |
|---------|-------------|
| `fw_will` | -0.3460 |
| `punct_dquote_rate` | -0.2840 |
| `fw_of` | -0.2493 |
| `punct_paren_rate` | -0.2453 |
| `cat_hedge_rate` | -0.2394 |

**buck:**

| Feature | Coefficient |
|---------|-------------|
| `fw_what` | -0.3228 |
| `char_i_rate` | -0.3148 |
| `fw_at` | -0.2822 |
| `fw_not` | -0.2822 |
| `punct_dquote_rate` | -0.2704 |

**david-gross:**

| Feature | Coefficient |
|---------|-------------|
| `cat_amplifier_rate` | -0.3201 |
| `uppercase_ratio` | -0.3129 |
| `word_len_5_frac` | -0.2856 |
| `fw_the` | -0.2671 |
| `punct_apost_rate` | -0.2577 |

**eliezer_yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `punct_paren_rate` | -0.5857 |
| `fw_for` | -0.4922 |
| `cat_conj_rate` | -0.3089 |
| `fw_do` | -0.2553 |
| `char_i_rate` | -0.2367 |

**elizabeth-1:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | -0.3062 |
| `fw_be` | -0.2907 |
| `uppercase_ratio` | -0.2642 |
| `punct_apost_rate` | -0.2412 |
| `std_sent_len` | -0.2363 |

**gordon-seidoh-worley:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dash_rate` | -0.4448 |
| `punct_paren_rate` | -0.3533 |
| `fw_who` | -0.3477 |
| `fw_be` | -0.3108 |
| `punct_semicolon_rate` | -0.2981 |

**holdenkarnofsky:**

| Feature | Coefficient |
|---------|-------------|
| `fw_you` | -0.2756 |
| `punct_comma_rate` | -0.2564 |
| `cat_discourse_rate` | -0.2002 |
| `fw_but` | -0.1746 |
| `fw_in` | -0.1679 |

**jacob-falkovich:**

| Feature | Coefficient |
|---------|-------------|
| `std_sent_len` | -0.3789 |
| `cat_discourse_rate` | -0.3501 |
| `punct_paren_rate` | -0.3413 |
| `cat_hedge_rate` | -0.3253 |
| `punct_semicolon_rate` | -0.3234 |

**jasoncrawford:**

| Feature | Coefficient |
|---------|-------------|
| `punct_apost_rate` | -0.4906 |
| `contraction_rate` | -0.3881 |
| `fw_about` | -0.3123 |
| `fw_with` | -0.2946 |
| `std_word_len` | -0.2739 |

**joe-carlsmith:**

| Feature | Coefficient |
|---------|-------------|
| `fw_which` | -0.2568 |
| `fw_not` | -0.2118 |
| `fw_a` | -0.2106 |
| `punct_excl_rate` | -0.2033 |
| `fw_who` | -0.2019 |

**johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | -0.3685 |
| `fw_you` | -0.2797 |
| `word_len_2_frac` | -0.2788 |
| `honore_r` | -0.2765 |
| `hapax_ratio` | -0.2651 |

**kaj_sotala:**

| Feature | Coefficient |
|---------|-------------|
| `fw_on` | -0.4184 |
| `fw_we` | -0.3256 |
| `punct_paren_rate` | -0.2807 |
| `word_len_2_frac` | -0.2719 |
| `fw_so` | -0.2545 |

**katjagrace:**

| Feature | Coefficient |
|---------|-------------|
| `contraction_rate` | -0.4836 |
| `min_sent_len` | -0.3158 |
| `punct_dquote_rate` | -0.3075 |
| `fw_but` | -0.2961 |
| `punct_comma_rate` | -0.2737 |

**petermccluskey:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | -0.3848 |
| `fw_and` | -0.3049 |
| `cat_conj_rate` | -0.2981 |
| `fw_you` | -0.2280 |
| `cat_amplifier_rate` | -0.2269 |

**raemon:**

| Feature | Coefficient |
|---------|-------------|
| `fw_in` | -0.3158 |
| `fw_from` | -0.2837 |
| `char_a_rate` | -0.2752 |
| `punct_semicolon_rate` | -0.2508 |
| `fw_as` | -0.2226 |

**ruby:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | -0.4111 |
| `fw_an` | -0.3451 |
| `max_word_len` | -0.3207 |
| `fw_by` | -0.2871 |
| `fw_it` | -0.2868 |

**ryan_greenblatt:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | -0.3371 |
| `punct_dquote_rate` | -0.2981 |
| `fw_a` | -0.2964 |
| `word_len_3_frac` | -0.2714 |
| `fw_not` | -0.2383 |

**sarahconstantin:**

| Feature | Coefficient |
|---------|-------------|
| `punct_colon_rate` | -0.3843 |
| `fw_as` | -0.3220 |
| `fw_would` | -0.2657 |
| `fw_if` | -0.2645 |
| `punct_period_rate` | -0.2513 |

**screwtape:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dash_rate` | -0.3933 |
| `std_word_len` | -0.3546 |
| `avg_word_len` | -0.3183 |
| `punct_colon_rate` | -0.3104 |
| `cat_hedge_rate` | -0.2905 |

**steve2152:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | -0.2712 |
| `fw_it` | -0.2636 |
| `min_sent_len` | -0.2059 |
| `fw_say` | -0.1994 |
| `honore_r` | -0.1985 |

**tsvibt:**

| Feature | Coefficient |
|---------|-------------|
| `fw_which` | -0.2937 |
| `fw_i` | -0.2682 |
| `median_sent_len` | -0.2390 |
| `fw_this` | -0.2380 |
| `hapax_dis_ratio` | -0.2324 |

**zack_m_davis:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | -0.5605 |
| `fw_will` | -0.4447 |
| `fw_so` | -0.2669 |
| `word_len_4_frac` | -0.2541 |
| `cat_conj_rate` | -0.2310 |

**zvi:**

| Feature | Coefficient |
|---------|-------------|
| `median_sent_len` | -0.2788 |
| `punct_dquote_rate` | -0.2675 |
| `fw_a` | -0.2296 |
| `fw_by` | -0.2276 |
| `word_len_1_frac` | -0.1999 |

### Most Discriminative Features Overall

_Features with highest absolute coefficient range across authors._

| Feature | Max Coef | Min Coef | Range |
|---------|----------|----------|-------|
| `punct_paren_rate` | 0.4161 | -0.5857 | 1.0018 |
| `punct_semicolon_rate` | 0.5806 | -0.4111 | 0.9917 |
| `fw_this` | 0.4296 | -0.5605 | 0.9901 |
| `punct_apost_rate` | 0.4929 | -0.4906 | 0.9834 |
| `fw_and` | 0.3909 | -0.5841 | 0.9750 |
| `fw_but` | 0.6670 | -0.2961 | 0.9631 |
| `cat_hedge_rate` | 0.6302 | -0.3253 | 0.9555 |
| `fw_which` | 0.6242 | -0.2937 | 0.9180 |
| `punct_dquote_rate` | 0.4794 | -0.3919 | 0.8713 |
| `punct_comma_rate` | 0.4792 | -0.3848 | 0.8640 |
