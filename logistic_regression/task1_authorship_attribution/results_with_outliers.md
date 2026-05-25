# Logistic Regression Authorship Classification — With Outliers

## Configuration

- **Classifier:** Logistic Regression (multinomial / softmax)
- **Scaler:** StandardScaler (re-fit per outer fold — no leakage)
- **Outlier removal:** none
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
| 1 | 0.7299 | 0.7393 | 0.7339 | 0.7200 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 2 | 0.7701 | 0.7955 | 0.7723 | 0.7704 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 3 | 0.7874 | 0.7955 | 0.7911 | 0.7758 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 4 | 0.6994 | 0.7293 | 0.7010 | 0.6879 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |
| 5 | 0.6763 | 0.6953 | 0.6784 | 0.6677 | `C=0.1, max_iter=1000, penalty=l2, solver=lbfgs` |

## Summary

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy           | 0.7326  | 0.0466  |
| Precision (macro)  | 0.7510 | 0.0438 |
| Recall (macro)     | 0.7353    | 0.0472    |
| Weighted F1        | 0.7244      | 0.0483      |

## Average Classification Report

_Per-class metrics averaged across all outer folds._

|                      |   precision |   recall |   f1-score |   support |
|:---------------------|------------:|---------:|-----------:|----------:|
| abramdemski          |    0.917857 | 0.917857 |   0.91619  |       7.4 |
| adamshimi            |    0.723333 | 0.817857 |   0.766732 |       7.6 |
| benquo               |    0.485859 | 0.55     |   0.483591 |       8   |
| buck                 |    0.78619  | 0.806667 |   0.792634 |       5.2 |
| david-gross          |    0.840873 | 0.871429 |   0.846092 |       7.8 |
| eliezer_yudkowsky    |    0.685758 | 0.608333 |   0.635094 |       8.2 |
| elizabeth-1          |    0.678571 | 0.793333 |   0.719267 |       5.8 |
| gordon-seidoh-worley |    0.653333 | 0.575    |   0.593557 |       8   |
| holdenkarnofsky      |    0.866667 | 0.673333 |   0.731111 |       5.6 |
| jacob-falkovich      |    0.66619  | 0.764286 |   0.7107   |       7.6 |
| jasoncrawford        |    0.765556 | 0.686111 |   0.712293 |       8.2 |
| joe-carlsmith        |    0.811111 | 0.909524 |   0.847179 |       6.4 |
| johnswentworth       |    0.95     | 0.857143 |   0.889377 |       6.8 |
| kaj_sotala           |    0.517692 | 0.483333 |   0.48022  |       8.2 |
| katjagrace           |    0.765556 | 0.777778 |   0.767161 |       8.2 |
| petermccluskey       |    0.971429 | 0.966667 |   0.966434 |       6.2 |
| raemon               |    0.672857 | 0.733333 |   0.694106 |       6   |
| ruby                 |    0.432381 | 0.42     |   0.419394 |       5.8 |
| ryan_greenblatt      |    0.888095 | 0.864286 |   0.871795 |       7.2 |
| sarahconstantin      |    0.665    | 0.628571 |   0.638974 |       7   |
| screwtape            |    0.764502 | 0.721429 |   0.715312 |       7.4 |
| steve2152            |    0.769286 | 0.82381  |   0.786451 |       6.8 |
| tsvibt               |    0.824762 | 0.6      |   0.639738 |       6.2 |
| zack_m_davis         |    0.750476 | 0.7      |   0.715711 |       6   |
| zvi                  |    0.921429 | 0.833333 |   0.837862 |       6   |
| macro avg            |    0.750991 | 0.735337 |   0.727079 |     173.6 |
| weighted avg         |    0.747388 | 0.732616 |   0.724355 |     173.6 |

## Feature Importance (Coefficient Analysis)

Trained on full dataset with best hyperparameters found via inner CV.

### Top 5 Positive Features Per Author

_Features with highest positive coefficient — strongly associated with this author._

**abramdemski:**

| Feature | Coefficient |
|---------|-------------|
| `cat_discourse_rate` | 0.4398 |
| `fw_which` | 0.3705 |
| `fw_this` | 0.3446 |
| `punct_semicolon_rate` | 0.3066 |
| `fw_a` | 0.2843 |

**adamshimi:**

| Feature | Coefficient |
|---------|-------------|
| `fw_what` | 0.5158 |
| `cat_conj_rate` | 0.4084 |
| `fw_one` | 0.2949 |
| `fw_the` | 0.2609 |
| `n_tokens` | 0.2586 |

**benquo:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | 0.3405 |
| `fw_the` | 0.3261 |
| `fw_they` | 0.3149 |
| `fw_if` | 0.3054 |
| `fw_who` | 0.2677 |

**buck:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | 0.3029 |
| `char_u_rate` | 0.2580 |
| `fw_this` | 0.2314 |
| `fw_that` | 0.2253 |
| `fw_for` | 0.2097 |

**david-gross:**

| Feature | Coefficient |
|---------|-------------|
| `fw_or` | 0.4371 |
| `fw_as` | 0.3824 |
| `char_u_rate` | 0.3091 |
| `fw_there` | 0.3004 |
| `honore_r` | 0.2959 |

**eliezer_yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | 0.4671 |
| `fw_his` | 0.3451 |
| `fw_that` | 0.3419 |
| `fw_not` | 0.3385 |
| `fw_go` | 0.3376 |

**elizabeth-1:**

| Feature | Coefficient |
|---------|-------------|
| `fw_but` | 0.6534 |
| `cat_amplifier_rate` | 0.3146 |
| `fw_so` | 0.2743 |
| `cat_conj_rate` | 0.2659 |
| `punct_paren_rate` | 0.2588 |

**gordon-seidoh-worley:**

| Feature | Coefficient |
|---------|-------------|
| `fw_we` | 0.4655 |
| `fw_so` | 0.3749 |
| `cat_discourse_rate` | 0.3604 |
| `cat_conj_rate` | 0.3316 |
| `fw_what` | 0.2782 |

**holdenkarnofsky:**

| Feature | Coefficient |
|---------|-------------|
| `cat_hedge_rate` | 0.6055 |
| `punct_paren_rate` | 0.4225 |
| `fw_be` | 0.3519 |
| `punct_dquote_rate` | 0.3123 |
| `fw_will` | 0.3096 |

**jacob-falkovich:**

| Feature | Coefficient |
|---------|-------------|
| `fw_who` | 0.4206 |
| `fw_and` | 0.3259 |
| `fw_by` | 0.3216 |
| `fw_her` | 0.2761 |
| `min_sent_len` | 0.2755 |

**jasoncrawford:**

| Feature | Coefficient |
|---------|-------------|
| `honore_r` | 0.4005 |
| `fw_and` | 0.3627 |
| `punct_comma_rate` | 0.3399 |
| `hapax_ratio` | 0.3076 |
| `n_vocab` | 0.2827 |

**joe-carlsmith:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | 0.5218 |
| `fw_in` | 0.4591 |
| `punct_dash_rate` | 0.3517 |
| `punct_colon_rate` | 0.3315 |
| `char_i_rate` | 0.2935 |

**johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `fw_which` | 0.6682 |
| `word_len_5_frac` | 0.2698 |
| `brunet_w` | 0.2580 |
| `fw_not` | 0.2289 |
| `fw_all` | 0.2218 |

**kaj_sotala:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | 0.5273 |
| `fw_as` | 0.3867 |
| `fw_it` | 0.3816 |
| `cat_discourse_rate` | 0.3432 |
| `fw_you` | 0.2958 |

**katjagrace:**

| Feature | Coefficient |
|---------|-------------|
| `punct_apost_rate` | 0.5298 |
| `word_len_2_frac` | 0.3763 |
| `fw_there` | 0.3414 |
| `fw_for` | 0.3097 |
| `fw_would` | 0.2977 |

**petermccluskey:**

| Feature | Coefficient |
|---------|-------------|
| `fw_that` | 0.3900 |
| `fw_will` | 0.1906 |
| `word_len_6_frac` | 0.1767 |
| `hapax_ratio` | 0.1656 |
| `fw_an` | 0.1654 |

**raemon:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | 0.5113 |
| `fw_get` | 0.3336 |
| `pronoun_start_rate` | 0.3198 |
| `fw_but` | 0.3165 |
| `fw_about` | 0.2987 |

**ruby:**

| Feature | Coefficient |
|---------|-------------|
| `fw_we` | 0.2538 |
| `punct_period_rate` | 0.2259 |
| `fw_my` | 0.2208 |
| `n_sentences` | 0.2203 |
| `cat_conj_rate` | 0.1797 |

**ryan_greenblatt:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | 0.4320 |
| `punct_paren_rate` | 0.3570 |
| `cat_hedge_rate` | 0.2829 |
| `min_sent_len` | 0.2777 |
| `brunet_w` | 0.2637 |

**sarahconstantin:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | 0.4936 |
| `fw_or` | 0.3427 |
| `punct_semicolon_rate` | 0.3415 |
| `fw_in` | 0.3339 |
| `punct_comma_rate` | 0.3009 |

**screwtape:**

| Feature | Coefficient |
|---------|-------------|
| `fw_a` | 0.3464 |
| `fw_say` | 0.3458 |
| `fw_if` | 0.2846 |
| `fw_i` | 0.2438 |
| `uppercase_ratio` | 0.2311 |

**steve2152:**

| Feature | Coefficient |
|---------|-------------|
| `punct_excl_rate` | 0.5635 |
| `punct_paren_rate` | 0.4116 |
| `punct_dquote_rate` | 0.4073 |
| `punct_dash_rate` | 0.3676 |
| `punct_comma_rate` | 0.3258 |

**tsvibt:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | 0.5553 |
| `punct_semicolon_rate` | 0.5187 |
| `cat_amplifier_rate` | 0.2675 |
| `punct_period_rate` | 0.2634 |
| `fw_that` | 0.2463 |

**zack_m_davis:**

| Feature | Coefficient |
|---------|-------------|
| `std_sent_len` | 0.3457 |
| `fw_who` | 0.3148 |
| `punct_excl_rate` | 0.3134 |
| `fw_would` | 0.3078 |
| `fw_the` | 0.3026 |

**zvi:**

| Feature | Coefficient |
|---------|-------------|
| `uppercase_ratio` | 0.5306 |
| `punct_period_rate` | 0.3149 |
| `n_sentences` | 0.3045 |
| `punct_apost_rate` | 0.2809 |
| `word_len_3_frac` | 0.2414 |

### Top 5 Negative Features Per Author

_Features with most negative coefficient — strongly disassociated from this author._

**abramdemski:**

| Feature | Coefficient |
|---------|-------------|
| `fw_and` | -0.5915 |
| `cat_conj_rate` | -0.4075 |
| `fw_that` | -0.3620 |
| `std_sent_len` | -0.2264 |
| `max_sent_len` | -0.2147 |

**adamshimi:**

| Feature | Coefficient |
|---------|-------------|
| `punct_period_rate` | -0.3839 |
| `punct_dquote_rate` | -0.3799 |
| `fw_to` | -0.3136 |
| `fw_who` | -0.2570 |
| `word_len_6_frac` | -0.2469 |

**benquo:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | -0.3728 |
| `char_u_rate` | -0.2779 |
| `cat_hedge_rate` | -0.2485 |
| `punct_paren_rate` | -0.2480 |
| `fw_will` | -0.2457 |

**buck:**

| Feature | Coefficient |
|---------|-------------|
| `fw_not` | -0.3264 |
| `char_i_rate` | -0.3252 |
| `fw_what` | -0.3102 |
| `punct_dquote_rate` | -0.2784 |
| `fw_of` | -0.2692 |

**david-gross:**

| Feature | Coefficient |
|---------|-------------|
| `cat_amplifier_rate` | -0.3687 |
| `uppercase_ratio` | -0.3337 |
| `fw_the` | -0.2911 |
| `fw_on` | -0.2855 |
| `word_len_5_frac` | -0.2822 |

**eliezer_yudkowsky:**

| Feature | Coefficient |
|---------|-------------|
| `punct_paren_rate` | -0.6171 |
| `fw_for` | -0.5205 |
| `cat_conj_rate` | -0.3128 |
| `fw_do` | -0.2761 |
| `fw_she` | -0.2386 |

**elizabeth-1:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dquote_rate` | -0.3235 |
| `fw_be` | -0.3131 |
| `punct_apost_rate` | -0.2803 |
| `uppercase_ratio` | -0.2633 |
| `std_sent_len` | -0.2471 |

**gordon-seidoh-worley:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dash_rate` | -0.4138 |
| `punct_paren_rate` | -0.3620 |
| `fw_which` | -0.3278 |
| `fw_would` | -0.3243 |
| `max_word_len` | -0.3134 |

**holdenkarnofsky:**

| Feature | Coefficient |
|---------|-------------|
| `fw_you` | -0.2975 |
| `punct_comma_rate` | -0.2416 |
| `fw_to` | -0.1752 |
| `fw_the` | -0.1704 |
| `fw_me` | -0.1589 |

**jacob-falkovich:**

| Feature | Coefficient |
|---------|-------------|
| `std_sent_len` | -0.3989 |
| `cat_hedge_rate` | -0.3852 |
| `cat_discourse_rate` | -0.3837 |
| `punct_semicolon_rate` | -0.3616 |
| `punct_comma_rate` | -0.2922 |

**jasoncrawford:**

| Feature | Coefficient |
|---------|-------------|
| `punct_apost_rate` | -0.4950 |
| `contraction_rate` | -0.3872 |
| `fw_about` | -0.3116 |
| `fw_i` | -0.2986 |
| `std_word_len` | -0.2965 |

**joe-carlsmith:**

| Feature | Coefficient |
|---------|-------------|
| `fw_which` | -0.3043 |
| `fw_not` | -0.2693 |
| `fw_there` | -0.2280 |
| `punct_excl_rate` | -0.2187 |
| `fw_a` | -0.1987 |

**johnswentworth:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | -0.3790 |
| `fw_you` | -0.2797 |
| `honore_r` | -0.2618 |
| `word_len_2_frac` | -0.2598 |
| `hapax_ratio` | -0.2586 |

**kaj_sotala:**

| Feature | Coefficient |
|---------|-------------|
| `fw_on` | -0.4065 |
| `word_len_2_frac` | -0.3070 |
| `fw_we` | -0.3019 |
| `fw_who` | -0.2906 |
| `punct_paren_rate` | -0.2587 |

**katjagrace:**

| Feature | Coefficient |
|---------|-------------|
| `contraction_rate` | -0.4629 |
| `punct_dquote_rate` | -0.3271 |
| `fw_but` | -0.3195 |
| `min_sent_len` | -0.3174 |
| `fw_get` | -0.2863 |

**petermccluskey:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | -0.3968 |
| `fw_and` | -0.3185 |
| `cat_conj_rate` | -0.2750 |
| `std_sent_len` | -0.2412 |
| `fw_you` | -0.2382 |

**raemon:**

| Feature | Coefficient |
|---------|-------------|
| `fw_in` | -0.3273 |
| `punct_semicolon_rate` | -0.3030 |
| `fw_from` | -0.2984 |
| `char_a_rate` | -0.2823 |
| `fw_that` | -0.2272 |

**ruby:**

| Feature | Coefficient |
|---------|-------------|
| `punct_semicolon_rate` | -0.4209 |
| `fw_an` | -0.3469 |
| `punct_dquote_rate` | -0.3125 |
| `fw_by` | -0.2971 |
| `fw_they` | -0.2877 |

**ryan_greenblatt:**

| Feature | Coefficient |
|---------|-------------|
| `punct_comma_rate` | -0.3299 |
| `fw_a` | -0.3194 |
| `word_len_3_frac` | -0.3011 |
| `punct_dquote_rate` | -0.2997 |
| `n_vocab` | -0.2494 |

**sarahconstantin:**

| Feature | Coefficient |
|---------|-------------|
| `punct_colon_rate` | -0.4132 |
| `fw_as` | -0.3233 |
| `punct_period_rate` | -0.3075 |
| `fw_about` | -0.2677 |
| `fw_would` | -0.2672 |

**screwtape:**

| Feature | Coefficient |
|---------|-------------|
| `punct_dash_rate` | -0.3990 |
| `std_word_len` | -0.3446 |
| `cat_hedge_rate` | -0.3151 |
| `avg_word_len` | -0.3111 |
| `punct_colon_rate` | -0.2990 |

**steve2152:**

| Feature | Coefficient |
|---------|-------------|
| `fw_to` | -0.3254 |
| `fw_it` | -0.3009 |
| `honore_r` | -0.2276 |
| `fw_say` | -0.2134 |
| `char_o_rate` | -0.2062 |

**tsvibt:**

| Feature | Coefficient |
|---------|-------------|
| `fw_i` | -0.2809 |
| `fw_all` | -0.2774 |
| `fw_this` | -0.2571 |
| `median_sent_len` | -0.2480 |
| `fw_he` | -0.2449 |

**zack_m_davis:**

| Feature | Coefficient |
|---------|-------------|
| `fw_this` | -0.5329 |
| `fw_will` | -0.4731 |
| `fw_so` | -0.3459 |
| `word_len_4_frac` | -0.2676 |
| `cat_amplifier_rate` | -0.2522 |

**zvi:**

| Feature | Coefficient |
|---------|-------------|
| `median_sent_len` | -0.3062 |
| `punct_dquote_rate` | -0.2892 |
| `fw_a` | -0.2786 |
| `fw_by` | -0.2388 |
| `punct_semicolon_rate` | -0.2270 |

### Most Discriminative Features Overall

_Features with highest absolute coefficient range across authors._

| Feature | Max Coef | Min Coef | Range |
|---------|----------|----------|-------|
| `punct_paren_rate` | 0.4225 | -0.6171 | 1.0396 |
| `punct_apost_rate` | 0.5298 | -0.4950 | 1.0248 |
| `fw_which` | 0.6682 | -0.3278 | 0.9961 |
| `cat_hedge_rate` | 0.6055 | -0.3852 | 0.9908 |
| `fw_but` | 0.6534 | -0.3195 | 0.9729 |
| `fw_this` | 0.4320 | -0.5329 | 0.9648 |
| `fw_and` | 0.3627 | -0.5915 | 0.9542 |
| `punct_comma_rate` | 0.5553 | -0.3968 | 0.9520 |
| `punct_semicolon_rate` | 0.5187 | -0.4209 | 0.9396 |
| `punct_dquote_rate` | 0.5113 | -0.3799 | 0.8913 |
