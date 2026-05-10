# AI Detection Report — Logistic Regression

## Kết quả trên tập Test (20%)

- **Accuracy:** 0.6642
- **AUC-ROC:** 0.7209

### Confusion Matrix

| | Pred HUMAN | Pred AI |
|---|---|---|
| Actual HUMAN | 44 | 23 |
| Actual AI | 22 | 45 |

### Cross-Validation (5-fold, trên Train)

- **Mean:** 0.6578
- **Std:** 0.0490

### Top Features -> HUMAN (<=2022)

| Feature | Coefficient |
|---|---|
| fw_we | -0.6278 |
| fw_the | -0.5881 |
| fw_this | -0.5353 |
| fw_of | -0.4805 |
| char_i_rate | -0.4509 |
| punct_excl_rate | -0.4304 |
| fw_to | -0.4150 |
| fw_it | -0.4040 |
| char_o_rate | -0.3906 |
| punct_question_rate | -0.3792 |

### Top Features → AI (≥2024)

| Feature | Coefficient |
|---|---|
| brunet_w | 0.7404 |
| std_word_len | 0.7257 |
| uppercase_ratio | 0.7232 |
| word_len_2_frac | 0.6932 |
| pronoun_start_rate | 0.6095 |
| word_len_3_frac | 0.4369 |
| std_sent_len | 0.3363 |
| avg_word_len | 0.3297 |
| fw_his | 0.3205 |
| median_word_len | 0.3083 |
