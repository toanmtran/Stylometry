# Chronological Stylometry Results

## Per-Author Summary

| Author | Early (<=2022) | Late (>=2024) | CV Accuracy | Std |
|--------|----------------|---------------|-------------|-----|
| abramdemski | 40 | 27 | 0.7330 | 0.0945 |
| jessica-liu-taylor | 24 | 22 | 0.7800 | 0.1238 |
| johnswentworth | 39 | 40 | 0.6583 | 0.0928 |
| kaj_sotala | 40 | 23 | 0.7795 | 0.1602 |
| marius-hobbhahn | 23 | 20 | 0.8889 | 0.0994 |
| neel-nanda-1 | 40 | 40 | 0.9625 | 0.0306 |
| raemon | 39 | 37 | 0.7242 | 0.1278 |
| ricraz | 39 | 23 | 0.7115 | 0.1180 |
| ruby | 29 | 22 | 0.6455 | 0.0847 |
| steve2152 | 39 | 40 | 0.7458 | 0.0726 |
| zvi | 38 | 39 | 0.9600 | 0.0327 |

## Feature Analysis Per Author

### abramdemski

**Accuracy:** 0.7330 +/- 0.0945

**Features → Early period:**

```
           feature  coefficient
           fw_this    -0.679230
          fw_which    -0.602635
   punct_excl_rate    -0.551836
cat_amplifier_rate    -0.512678
             fw_so    -0.452899
```

**Features → Late period:**

```
             feature  coefficient
               fw_it     0.741499
punct_semicolon_rate     0.740999
  pronoun_start_rate     0.590829
              fw_she     0.511881
             fw_from     0.476875
```

### jessica-liu-taylor

**Accuracy:** 0.7800 +/- 0.1238

**Features → Early period:**

```
           feature  coefficient
           fw_this    -0.630819
           fw_that    -0.546122
cat_discourse_rate    -0.534523
           fw_they    -0.474190
            fw_who    -0.456504
```

**Features → Late period:**

```
        feature  coefficient
          fw_or     0.618008
        fw_with     0.565446
punct_dash_rate     0.462301
   std_word_len     0.451640
          fw_as     0.424895
```

### johnswentworth

**Accuracy:** 0.6583 +/- 0.0928

**Features → Early period:**

```
        feature  coefficient
punct_dash_rate    -1.058166
        fw_from    -0.929852
           fw_i    -0.700635
         fw_who    -0.607447
   min_sent_len    -0.571588
```

**Features → Late period:**

```
 feature  coefficient
 fw_that     1.266483
   fw_me     0.464208
   fw_so     0.442535
honore_r     0.440215
   fw_as     0.380627
```

### kaj_sotala

**Accuracy:** 0.7795 +/- 0.1602

**Features → Early period:**

```
         feature  coefficient
punct_comma_rate    -0.674904
        fw_which    -0.399383
          fw_for    -0.366748
           fw_by    -0.353039
          fw_the    -0.347569
```

**Features → Late period:**

```
           feature  coefficient
  punct_colon_rate     0.482721
   uppercase_ratio     0.437948
          fw_about     0.437490
cat_amplifier_rate     0.436392
            fw_she     0.381449
```

### marius-hobbhahn

**Accuracy:** 0.8889 +/- 0.0994

**Features → Early period:**

```
        feature  coefficient
         fw_who    -0.449542
         fw_but    -0.403947
       fw_which    -0.403095
word_len_4_frac    -0.392949
          fw_on    -0.349292
```

**Features → Late period:**

```
         feature  coefficient
           fw_an     0.566752
     char_a_rate     0.434112
 punct_dash_rate     0.396788
punct_comma_rate     0.305547
           fw_at     0.273241
```

### neel-nanda-1

**Accuracy:** 0.9625 +/- 0.0306

**Features → Early period:**

```
           feature  coefficient
   punct_excl_rate    -0.582795
          fw_there    -0.449345
cat_amplifier_rate    -0.433005
             fw_to    -0.414054
            fw_and    -0.383349
```

**Features → Late period:**

```
           feature  coefficient
       char_e_rate     0.366152
             fw_we     0.340860
      avg_word_len     0.298280
             fw_at     0.278946
cat_discourse_rate     0.266532
```

### raemon

**Accuracy:** 0.7242 +/- 0.1278

**Features → Early period:**

```
     feature  coefficient
       fw_be    -0.739174
       fw_my    -0.577562
      fw_and    -0.538807
avg_word_len    -0.536785
 char_o_rate    -0.444187
```

**Features → Late period:**

```
          feature  coefficient
punct_dquote_rate     0.738556
           fw_get     0.712530
         fw_would     0.600164
  hapax_dis_ratio     0.561615
     avg_sent_len     0.529344
```

### ricraz

**Accuracy:** 0.7115 +/- 0.1180

**Features → Early period:**

```
            feature  coefficient
     cat_hedge_rate    -0.580905
punct_question_rate    -0.553671
       min_sent_len    -0.526065
              fw_do    -0.489906
       avg_sent_len    -0.477448
```

**Features → Late period:**

```
          feature  coefficient
           fw_out     0.754377
punct_period_rate     0.673101
            fw_at     0.529510
          fw_from     0.518377
punct_dquote_rate     0.476849
```

### ruby

**Accuracy:** 0.6455 +/- 0.0847

**Features → Early period:**

```
         feature  coefficient
        fw_which    -0.769523
           fw_be    -0.616757
punct_colon_rate    -0.598857
           fw_of    -0.591454
     char_u_rate    -0.575630
```

**Features → Late period:**

```
          feature  coefficient
            fw_up     0.588588
punct_dquote_rate     0.568126
  word_len_4_frac     0.450210
          fw_that     0.394722
           fw_not     0.331628
```

### steve2152

**Accuracy:** 0.7458 +/- 0.0726

**Features → Early period:**

```
           feature  coefficient
              fw_i    -0.826599
             fw_we    -0.626254
pronoun_start_rate    -0.608405
           fw_have    -0.553699
            fw_say    -0.536307
```

**Features → Late period:**

```
         feature  coefficient
           fw_so     0.706483
punct_comma_rate     0.687256
   cat_conj_rate     0.665544
 word_len_6_frac     0.593393
           fw_an     0.577185
```

### zvi

**Accuracy:** 0.9600 +/- 0.0327

**Features → Early period:**

```
            feature  coefficient
   punct_paren_rate    -0.426447
punct_question_rate    -0.403899
    word_len_5_frac    -0.346406
             fw_not    -0.345894
             fw_get    -0.342081
```

**Features → Late period:**

```
         feature  coefficient
 uppercase_ratio     0.719954
          fw_but     0.619948
contraction_rate     0.519399
    std_word_len     0.432659
    std_sent_len     0.374115
```

