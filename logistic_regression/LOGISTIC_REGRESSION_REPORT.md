# Logistic Regression for Authorship Stylometry

## Overview

This report documents the application of Logistic Regression to three stylometry tasks: authorship attribution, authorship verification, and AI-generated text detection. All experiments use 107 handcrafted stylometric features extracted from the LessWrong dataset.

**Key results:**

| Task | Type | Dataset | Metric | Score | Random Baseline |
|------|------|---------|--------|-------|-----------------|
| Attribution (w/ outliers) | 25-class | 868 passages | Accuracy | 73.26% ± 4.66% | 4% |
| Attribution (w/o outliers) | 25-class | 868 passages | Accuracy | 70.73% ± 4.00% | 4% |
| Verification | Binary pairwise | 1,484 passages, 35 authors | AUC | 0.7759 | 0.5 |
| AI Detection | Binary | 666 passages | Accuracy | 65.67% | 50% |

**Repository:** `logistic_regression/` — standalone scripts with a shared `_config.py` module.

---

## 1. Mathematical Foundation

### 1.1 Binary Classification (Sigmoid)

Logistic Regression is a linear model for classification. Given a feature vector $\mathbf{x} \in \mathbb{R}^{107}$, the model computes a raw score and maps it to a probability via the sigmoid function:

$$z = \mathbf{w}^T \mathbf{x} + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- $\mathbf{w} \in \mathbb{R}^{107}$ — weight vector (one coefficient per feature)
- $b$ — bias term
- $\hat{y} \in (0, 1)$ — predicted probability of the positive class

**Decision rule:** predict class 1 if $\hat{y} \geq 0.5$, else class 0.

**Interpretation:** $z$ is the log-odds of the positive class: $\ln\left(\frac{\hat{y}}{1-\hat{y}}\right) = \mathbf{w}^T \mathbf{x} + b$. Each weight $w_i$ quantifies how much the log-odds increase per unit increase in feature $x_i$.

### 1.2 Multi-class Extension (Softmax)

For K-class problems (e.g., 25 authors), the model uses softmax instead of sigmoid. Each class $k$ has its own weight vector $\mathbf{w}_k$, producing a probability distribution:

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad\text{where}\quad z_k = \mathbf{w}_k^T \mathbf{x} + b_k$$

The denominator $\sum e^{z_j}$ normalizes the outputs so that $\sum_{k=1}^{K} P(y=k \mid \mathbf{x}) = 1$, forming a valid probability distribution. The predicted class is $\arg\max_k P(y=k \mid \mathbf{x})$.

Total parameters for 25 authors: 25 × 107 weights + 25 biases = **2,700 parameters**.

### 1.3 Loss Function — Cross-Entropy

Training minimizes the cross-entropy (log-loss) between true labels and predicted probabilities:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N}\Bigl[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\Bigr]$$

where $N$ is the number of training samples, $y_i \in \{0, 1\}$ is the ground-truth label, and $\hat{y}_i$ is the predicted probability.

Cross-entropy heavily penalizes confident but wrong predictions:

| Ground Truth $y$ | Prediction $\hat{y}$ | $-\log(\hat{y})$ if $y=1$, $-\log(1-\hat{y})$ if $y=0$ | Interpretation |
|:---:|:---:|:---:|---|
| 1 | 0.99 | 0.01 | Correct, high confidence — minimal penalty |
| 1 | 0.01 | 4.61 | Wrong, high confidence — severe penalty |
| 0 | 0.01 | 0.01 | Correct, high confidence — minimal penalty |
| 0 | 0.51 | 0.71 | Wrong, low confidence — moderate penalty |

The gradient of the loss with respect to each weight $w_j$ drives the update: $\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)\,x_{ij}$. The model corrects itself proportionally to the error — larger mistakes produce larger updates.

### 1.4 Optimization — L-BFGS

Weights are optimized via **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), a quasi-Newton method. Unlike vanilla gradient descent, L-BFGS estimates the curvature of the loss surface and adapts the effective step size per parameter, converging in fewer iterations.

### 1.5 L2 Regularization

An L2 penalty is added to the loss to prevent overfitting:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cross-entropy}} + \lambda \sum_{i} w_i^2$$

Scikit-learn exposes the inverse regularization strength as $C = \frac{1}{\lambda}$:
- $C = 0.01$ → $\lambda$ large → strong regularization, weights shrunk toward zero
- $C = 100$ → $\lambda$ small → weak regularization, model may overfit

Grid search explores $C \in \{0.01, 0.1, 1, 10, 100\}$ to find the optimal value.

---

## 2. Feature Set — 107 Stylometric Features

Features are extracted via `src/features.py` and grouped into the following categories:

| Category | Count | Examples | Stylometric Significance |
|----------|:-----:|----------|--------------------------|
| Basic statistics | 4 | `n_tokens`, `type_token_ratio` | Text length, lexical diversity |
| Word length | 9 | `avg_word_len`, `word_len_3_frac` | Preference for short vs. long words |
| Sentence length | 6 | `avg_sent_len`, `std_sent_len` | Sentence rhythm and complexity |
| Vocabulary richness | 6 | `yule_k`, `honore_r`, `hapax_ratio` | Lexical diversity independent of text length |
| **Function words** | **50** | `fw_the`, `fw_of`, `fw_and`, `fw_that` | **Most important group** — subconscious, hard to manipulate |
| Stylistic categories | 4 | `cat_hedge_rate`, `cat_discourse_rate` | Hedges, amplifiers, discourse markers, conjunctions |
| Punctuation | 10 | `punct_comma_rate`, `punct_semicolon_rate` | Punctuation habits — highly author-specific |
| POS tags | 9 | `pos_noun`, `pos_verb`, `pos_modal` | Part-of-speech distribution |
| Character-level | 6 | `uppercase_ratio`, `char_a_rate` | Character and vowel frequencies |
| Other | 3 | `contraction_rate`, `pronoun_start_rate`, `avg_paragraph_len` | Contractions, sentence starters, paragraph structure |

**Why function words dominate stylometry:** Function words (the, of, and, that, etc.) are used subconsciously — authors do not consciously control them the way they control content words. This makes them a stable, hard-to-forge stylometric fingerprint.

---

## 3. Why StandardScaler Is Mandatory

Features span vastly different scales:

```
n_tokens        : 500–5000
avg_word_len    : 3.5–5.5
fw_the          : 0.01–0.08 (proportion)
yule_k          : 50–200
```

Logistic Regression uses gradient-based optimization (L-BFGS). Without scaling, features with larger numeric ranges dominate the gradients, causing slow or unstable convergence. StandardScaler transforms each feature to zero mean and unit variance:

$$x' = \frac{x - \mu}{\sigma}$$

The scaler is **fit on training data only** and then applied to test data to prevent data leakage.

---

## 4. Task 1 — Authorship Attribution

**Problem:** Given a text passage, identify the author among 25 candidates.  
**Type:** Multi-class classification (25 classes)  
**Model:** Logistic Regression with softmax  
**Dataset:** `features_25authors.csv` — 25 authors, 868 passages, 107 features

### 4.1 Pipeline

```
features_25authors.csv
    |
    +-- [Optional] IsolationForest per author --> removes ~85 outlier passages
    |
    v

  NESTED CROSS-VALIDATION
  ========================
  Outer: 5-fold StratifiedKFold --> estimates generalization
  Inner: 3-fold GridSearchCV --> tunes C on training split

  Hyperparameter grid:
    C in {0.01, 0.1, 1, 10, 100}
    penalty = L2, solver = L-BFGS
    max_iter in {1000, 2000, 5000}

  Per outer fold:
    1. Split into train (80%) / test (20%)
    2. Fit StandardScaler on train, transform both
    3. GridSearchCV on train (3-fold inner CV)
    4. Select best C, retrain on full train split
    5. Predict test, record metrics

  Final: Mean +/- Std across 5 outer folds
```

### 4.2 Results — With Outliers (868 passages)

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 73.26% | ±4.66% |
| Precision (macro) | 75.10% | ±4.38% |
| Recall (macro) | 73.53% | ±4.72% |
| Weighted F1 | 72.44% | ±4.83% |

**Per-fold breakdown:** C = 0.1 was selected in all 5 folds.

| Fold | Accuracy | Best C |
|------|----------|--------|
| 1 | 72.99% | 0.1 |
| 2 | 77.01% | 0.1 |
| 3 | 78.74% | 0.1 |
| 4 | 69.94% | 0.1 |
| 5 | 67.63% | 0.1 |

### 4.3 Results — Without Outliers (868 passages, outliers removed per-fold)

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 70.73% | ±4.00% |
| Precision (macro) | 72.71% | ±4.01% |
| Recall (macro) | 70.97% | ±4.32% |
| Weighted F1 | 69.99% | ±3.92% |

The outlier-free variant scores slightly lower, suggesting IsolationForest may remove challenging but valid passages that carry useful signal.

### 4.4 Feature Importance — Coefficient Analysis

Logistic Regression allows direct interpretation of learned weights. For each author, features with the largest positive coefficients represent that author's stylistic fingerprint; features with the largest negative coefficients are traits the author avoids.

**Top 10 most discriminative features overall** (by coefficient range = max - min across authors):

| Feature | Max Coef | Min Coef | Range |
|---------|:--------:|:--------:|:-----:|
| `punct_paren_rate` | +0.42 | −0.62 | 1.04 |
| `punct_apost_rate` | +0.53 | −0.50 | 1.02 |
| `fw_which` | +0.67 | −0.33 | 1.00 |
| `cat_hedge_rate` | +0.61 | −0.39 | 0.99 |
| `fw_but` | +0.65 | −0.32 | 0.97 |
| `fw_this` | +0.43 | −0.53 | 0.96 |
| `fw_and` | +0.36 | −0.59 | 0.95 |
| `punct_comma_rate` | +0.56 | −0.40 | 0.95 |
| `punct_semicolon_rate` | +0.52 | −0.42 | 0.94 |
| `punct_dquote_rate` | +0.51 | −0.38 | 0.89 |

Punctuation features account for 6 of the top 10 — each author has a distinctive "punctuation fingerprint." For instance, Eliezer Yudkowsky is characterized by high semicolon usage (`punct_semicolon_rate` = +0.47) and low parenthetical usage (`punct_paren_rate` = −0.62).

### 4.5 Prediction Demo

```bash
python predict_demo.py                        # sample text
python predict_demo.py --file input.txt       # from file
python predict_demo.py --text "Your text..."  # inline
```

The script retrains the model on the full dataset, extracts features from the input text, and outputs the predicted author with a probability bar chart for all 25 candidates. The confidence margin (top1 − top2 probability) quantifies prediction reliability.

---

## 5. Task 2 — Authorship Verification

**Problem:** Given two texts A and B, determine whether they were written by the same author.  
**Type:** Binary classification  
**Model:** Logistic Regression with sigmoid  
**Dataset:** `SVM/LesswrongLarge.csv` — 35 authors, 1,484 passages

### 5.1 Pair Features

This task uses a fundamentally different approach from attribution. Instead of classifying an author directly, the model learns to compare two feature vectors. For two passages with feature vectors $\mathbf{f}_A, \mathbf{f}_B \in \mathbb{R}^{107}$, a **pair feature vector** of 321 dimensions is constructed:

$$\text{pair}(\mathbf{f}_A, \mathbf{f}_B) = \Bigl[\;\underbrace{|\mathbf{f}_A - \mathbf{f}_B|}_{\text{107 abs. diff.}},\;\underbrace{(\mathbf{f}_A - \mathbf{f}_B)^2}_{\text{107 sq. diff.}},\;\underbrace{\mathbf{f}_A \odot \mathbf{f}_B}_{\text{107 products}}\;\Bigr] \in \mathbb{R}^{321}$$

| Component | Notation | Interpretation |
|-----------|----------|---------------|
| Absolute difference | $\|\mathbf{f}_A - \mathbf{f}_B\|$ | Small when same author |
| Squared difference | $(\mathbf{f}_A - \mathbf{f}_B)^2$ | Penalizes large deviations heavily |
| Element-wise product | $\mathbf{f}_A \odot \mathbf{f}_B$ | Captures shared stylistic profile (both high/low together) |

### 5.2 Pipeline

```
SVM/LesswrongLarge.csv (1,484 passages, 35 authors)
    |
    v

  1. AUTHOR-LEVEL SPLIT (not passage-level!)
     35 authors --> 28 train (80%) + 7 test (20%)
     Test authors are ENTIRELY UNSEEN during training
    |
    v

  2. PAIR GENERATION
     Positive pairs: same author, different passages
     Negative pairs: different authors (equal count)
     Maximum 500 pairs per author to control imbalance
    |
    v

  3. STRATIFIED K-FOLD CV (3 splits)
     Author-level split ensures all passages of one author
     stay in the same fold → no author-level data leakage
    |
    v

  4. TRAIN + EVALUATE ON HELD-OUT AUTHORS
     Train on 28 authors, test on 7 completely new authors
```

### 5.3 Results

| Metric | Value |
|--------|-------|
| CV Accuracy (StratifiedKFold) | 78.12% ± 0.12% |
| Test Accuracy (unseen authors) | 69.99% |
| Precision | 67.86% |
| Recall | 75.93% |
| F1-score | 71.67% |
| **AUC-ROC** | **0.7759** |

**Confusion matrix on test (7 held-out authors):**

|  | Predicted: Different | Predicted: Same |
|:---|---:|---:|
| **Actual: Different** | 2,086 (TN) | 1,171 (FP) |
| **Actual: Same** | 784 (FN) | 2,473 (TP) |

The model achieves AUC = 0.776 on authors never seen during training — substantially above the 0.5 random baseline. Recall exceeds precision (75.9% vs. 67.9%), indicating the model tends to err on the side of claiming authorship matches.

### 5.4 Why Author-Level Split?

A standard KFold would split passages randomly, placing passages from the same author in both train and test folds. The model would then learn to recognize specific authors rather than learning to *compare* writing styles. An author-level train/test split (80%/20% by author) combined with StratifiedKFold ensures all passages from a given author stay in the same fold, forcing the model to learn a genuine comparison function.

### 5.5 Verification Demo

```bash
python verify_demo.py                                # sample texts
python verify_demo.py --file1 a.txt --file2 b.txt    # from files
python verify_demo.py --text1 "..." --text2 "..."    # inline
```

Output: "SAME AUTHOR (confidence: XX%)" or "DIFFERENT AUTHORS (confidence: XX%)".

---

## 6. Bonus — AI vs. Human Text Detection

**Problem:** Classify a passage as human-written (≤2022) or AI-generated (≥2024).  
**Type:** Binary classification  
**Model:** Logistic Regression with sigmoid  
**Dataset:** `dataset/lesswrong_regular/cleaned/` — passages with date metadata

### 6.1 Labeling Strategy

Passages are labeled by year of publication:

```
Year ≤ 2022 → HUMAN (class 0) — pre-AI era
Year ≥ 2024 → AI (class 1)    — AI widely available
Year = 2023 → EXCLUDED        — ambiguous transition year
```

This creates a multi-author generalized model: the classifier does not depend on knowing the author, only the stylistic signals that correlate with the writing era.

### 6.2 Pipeline

```
Raw passages with dates
    |
    v

  PHASE 1: DATA PREPARATION
    1. Filter by year --> label HUMAN/AI, drop 2023
    2. Clean text, filter < 500 words
    3. Balance classes (sample min count from each)
    4. Truncate all passages to minimum word count M
    5. Extract 107 stylometric features
    |
    v

  PHASE 2: TRAINING + EVALUATION
    6. 80/20 stratified train/test split
    7. StandardScaler (fit on train only)
    8. LogisticRegression(C=0.01, class_weight="balanced")
    9. 5-fold StratifiedKFold CV on train set
    10. Final evaluation on held-out test set
```

### 6.3 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 65.67% |
| Test AUC-ROC | 74.20% |
| CV Mean Accuracy | 67.11% ± 4.44% |

**Confusion matrix:**

|  | Predicted HUMAN | Predicted AI |
|:---|---:|---:|
| **Actual HUMAN** | 44 | 23 |
| **Actual AI** | 23 | 44 |

### 6.4 Top Discriminative Features

**Features associated with HUMAN writing (negative coefficients):**

| Feature | Coefficient | Interpretation |
|---------|:-----------:|----------------|
| `fw_this` | −0.1785 | Humans use "this" more often |
| `fw_which` | −0.1519 | Humans use "which" more often |
| `fw_the` | −0.1407 | Humans use "the" more often |
| `fw_of` | −0.1278 | Humans use "of" more often |
| `punct_excl_rate` | −0.1256 | Humans use more exclamation marks |

**Features associated with AI writing (positive coefficients):**

| Feature | Coefficient | Interpretation |
|---------|:-----------:|----------------|
| `uppercase_ratio` | +0.1769 | AI uses more uppercase |
| `pronoun_start_rate` | +0.1503 | AI starts sentences with pronouns more often |
| `std_word_len` | +0.1343 | AI varies word length more |
| `punct_semicolon_rate` | +0.1067 | AI uses more semicolons |
| `fw_so` | +0.1066 | AI uses "so" more often |

The pattern suggests AI-generated text exhibits greater variation in word/sentence length and distinctive punctuation habits, while human text has higher function-word density — consistent with the hypothesis that AI writing is "smooth but shallow" in its stylistic profile.

### 6.5 Usage

```bash
python ai_detection_lr.py                         # train model
python ai_detection_lr.py --text "Your text..."   # classify a passage
python ai_detection_lr.py --file input.txt         # classify from file
```

---

## 7. Evaluation Methodology

### 7.1 Metrics

| Metric | Formula | Used In |
|--------|---------|---------|
| Accuracy | (TP + TN) / Total | Tasks 1, 2, Bonus |
| Precision (macro) | Mean of per-class precision | Task 1 |
| Recall (macro) | Mean of per-class recall | Task 1 |
| Weighted F1 | Mean of per-class F1, weighted by support | Task 1 |
| AUC-ROC | Area under ROC curve | Tasks 2, Bonus |
| Confusion Matrix | TN / FP / FN / TP table | Tasks 2, Bonus |

### 7.2 Cross-Validation Design

| Method | Description | Prevents |
|--------|-------------|----------|
| **Nested CV** (5 outer × 3 inner) | Outer loop estimates performance; inner loop tunes C. Test data never influences hyperparameter selection. | Optimistic bias from tuning on test data |
| **StratifiedKFold + Author-level split** | Train on 80% authors, test on 20% new authors. StratifiedKFold within training set. All passages of one author stay together. | Author-level data leakage |

### 7.3 Nested vs. Flat Cross-Validation

Flat CV (single GridSearchCV) produces biased performance estimates because the same data is used to select C and report accuracy. Nested CV separates these concerns: the inner loop tunes C on the training split only, while the outer loop evaluates the selected model on untouched test data. The reported accuracy (73.26%) is an unbiased estimate of generalization performance.

---

## 8. Project Structure

```
logistic_regression/
│
├── _config.py                              # Shared constants, parameter grids, helpers
│
├── task1_authorship_attribution/
│   ├── logistic_regression_code.py         # Nested CV training + evaluation
│   ├── predict_demo.py                     # Interactive author prediction
│   ├── extract_features_for_n_authors.py   # Feature extraction from raw dataset
│   ├── features_10authors.csv              # Pre-extracted features (10 authors)
│   ├── features_25authors.csv              # Pre-extracted features (25 authors)
│   ├── results_with_outliers.md            # Full results (with outliers)
│   └── results_without_outliers.md         # Full results (without outliers)
│
├── task2_verification/
│   ├── logistic_regression_verification.py # Pairwise verification training
│   ├── verify_demo.py                      # Interactive pair verification
│   └── verification_results.md             # Verification results
│
├── bonus_ai_detection/
│   ├── ai_detection_lr.py                  # Training + inference script
│   ├── ai_detection_lr_model.pkl           # Serialized model
│   ├── ai_detection_lr_scaler.pkl          # Serialized scaler
│   ├── ai_detection_lr_features.pkl        # Feature name list
│   ├── ai_detection_lr_feature_importance.csv
│   └── ai_detection_lr_report.md
│
└── read_me.md                              # Quick-start guide (Vietnamese)
```

---

## 9. Advantages and Limitations

### 9.1 Strengths of Logistic Regression for Stylometry

1. **Interpretability.** Each coefficient w_i directly quantifies the contribution of a feature to the log-odds of each class. The sign and magnitude tell a clear story: this is the author's stylistic fingerprint. This is the primary advantage over tree ensembles or neural networks.

2. **Efficiency on small data.** With 868 passages and 2,700 parameters + L2 regularization, LR is less prone to overfitting than more complex models.

3. **Fast training.** L-BFGS converges in seconds, compared to the minutes required for GridSearchCV over neural networks.

4. **Probabilistic output.** Softmax/sigmoid outputs are well-calibrated probabilities, enabling confidence estimation and margin analysis.

### 9.2 Limitations

1. **Linearity assumption.** LR assumes features contribute linearly to log-odds. Complex non-linear interactions between features are not captured.

2. **Sensitivity to outliers.** Despite IsolationForest filtering, extreme feature values can distort the linear decision boundary.

3. **No sequential modeling.** LR treats the text as a bag of features, ignoring word order, syntactic structure, and rhetorical patterns.

4. **Moderate accuracy.** 73% accuracy for 25-author attribution, while 18× better than random, is insufficient for high-stakes forensic applications.

### 9.3 Potential Improvements

| Direction | Description | Trade-off |
|-----------|-------------|-----------|
| Feature expansion | Add character n-grams, TF-IDF, sentence embeddings | Higher accuracy, reduced interpretability |
| Ensemble | Combine LR with Random Forest and SVM | Better accuracy, retains LR interpretability |
| Feature selection | Remove low-variance or low-discriminative features | Reduces noise, improves generalization |
| Probability calibration | Platt scaling or isotonic regression | More reliable confidence scores |

---

## 10. Reproducibility

All experiments use `RANDOM_STATE = 42` for deterministic results. To reproduce:

```bash
pip install numpy pandas scikit-learn

# Task 1 — Attribution
cd logistic_regression/task1_authorship_attribution
python logistic_regression_code.py

# Task 2 — Verification
cd logistic_regression/task2_verification
python logistic_regression_verification.py

# Bonus — AI Detection
cd logistic_regression/bonus_ai_detection
python ai_detection_lr.py
```

Feature extraction depends on `src/features.py` and `src/preprocess.py` from the project root. These modules must be importable (the scripts handle path setup automatically).

---

*Generated from code in `logistic_regression/`. Results sourced from `results_with_outliers.md`, `results_without_outliers.md`, `verification_results.md`, and `ai_detection_lr_report.md`.*
