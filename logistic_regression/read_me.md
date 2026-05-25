# Logistic Regression - Authorship Stylometry

This directory contains **Logistic Regression** experiments for stylometry tasks.

---

## Directory Structure

```
logistic_regression/
|-- _config.py                                              # Shared config, constants, helpers
|-- read_me.md                                             # Usage guide
|
|-- task1_authorship_attribution/                          # Task 1: Authorship Attribution
|   |-- logistic_regression_code.py                        #   Nested CV training
|   |-- predict_demo.py                                    #   Demo: predict author
|   |-- results_with_outliers.md                           #   Results (with outliers)
|   |-- results_without_outliers.md                        #   Results (without outliers)
|   |-- extract_features_for_n_authors.py                  #   Helper: extract features for N authors
|   |-- features_10authors.csv                             #   Extracted features (10 authors)
|   |-- features_25authors.csv                             #   Extracted features (25 authors)
|   +-- _test_steve.txt, _test_zvi.txt                     #   Sample text files for demo
|
|-- task2_verification/                                    # Task 2: Authorship Verification
|   |-- logistic_regression_verification.py                #   Pairwise verification (Logistic Regression)
|   |-- verify_demo.py                                     #   Demo: check 2 texts
|   +-- verification_results.md                            #   Summary results
|
+-- bonus_ai_detection/                                    # Bonus: AI / Human Detection
    |-- ai_detection_lr.py                                 #   Train + predict
    |-- ai_detection_lr_model.pkl                          #   Trained model
    |-- ai_detection_lr_scaler.pkl                         #   Scaler
    |-- ai_detection_lr_features.pkl                       #   Feature names
    |-- ai_detection_lr_feature_importance.csv             #   Feature importance
    +-- ai_detection_lr_report.md                          #   Results report
```

---

## Requirements

- Python 3.10+
- Libraries: numpy, pandas, scikit-learn, joblib

```
pip install numpy pandas scikit-learn joblib
```

---

## How to Run Each Task

### Task 1: Authorship Attribution

Identify the author of a text (multi-class, 25 authors) using nested cross-validation.

```
cd logistic_regression/task1_authorship_attribution
python logistic_regression_code.py
```

**Input:**
- `features_25authors.csv` - 25 authors, 868 passages, 107 stylometric features

**Output:**
- `results_with_outliers.md`
- `results_without_outliers.md`

**Configuration in code:**
- `PARAM_GRID_LR` - hyperparameter grid (C, max_iter, ...)
- `OUTER_FOLDS` / `INNER_FOLDS` - number of outer/inner folds
- `RANDOM_STATE` - seed for reproducibility

#### Demo: Predict author for any text

```
cd logistic_regression/task1_authorship_attribution

# Run with sample text (default)
python predict_demo.py

# Input text directly
python predict_demo.py --text "Your text here..."

# Input from .txt file
python predict_demo.py --file _test_steve.txt
python predict_demo.py --file _test_zvi.txt

# Use model with outliers (default is without outliers)
python predict_demo.py --with-outliers
```

**Explanation:** The script retrains Logistic Regression on the full dataset, extracts features from the input text, and predicts the author with probabilities for each author (displayed as a bar chart). Default uses features without outliers.

#### Extract features for N authors (extension)

```
cd logistic_regression/task1_authorship_attribution

python extract_features_for_n_authors.py ^
    --input ../../dataset/lesswrong_regular/cleaned_10 ^
    --output features_10authors.csv

python extract_features_for_n_authors.py ^
    --input ../../dataset/lesswrong_large/cleaned_25 ^
    --output features_25authors.csv
```

---

### Task 2: Authorship Verification

Determine whether two texts are by the same author (binary pairwise). Uses Logistic Regression.

```
cd logistic_regression/task2_verification
python logistic_regression_verification.py
```

**Input:**
- `../../SVM/data/LesswrongLarge.csv` - 35 authors, 1484 passages

**Output:** `verification_results.md`

**How it works:**
- Generate positive pairs (same author) and negative pairs (different authors)
- Pair feature = `[abs_diff(A,B), sq_diff(A,B), mult(A,B)]` = 3 × n_features
- Train Logistic Regression binary classifier
- Uses author-level split + StratifiedKFold to prevent data leakage

#### Demo: Check if two texts share the same author

```
cd logistic_regression/task2_verification

# Run with sample texts (default)
python verify_demo.py

# Input 2 texts directly
python verify_demo.py --text1 "First text..." --text2 "Second text..."

# Input from 2 .txt files
python verify_demo.py --file1 text_a.txt --file2 text_b.txt
```

**Explanation:** The script trains Logistic Regression on all 35 authors, extracts features for both input texts, builds a pair feature vector, and predicts "SAME AUTHOR" or "DIFFERENT AUTHORS" with confidence.

---

### Bonus: AI / Human Detection

Classify a passage as HUMAN (<=2022) or AI (>=2024).

```
cd logistic_regression/bonus_ai_detection

# Train model
python ai_detection_lr.py

# Predict new text
python ai_detection_lr.py --text "Your text here..."
python ai_detection_lr.py --file input.txt
```

---

## Summary of Results

| Task | Dataset | Metric | Value |
|------|---------|--------|-------|
| Attribution (with outliers) | 25 auth, 868 passages | Accuracy | 73.26% |
| Attribution (without outliers) | 25 auth, 868 passages | Accuracy | 70.73% |
| Verification (35 auth) | 1484 passages | AUC | 0.7759 |
| AI Detection | 10 auth, 666 passages | Accuracy | 65.67% |
| AI Detection | 10 auth, 666 passages | AUC | 0.7420 |

> **Note:** Results updated after fixing data leakage (Scaler + IsolationForest fit only on training fold). Before the fix, the "without outliers" result was inflated to 75.10%. Verification and AI Detection models now include hyperparameter tuning (GridSearchCV).

---

## Notes

- All Python scripts are standalone — just point them at the correct dataset paths
- Feature extraction uses the `src.features` and `src.preprocess` modules from the project root
- Demo scripts retrain the model on each run (no pre-saved .pkl model used)
- Other models (Random Forest, XGBoost, Neural Network, SVM) are in separate directories
- Task 2 uses only Logistic Regression (no longer SVM, RF)
- **2026-05-25:** Fixed data leakage (Scaler + IsolationForest fit inside CV fold), added hyperparameter tuning for Verification + AI Detection, created `_config.py` shared module
