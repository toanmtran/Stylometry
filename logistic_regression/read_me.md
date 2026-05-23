"""# Logistic Regression - Authorship Stylometry

Thu muc nay chua cac thuc nghiem **Logistic Regression** cho cac bai toan stylometry.

---

## Cau truc thu muc

```
logistic_regression/
|-- _config.py                                              # Shared config, constants, helpers
|-- read_me.md                                             # Huong dan su dung
|
|-- task1_authorship_attribution/                          # Task 1: Authorship Attribution
|   |-- logistic_regression_code.py                        #   Nested CV training
|   |-- predict_demo.py                                    #   Demo: du doan tac gia
|   |-- results_with_outliers.md                           #   Ket qua co outliers
|   |-- results_without_outliers.md                        #   Ket qua khong outliers
|   |-- extract_features_for_n_authors.py                  #   Helper: trich features cho N authors
|   |-- features_10authors.csv                             #   Features da trich (10 authors)
|   |-- features_25authors.csv                             #   Features da trich (25 authors)
|   +-- _test_steve.txt, _test_zvi.txt                     #   File text mau de test demo
|
|-- task2_verification/                                    # Task 2: Authorship Verification
|   |-- logistic_regression_verification.py                #   Pairwise verification (Logistic Regression)
|   |-- verify_demo.py                                     #   Demo: kiem tra 2 van ban
|   +-- verification_results.md                            #   Ket qua tong hop
|
+-- bonus_ai_detection/                                    # Bonus: AI / Human Detection
    |-- ai_detection_lr.py                                 #   Train + predict
    |-- ai_detection_lr_model.pkl                          #   Model da train
    |-- ai_detection_lr_scaler.pkl                         #   Scaler
    |-- ai_detection_lr_features.pkl                       #   Feature names
    |-- ai_detection_lr_feature_importance.csv             #   Feature importance
    +-- ai_detection_lr_report.md                          #   Bao cao ket qua
```

---

## Yeu cau

- Python 3.10+
- Thu vien: numpy, pandas, scikit-learn

```
pip install numpy pandas scikit-learn
```

---

## Cach chay tung task

### Task 1: Authorship Attribution

Xac dinh tac gia cua mot van ban (multi-class, 25 authors) voi nested cross-validation.

```
cd logistic_regression/task1_authorship_attribution
python logistic_regression_code.py
```

**Dau vao:**
- `features_25authors.csv` - 25 authors, 868 passages, 107 stylometric features

**Dau ra:**
- `results_with_outliers.md`
- `results_without_outliers.md`

**Cau hinh trong code:**
- `PARAM_GRID` - luoi sieu tham so (C, max_iter, ...)
- `OUTER_FOLDS` / `INNER_FOLDS` - so fold ngoai/trong
- `RANDOM_STATE` - seed de tai lap ket qua

#### Demo: Du doan tac gia cho van ban bat ky

```
cd logistic_regression/task1_authorship_attribution

# Chay voi text mau (mac dinh)
python predict_demo.py

# Nhap text truc tiep
python predict_demo.py --text "Your text here..."

# Nhap tu file .txt
python predict_demo.py --file _test_steve.txt
python predict_demo.py --file _test_zvi.txt

# Dung model co outliers (mac dinh la khong outliers)
python predict_demo.py --with-outliers
```

**Giai thich:** Script se retrain Logistic Regression tren toan bo dataset, trich features tu text dau vao, va du doan tac gia kem xac suat cho tung tac gia (hien thi bang bar chart). Mac dinh dung bo features da loai outliers.

#### Trich xuat features cho N authors (mo rong)

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

Xac dinh xem hai van ban co cung tac gia khong (binary pairwise). Dung Logistic Regression.

```
cd logistic_regression/task2_verification
python logistic_regression_verification.py
```

**Dau vao:**
- `../../SVM/LesswrongLarge.csv` - 35 authors, 1484 passages

**Dau ra:** `verification_results.md`

**Cach hoat dong:**
- Tao positive pairs (cung tac gia) va negative pairs (khac tac gia)
- Pair feature = `[abs_diff(A,B), sq_diff(A,B), mult(A,B)]` = 3 * n_features
- Train Logistic Regression binary classifier
- Dung GroupKFold de tranh data leakage

#### Demo: Kiem tra 2 van ban co cung tac gia khong

```
cd logistic_regression/task2_verification

# Chay voi text mau (mac dinh)
python verify_demo.py

# Nhap 2 text truc tiep
python verify_demo.py --text1 "First text..." --text2 "Second text..."

# Nhap tu 2 file .txt
python verify_demo.py --file1 text_a.txt --file2 text_b.txt
```

**Giai thich:** Script se train Logistic Regression tren toan bo 35 authors, trich features cho ca 2 van ban dau vao, tao pair feature, va du doan "CUNG TAC GIA" hay "KHAC TAC GIA" kem confidence.

---

### Bonus: AI / Human Detection

Phan loai bai viet la HUMAN (<=2022) hay AI (>=2024).

```
cd logistic_regression/bonus_ai_detection

# Train model
python ai_detection_lr.py

# Du doan van ban moi
python ai_detection_lr.py --text "Your text here..."
python ai_detection_lr.py --file input.txt
```

---


---

## Ket qua tom tat

| Task | Dataset | Metric | Gia tri |
|------|---------|--------|---------|
| Attribution (co outliers) | 25 auth, 868 passages | Accuracy | 73.26% |
| Attribution (khong outliers) | 25 auth, ~783 passages | Accuracy | 70.73% |
| Verification (35 auth) | 1484 passages | AUC | 0.7759 |
| AI Detection | 10 auth, 666 passages | Accuracy | 65.67% |
| AI Detection | 10 auth, 666 passages | AUC | 0.7420 |

> **Luu y:** Ket qua da duoc cap nhat sau khi fix data leakage (Scaler + IsolationForest chi fit tren training fold). Truoc khi fix, ket qua "khong outliers" bi thoi phong len 75.10%. Cac model Verification va AI Detection hien co hyperparameter tuning (GridSearchCV).

---

## Ghi chu

- Cac file Python chay doc lap (standalone), chi can dung duong dan dataset
- Feature extraction dung module `src.features` va `src.preprocess` tu thu muc goc
- File demo se retrain model moi moi lan chay (khong dung model pkl san)
- Cac model khac (Random Forest, XGBoost, Neural Network, SVM) nam o thu muc rieng
- Task 2 chi dung Logistic Regression (khong con SVM, RF)
- **2026-05-23:** Da fix data leakage (Scaler + IsolationForest fit trong CV fold), them hyperparameter tuning cho Verification + AI Detection, tao `_config.py` shared module
"""