# Logistic Regression — Authorship Stylometry

Thư mục này chứa các thí nghiệm **Logistic Regression** cho 3 bài toán stylometry:

1. **Authorship Attribution** — Xác định tác giả của một văn bản (multi-class)
2. **Chronological Stylometry** — Phân loại thời kỳ sáng tác (binary: early vs late)
3. **Authorship Verification** — Xác định hai văn bản có cùng tác giả không (binary pairwise)

---

## 📁 Cấu trúc files

```
logistic_regression/
├── logistic_regression_code.py           # Task 1: Authorship Attribution (nested CV)
├── run_chronological.py                  # Task 2: Chronological Stylometry
├── logistic_regression_verification.py   # Task 3: Authorship Verification (pairwise)
├── extract_features_for_n_authors.py     # Helper: trích xuất features cho N authors
├── plan.md                               # Kế hoạch nghiên cứu tổng thể
├── results_with_outliers.md              # Kết quả Attribution — có outliers
├── results_without_outliers.md           # Kết quả Attribution — không outliers
└── README.md                             # Hướng dẫn sử dụng (file này)
```

---

## ⚙️ Yêu cầu

- Python 3.10+
- Các thư viện: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tabulate`

Cài đặt nhanh:

```bash
pip install numpy pandas scikit-learn matplotlib tabulate
```

---

## 🚀 Cách chạy từng file

### 1. Authorship Attribution — `logistic_regression_code.py`

**Mô tả:** Huấn luyện Logistic Regression multi-class (softmax) với nested cross-validation để xác định tác giả của một văn bản dựa trên 107 stylometric features.

**Cách chạy:**

```bash
cd logistic_regression
python logistic_regression_code.py
```

**Đầu vào:**
- `../neural_network/author_features_extracted_full.csv` — 5 authors, 723 passages
- `../neural_network/feature_extracted_without_outliers.csv` — đã loại outliers

**Đầu ra:**
- `results_with_outliers.md` — báo cáo kết quả có outliers
- `results_without_outliers.md` — báo cáo kết quả không outliers

**Cấu hình có thể thay đổi trong code:**
- `PARAM_GRID` — lưới siêu tham số (C, max_iter, ...)
- `OUTER_FOLDS` / `INNER_FOLDS` — số fold ngoài/trong
- `RANDOM_STATE` — seed để tái lập kết quả

---

### 2. Chronological Stylometry — `run_chronological.py`

**Mô tả:** Phân tích sự thay đổi phong cách viết theo thời gian cho từng tác giả. So sánh các bài viết **early (≤2022)** vs **late (≥2024)**.

**Cách chạy:**

```bash
cd logistic_regression
python run_chronological.py
```

**Đầu vào:**
- `../dataset/lesswrong_regular/cleaned/` — các file JSON chứa văn bản + ngày tháng

**Đầu ra:**
- `chronological_results.md` — báo cáo kết quả theo từng tác giả

**Tùy chọn (command-line arguments):**

```bash
python run_chronological.py --data-dir ../dataset/lesswrong_regular/cleaned_5 \
                            --output-dir . \
                            --min-words 500
```

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--data-dir` | `../dataset/lesswrong_regular/cleaned` | Thư mục chứa JSON |
| `--output-dir` | `.` | Thư mục lưu kết quả |
| `--min-words` | `500` | Số từ tối thiểu mỗi passage |

---

### 3. Authorship Verification — `logistic_regression_verification.py`

**Mô tả:** Xác định xem hai văn bản có được viết bởi cùng một tác giả không. Sử dụng pairwise features: abs_diff, sq_diff, mult.

**Cách chạy:**

```bash
cd logistic_regression
python logistic_regression_verification.py
```

Chạy với model cụ thể:

```bash
# Chỉ chạy Logistic Regression
python logistic_regression_verification.py --models lr

# Chạy cả LR, SVM, Random Forest
python logistic_regression_verification.py --models lr svm rf
```

**Đầu vào:**
- `../neural_network/author_features_extracted_full.csv` — 5 authors
- `../neural_network/feature_extracted_without_outliers.csv` — không outliers
- `../SVM/LesswrongLarge.csv` — 35 authors (nếu có)

**Đầu ra:**
- `verification_results.md` — báo cáo kết quả tổng hợp

**Tùy chọn:**

```bash
python logistic_regression_verification.py --models lr --output ket_qua.md
```

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--models` | `['lr', 'svm', 'rf']` | Danh sách model cần chạy |
| `--output` | `verification_results.md` | File đầu ra |

---

### 4. Trích xuất features — `extract_features_for_n_authors.py`

**Mô tả:** Trích xuất 107 stylometric features từ dataset gốc cho N authors. Dùng để mở rộng thí nghiệm lên nhiều tác giả hơn.

**Cách chạy:**

```bash
cd logistic_regression

# 10 authors
python extract_features_for_n_authors.py \
    --input ../dataset/lesswrong_regular/cleaned_10 \
    --output features_10authors.csv

# 25 authors
python extract_features_for_n_authors.py \
    --input ../dataset/lesswrong_large/cleaned_25 \
    --output features_25authors.csv

# 35 authors
python extract_features_for_n_authors.py \
    --input ../dataset/lesswrong_large/cleaned_35 \
    --output features_35authors.csv
```

**Tùy chọn:**

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--input` | — | Thư mục chứa JSON (bắt buộc) |
| `--output` | — | File CSV đầu ra (bắt buộc) |
| `--min-words` | `500` | Số từ tối thiểu mỗi passage |

---

## 📊 Kiểm tra kết quả

Sau khi chạy, mở các file `.md` để xem báo cáo:

### File kết quả có sẵn (đã chạy xong)

| File | Nội dung |
|------|----------|
| `results_with_outliers.md` | Kết quả Attribution với outliers (Accuracy ~93.64%) |
| `results_without_outliers.md` | Kết quả Attribution không outliers (Accuracy ~95.78%) |

Các báo cáo này bao gồm:
- Bảng kết quả từng fold
- Confusion matrix trung bình
- Feature importance: top features đặc trưng cho từng tác giả
- Top features phân biệt nhất giữa các tác giả

### File kết quả mới sinh

| File | Khi nào có |
|------|-----------|
| `chronological_results.md` | Sau khi chạy `run_chronological.py` |
| `verification_results.md` | Sau khi chạy `logistic_regression_verification.py` |

---

## 🔬 Thứ tự chạy đề xuất

Nếu bạn muốn chạy từ đầu:

```bash
# Bước 1: Trích xuất features (nếu cần)
python extract_features_for_n_authors.py --input <path> --output <file.csv>

# Bước 2: Authorship Attribution
python logistic_regression_code.py

# Bước 3: Chronological Stylometry
python run_chronological.py

# Bước 4: Authorship Verification
python logistic_regression_verification.py --models lr svm rf
```

---

## 📝 Ghi chú

- Tất cả các file Python đều chạy độc lập (standalone), chỉ cần đúng đường dẫn dataset
- Các model khác (Random Forest, XGBoost, Neural Network, SVM) nằm ở thư mục riêng: `../random_forest_5author_150/`, `../xgboost_5author_150/`, `../neural_network/`, `../SVM/`
- Feature extraction dùng module `src.features` và `src.preprocess` từ thư mục gốc
