import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from data_prepare import generate_pairs

def run_logistic_regression_baseline(seed_path, c=1.0):
    print(f"============================================================")
    print(f"🚀 CHẠY THỬ LOGISTIC REGRESSION (BASELINE MODEL)")
    print(f"File: {seed_path} | C = {c}")
    print(f"============================================================\n")
    
    # Bắt đầu bấm giờ
    start_time = time.time()

    # 1. Load data
    data = np.load(seed_path, allow_pickle=True)
    X_train_raw = data['X_train_raw']
    y_train_authors = data['y_train_authors']
    X_test_pairs = data['X_test_pairs']
    y_test_pairs = data['y_test_pairs']
    unique_authors = np.unique(y_train_authors)
    
    # 2. Sinh dữ liệu (chỉ tốn thời gian ở bước này)
    print("[1/3] Đang sinh cặp dữ liệu Train (generate_pairs)...")
    X_train_full, y_train_full = generate_pairs(X_train_raw, y_train_authors, unique_authors)

    # 3. Chuẩn hóa
    print("[2/3] Đang chuẩn hóa dữ liệu (StandardScaler)...")
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_pairs_scaled = scaler.transform(X_test_pairs)
    
    # 4. Huấn luyện Logistic Regression
    print(f"[3/3] Đang huấn luyện Logistic Regression...")
    model = LogisticRegression(C=c, max_iter=1000, random_state=42) 
    model.fit(X_train_full_scaled, y_train_full)
    
    # Bấm giờ xong!
    train_time = time.time() - start_time
    
    # 5. Dự đoán và Đánh giá
    y_pred = model.predict(X_test_pairs_scaled)
    y_pred_proba = model.decision_function(X_test_pairs_scaled)
    
    test_acc = model.score(X_test_pairs_scaled, y_test_pairs)
    test_auc = roc_auc_score(y_test_pairs, y_pred_proba)
    
    print("\n" + "-"*45)
    print("🎯 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("-"*45)
    print(f"⏱️ Tổng thời gian (Data + Train): {train_time:.2f} giây")
    print(f"✅ Accuracy : {test_acc:.4f}")
    print(f"📈 ROC-AUC  : {test_auc:.4f}\n")
    
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(confusion_matrix(y_test_pairs, y_pred))
    print("\nBáo cáo chi tiết (Classification Report):")
    print(classification_report(y_test_pairs, y_pred, target_names=['Khác tác giả (0)', 'Cùng tác giả (1)']))
        
    print("\n============================================================\n")
