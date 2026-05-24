import numpy as np
import pandas as pd
import random
from itertools import combinations
from sklearn.svm import LinearSVC # DÙNG LINEARSVC THAY VÌ SVC
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def generate_pairs(features, authors_array, selected_authors, max_pairs_per_auth=200):
    """Hàm sinh cặp On-the-fly (Sao chép y hệt từ File 1)"""
    author_indices = {auth: np.where(authors_array == auth)[0] for auth in selected_authors}
    pos_pairs, neg_pairs = [], []

    for auth in selected_authors:
        idx = author_indices[auth]
        all_combos = list(combinations(idx, 2))
        if len(all_combos) > max_pairs_per_auth:
            all_combos = random.sample(all_combos, max_pairs_per_auth)

        for i, j in all_combos:
            f1, f2 = features[i], features[j]
            combined = np.concatenate([np.abs(f1 - f2), (f1 - f2)**2, f1 * f2])
            pos_pairs.append(combined)

    num_pos = len(pos_pairs)
    selected_authors_list = list(selected_authors)
    
    if len(selected_authors_list) < 2:
        return np.array([]), np.array([])

    for _ in range(num_pos):
        a1, a2 = random.sample(selected_authors_list, 2)
        idx1 = random.choice(author_indices[a1])
        idx2 = random.choice(author_indices[a2])

        f1, f2 = features[idx1], features[idx2]
        combined = np.concatenate([np.abs(f1 - f2), (f1 - f2)**2, f1 * f2])
        neg_pairs.append(combined)

    X = np.vstack((pos_pairs, neg_pairs)).astype(np.float32)
    y = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))]).astype(np.int8)

    return X, y

def train_and_evaluate_linear(seed_path):
    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU HUẤN LUYỆN LINEARSVC VỚI: {seed_path}")
    print(f"{'='*50}")
    
    data = np.load(seed_path, allow_pickle=True)
    X_train_raw = data['X_train_raw']
    y_train_authors = data['y_train_authors']
    X_test_pairs = data['X_test_pairs']
    y_test_pairs = data['y_test_pairs']

    unique_authors = np.unique(y_train_authors)

    # 1. Không gian tìm kiếm cho Linear (KHÔNG CÓ GAMMA)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10]
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_score = 0
    best_params = None

    print("\n[PHASE 1: TÌM SIÊU THAM SỐ C]")
    for params in ParameterGrid(param_grid):
        fold_val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(unique_authors)):
            cv_train_authors = unique_authors[train_idx]
            cv_val_authors = unique_authors[val_idx]
            
            X_cv_train, y_cv_train = generate_pairs(X_train_raw, y_train_authors, cv_train_authors)
            X_cv_val, y_cv_val = generate_pairs(X_train_raw, y_train_authors, cv_val_authors)
            
            scaler = StandardScaler()
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)
            
            # DÙNG LINEARSVC (dual=False tối ưu cực tốt khi số samples > số features)
            model = LinearSVC(C=params['C'], dual=False, random_state=42, max_iter=5000)
            model.fit(X_cv_train_scaled, y_cv_train)
            
            val_acc = model.score(X_cv_val_scaled, y_cv_val)
            fold_val_scores.append(val_acc)
            
        mean_val = np.mean(fold_val_scores)
        print(f" => C = {params['C']} | Val Acc (Mean): {mean_val:.4f}")
        
        if mean_val > best_score:
            best_score = mean_val
            best_params = params

    # ==========================================
    # GIAI ĐOẠN 2: ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP
    # ==========================================
    print(f"\n[PHASE 2: TEST VỚI THAM SỐ TỐT NHẤT {best_params}]")
    X_train_full, y_train_full = generate_pairs(X_train_raw, y_train_authors, unique_authors)
    
    final_scaler = StandardScaler()
    X_train_full_scaled = final_scaler.fit_transform(X_train_full)
    X_test_pairs_scaled = final_scaler.transform(X_test_pairs)
    
    best_model = LinearSVC(C=best_params['C'], dual=False, random_state=42, max_iter=5000)
    best_model.fit(X_train_full_scaled, y_train_full)
    
    y_pred = best_model.predict(X_test_pairs_scaled)
    # LinearSVC dùng decision_function để tính khoảng cách tới siêu mặt phẳng (thay cho xác suất)
    y_pred_dist = best_model.decision_function(X_test_pairs_scaled)
    
    test_acc = best_model.score(X_test_pairs_scaled, y_test_pairs)
    test_auc = roc_auc_score(y_test_pairs, y_pred_dist)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_pairs, y_pred))
    
    print(classification_report(y_test_pairs, y_pred, target_names=['Different', 'Same']))
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC-AUC : {test_auc:.4f}")
    
    return test_acc, test_auc

if __name__ == "__main__":
    test_accuracies = []
    test_aucs = []
    
    # Chạy vòng lặp qua 5 file seed bạn đã tạo ở File 1
    for i in range(5):
        t = i + 19

        random.seed(t)
        np.random.seed(t)

        t = i + 19
        seed_path = f'processed_data_seed_{t}.npz'
        
        # Chạy model và thu thập kết quả
        acc, auc = train_and_evaluate_linear(seed_path)
        test_accuracies.append(acc)
        test_aucs.append(auc)
        
    print("\n" + "*"*50)
    print("TỔNG KẾT KẾT QUẢ LINEARSVC TRÊN 5 SEED")
    print("*"*50)
    print(f"Accuracy trung bình : {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    print(f"ROC-AUC trung bình  : {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")