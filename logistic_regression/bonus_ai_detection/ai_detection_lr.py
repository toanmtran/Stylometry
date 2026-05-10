"""
AI Detection using Logistic Regression — Multi-Author Generalised Model

Ý tưởng:
  - Gộp tất cả bài viết của MỌI tác giả thành 2 nhóm:
      0 = HUMAN (<=2022) — trước khi AI phổ biến
      1 = AI      (>=2024) — khi AI đã được dùng phổ biến
  - Train một Logistic Regression duy nhất để phân loại
  - Dùng được cho BẤT KỲ bài viết mới nào, không cần biết tác giả
  - Chia 80% train / 20% test để đánh giá khách quan

Usage:
    python ai_detection_lr.py
    python ai_detection_lr.py --text "Your sample text here..."
    python ai_detection_lr.py --file input.txt
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features import extract_features
from src.preprocess import (
    strip_lesswrong_boilerplate,
    canonicalize_typography,
    word_count,
    truncate_to_words,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

# ── Config ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
MIN_WORDS = 500
TEST_SIZE = 0.2  # 80% train / 20% test
DATA_DIR = os.path.join("..", "..", "dataset", "lesswrong_regular", "cleaned")


def load_passages_with_dates(folder: str) -> list[dict]:
    """Load all passages with date info from cleaned JSON files."""
    records = []
    for fp in sorted(Path(folder).glob("*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            raw = f.read()
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([\]}])", r"\1", raw)
            items = json.loads(cleaned)
        for item in items:
            text = item.get("text", "")
            author = item.get("author", "Unknown").strip().lower()
            date = item.get("date", "")
            records.append({
                "author": author,
                "text": text,
                "date": date,
            })
    return records


def classify_era(date_str: str) -> int | None:
    """
    0 = HUMAN (<=2022) — trước AI
    1 = AI (>=2024)    — sau AI
    None = 2023 — bỏ qua (ambiguous)
    """
    if not date_str or len(date_str) < 4:
        return None
    try:
        year = int(date_str[:4])
    except ValueError:
        return None
    if year <= 2022:
        return 0  # HUMAN
    elif year >= 2024:
        return 1  # AI
    else:
        return None  # 2023 — ambiguous


def prepare_data(data_dir: str, min_words: int = MIN_WORDS):
    """Load data, phân loại HUMAN/AI, trích features."""
    print(f"[*] Loading passages from {data_dir}...")
    all_passages = load_passages_with_dates(data_dir)
    print(f"   Total passages loaded: {len(all_passages)}")

    # Phân loại era
    texts = []
    labels = []
    authors = []
    
    for p in all_passages:
        era = classify_era(p["date"])
        if era is None:
            continue
        
        text = canonicalize_typography(p["text"])
        text = strip_lesswrong_boilerplate(text)
        
        if word_count(text) < min_words:
            continue
        
        texts.append(text)
        labels.append(era)  # 0 = HUMAN, 1 = AI
        authors.append(p["author"])
    
    print(f"\n[Data summary]:")
    human_count = sum(1 for l in labels if l == 0)
    ai_count = sum(1 for l in labels if l == 1)
    print(f"   HUMAN (<=2022): {human_count} passages")
    print(f"   AI (>=2024):    {ai_count} passages")
    print(f"   Total:         {len(labels)} passages")
    
    # Cân bằng dữ liệu (lấy min giữa 2 class)
    min_count = min(human_count, ai_count)
    human_indices = [i for i, l in enumerate(labels) if l == 0]
    ai_indices = [i for i, l in enumerate(labels) if l == 1]
    
    # Lấy mẫu ngẫu nhiên để cân bằng
    np.random.seed(RANDOM_STATE)
    sampled_human = np.random.choice(human_indices, min_count, replace=False)
    sampled_ai = np.random.choice(ai_indices, min_count, replace=False)
    selected_indices = sorted(list(sampled_human) + list(sampled_ai))
    
    texts_balanced = [texts[i] for i in selected_indices]
    labels_balanced = [labels[i] for i in selected_indices]
    authors_balanced = [authors[i] for i in selected_indices]
    
    print(f"   After balancing: {len(texts_balanced)} passages")
    print(f"     HUMAN: {sum(1 for l in labels_balanced if l == 0)}")
    print(f"     AI:    {sum(1 for l in labels_balanced if l == 1)}")
    
    # Truncate to same word length
    min_wc = min(word_count(t) for t in texts_balanced)
    texts_balanced = [truncate_to_words(t, min_wc) for t in texts_balanced]
    
    # Trích features
    print(f"\n[Extracting 100+ stylometric features...]")
    feature_list = []
    feature_names = None
    for i, t in enumerate(texts_balanced):
        feats = extract_features(t)
        if feature_names is None:
            feature_names = list(feats.keys())
        feature_list.append(list(feats.values()))
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(texts_balanced)} passages...")
    
    X = np.array(feature_list)
    y = np.array(labels_balanced)
    
    return X, y, feature_names, authors_balanced


def train_and_evaluate(X, y, feature_names):
    """
    Train Logistic Regression với 80/20 split.
    - Train: 80% → dùng để train model
    - Test:  20% → dùng để đánh giá (không được nhìn khi train)
    """
    # ── Bước 1: Chia train/test ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CHIA DU LIEU: 80% TRAIN / 20% TEST")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Giữ tỷ lệ HUMAN/AI ở cả 2 tập
    )
    
    print(f"   Train: {len(X_train)} passages ({sum(y_train==0)} HUMAN + {sum(y_train==1)} AI)")
    print(f"   Test:  {len(X_test)} passages ({sum(y_test==0)} HUMAN + {sum(y_test==1)} AI)")
    
    # ── Bước 2: Scale features (chỉ fit trên TRAIN) ──────────────────
    print(f"\nTraining Logistic Regression...")
    print(f"   Features: {X.shape[1]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit chỉ trên train
    X_test_scaled = scaler.transform(X_test)        # transform trên test
    
    # ── Bước 3: Train model ──────────────────────────────────────────
    clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train_scaled, y_train)
    
    # ── Bước 4: Đánh giá trên TEST (quan trọng!) ─────────────────────
    print(f"\n{'='*60}")
    print(f"  KET QUA TREN TAP TEST (20%)")
    print(f"{'='*60}")
    
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    
    print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   AUC-ROC:   {test_auc:.4f}")
    print()
    print(f"   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  +----------------------+")
    print(f"                  | Pred HUMAN | Pred AI  |")
    print(f"   ---------------+------------+----------+")
    print(f"   Actual HUMAN   |     {cm[0,0]:3d}      |    {cm[0,1]:3d}     |")
    print(f"   ---------------+------------+----------+")
    print(f"   Actual AI      |     {cm[1,0]:3d}      |    {cm[1,1]:3d}     |")
    print(f"                  +----------------------+")
    print()
    print(classification_report(y_test, y_pred, target_names=['HUMAN (<=2022)', 'AI (>=2024)']))
    
    # ── Bước 5: Cross-validation trên TRAIN để kiểm tra stability ────
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION (5-fold, chi tren TRAIN)")
    print(f"{'='*60}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    
    print(f"   CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"   Min: {cv_scores.min():.4f}, Max: {cv_scores.max():.4f}")
    
    # ── Bước 6: Feature importance (từ model đã train) ───────────────
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False)
    
    print(f"\n{'='*60}")
    print(f"  TOP 15 FEATURES PHAN BIET HUMAN vs AI")
    print(f"{'='*60}")
    print(f"   {'Feature':25s} {'Coef':>10s} {'Huong':>15s}")
    print(f"   {'-'*25} {'-'*10} {'-'*15}")
    for _, row in coef_df.head(15).iterrows():
        direction = "AI (>=2024)" if row["coefficient"] > 0 else "HUMAN (<=2022)"
        print(f"   {row['feature']:25s} {row['coefficient']:>+10.4f} {direction:>20s}")
    
    # ── Lưu kết quả ra file ──────────────────────────────────────────
    results = {
        "test_accuracy": test_acc,
        "test_auc_roc": test_auc,
        "confusion_matrix": cm.tolist(),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "top_features_human": coef_df[coef_df["coefficient"] < 0].head(10)[
            ["feature", "coefficient"]
        ].to_dict("records"),
        "top_features_ai": coef_df[coef_df["coefficient"] > 0].head(10)[
            ["feature", "coefficient"]
        ].to_dict("records"),
    }
    
    return clf, scaler, coef_df, results, X_test, y_test, y_pred, y_prob


def predict_text(text: str, clf, scaler, feature_names, threshold: float = 0.5):
    """Dự đoán một đoạn text bất kỳ: HUMAN hay AI?"""
    # Tiền xử lý
    text = canonicalize_typography(text)
    text = strip_lesswrong_boilerplate(text)
    
    if word_count(text) < MIN_WORDS:
        print(f"[!] Warning: Text chi co {word_count(text)} tu (toi thieu {MIN_WORDS})")
    
    # Trích features
    feats = extract_features(text)
    X = np.array([list(feats.values())])
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Dự đoán
    prob = clf.predict_proba(X_scaled)[0]
    pred = clf.predict(X_scaled)[0]
    
    result = {
        "label": "AI (>=2024)" if pred == 1 else "HUMAN (<=2022)",
        "prob_human": prob[0],
        "prob_ai": prob[1],
        "confidence": max(prob),
    }
    
    return result


def save_model(clf, scaler, coef_df, feature_names, results, output_dir="."):
    """Save model artifacts."""
    import joblib
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(clf, output_path / "ai_detection_lr_model.pkl")
    joblib.dump(scaler, output_path / "ai_detection_lr_scaler.pkl")
    joblib.dump(feature_names, output_path / "ai_detection_lr_features.pkl")
    
    # Save feature importance
    coef_df.to_csv(output_path / "ai_detection_lr_feature_importance.csv", index=False)
    
    # Save results report
    report_path = output_path / "ai_detection_lr_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AI Detection Report — Logistic Regression\n\n")
        f.write("## Kết quả trên tập Test (20%)\n\n")
        f.write(f"- **Accuracy:** {results['test_accuracy']:.4f}\n")
        f.write(f"- **AUC-ROC:** {results['test_auc_roc']:.4f}\n\n")
        f.write("### Confusion Matrix\n\n")
        f.write("| | Pred HUMAN | Pred AI |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Actual HUMAN | {results['confusion_matrix'][0][0]} | {results['confusion_matrix'][0][1]} |\n")
        f.write(f"| Actual AI | {results['confusion_matrix'][1][0]} | {results['confusion_matrix'][1][1]} |\n\n")
        f.write("### Cross-Validation (5-fold, trên Train)\n\n")
        f.write(f"- **Mean:** {results['cv_mean']:.4f}\n")
        f.write(f"- **Std:** {results['cv_std']:.4f}\n\n")
        f.write("### Top Features -> HUMAN (<=2022)\n\n")
        f.write("| Feature | Coefficient |\n")
        f.write("|---|---|\n")
        for feat in results["top_features_human"]:
            f.write(f"| {feat['feature']} | {feat['coefficient']:.4f} |\n")
        f.write("\n### Top Features → AI (≥2024)\n\n")
        f.write("| Feature | Coefficient |\n")
        f.write("|---|---|\n")
        for feat in results["top_features_ai"]:
            f.write(f"| {feat['feature']} | {feat['coefficient']:.4f} |\n")
    
    print(f"\n[*] Model saved to {output_path / 'ai_detection_lr_model.pkl'}")
    print(f"[*] Scaler saved to {output_path / 'ai_detection_lr_scaler.pkl'}")
    print(f"[*] Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Detection using Logistic Regression — Multi-Author Generalised Model"
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to cleaned dataset")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS, help="Min words per passage")
    parser.add_argument("--text", type=str, default=None, help="Text to classify (optional)")
    parser.add_argument("--file", type=str, default=None, help="File containing text to classify")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  AI DETECTION USING LOGISTIC REGRESSION")
    print("  Multi-Author Generalised Model (HUMAN vs AI)")
    print("  80% Train / 20% Test Split")
    print("=" * 60)
    
    # Nếu có text đầu vào → chỉ dự đoán, không train
    if args.text or args.file:
        # Load model đã train
        here = Path(__file__).parent
        model_path = here / "ai_detection_lr_model.pkl"
        scaler_path = here / "ai_detection_lr_scaler.pkl"
        features_path = here / "ai_detection_lr_features.pkl"
        
        if not model_path.exists():
            print("[!] Model chua duoc train. Chay lenh nay truoc:")
            print("   python ai_detection_lr.py")
            return
        
        import joblib
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        
        if args.text:
            input_text = args.text
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                input_text = f.read()
        
        print(f"\n[*] Analyzing text ({word_count(input_text)} words)...")
        result = predict_text(input_text, clf, scaler, feature_names, args.threshold)
        
        print(f"  KET QUA:")
        print(f"  {'='*40}")
        print(f"  Label:      {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"  Prob HUMAN: {result['prob_human']:.4f} ({result['prob_human']*100:.2f}%)")
        print(f"  Prob AI:    {result['prob_ai']:.4f} ({result['prob_ai']*100:.2f}%)")
        print(f"{'='*60}")
        
        if result['label'] == "AI (>=2024)":
            print(f"    Bai viet nay CO DAU HIEU duoc viet boi AI hoac co AI ho tro!")
        else:
            print(f"   Bai viet nay giong phong cach HUMAN (tu viet).")
        print(f"{'='*60}")
        return
    
    # Train model
    X, y, feature_names, authors = prepare_data(args.data_dir, args.min_words)
    
    clf, scaler, coef_df, results, X_test, y_test, y_pred, y_prob = train_and_evaluate(
        X, y, feature_names
    )
    
    if not args.no_save:
        save_model(clf, scaler, coef_df, feature_names, results)
    
    print(f"\n{'='*60}")
    print(f"   HOAN THANH!")
    print(f"{'='*60}")
    print(f"  Dung lenh sau de du doan text moi:")
    print(f'    python ai_detection_lr.py --text "Your text here..."')
    print(f"    python ai_detection_lr.py --file input.txt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
