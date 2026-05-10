"""
Demo: Verification - Kiểm tra 2 văn bản có cùng tác giả không
Dùng Logistic Regression đã train trên 35 authors

Cách dùng:
  python verify_demo.py                                    # text mẫu
  python verify_demo.py --text1 "text 1" --text2 "text 2"  # tự nhập
  python verify_demo.py --file1 a.txt --file2 b.txt        # từ file
"""
import argparse
import io
import os
import sys
import warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from src.features import extract_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import combinations
import random

# ── Config ────────────────────────────────────────────────────────────────
LARGE_CSV = os.path.join(ROOT, "SVM", "LesswrongLarge.csv")
METADATA_COLS = {"author", "passage_id", "_label"}
RANDOM_STATE = 42


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["author"] = df["author"].str.strip().str.lower()
    return df


def train_model(df):
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    authors = df["author"].values
    unique_authors = sorted(set(authors))

    # Train/test split
    random.seed(RANDOM_STATE)
    train_authors, _ = train_test_split(unique_authors, train_size=28, random_state=RANDOM_STATE)
    mask = np.isin(authors, train_authors)

    X_train = X[mask]
    y_train = authors[mask]

    # Generate pairs
    pos_pairs, neg_pairs = [], []
    for auth in train_authors:
        idx = np.where(y_train == auth)[0]
        if len(idx) < 2:
            continue
        for i, j in combinations(idx[:100], 2):  # limit
            f1, f2 = X_train[i], X_train[j]
            pos_pairs.append(np.concatenate([np.abs(f1 - f2), (f1 - f2) ** 2, f1 * f2]))

    num_pos = len(pos_pairs)
    for _ in range(num_pos):
        a1, a2 = random.sample(list(train_authors), 2)
        i = random.choice(np.where(y_train == a1)[0])
        j = random.choice(np.where(y_train == a2)[0])
        f1, f2 = X_train[i], X_train[j]
        neg_pairs.append(np.concatenate([np.abs(f1 - f2), (f1 - f2) ** 2, f1 * f2]))

    X_pairs = np.vstack((pos_pairs, neg_pairs))
    y_pairs = np.concatenate([np.ones(num_pos), np.zeros(num_pos)])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pairs)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_scaled, y_pairs)

    return model, scaler, feature_cols


def extract_feat(text, feature_cols):
    feats = extract_features(text)
    return np.array([feats.get(c, 0.0) for c in feature_cols])


def predict(text1, text2, model, scaler, feature_cols):
    f1 = extract_feat(text1, feature_cols)
    f2 = extract_feat(text2, feature_cols)

    pair = np.concatenate([np.abs(f1 - f2), (f1 - f2) ** 2, f1 * f2]).reshape(1, -1)
    pair_scaled = scaler.transform(pair)

    prob_same = model.predict_proba(pair_scaled)[0, 1]
    pred = model.predict(pair_scaled)[0]

    return pred, prob_same


def main():
    parser = argparse.ArgumentParser(description="Authorship Verification Demo")
    parser.add_argument("--text1", type=str, help="Văn bản thứ nhất")
    parser.add_argument("--text2", type=str, help="Văn bản thứ hai")
    parser.add_argument("--file1", type=str, help="File chứa văn bản thứ nhất")
    parser.add_argument("--file2", type=str, help="File chứa văn bản thứ hai")
    args = parser.parse_args()

    # Đọc input
    if args.file1 and args.file2:
        with open(args.file1, "r", encoding="utf-8") as f:
            text1 = f.read()
        with open(args.file2, "r", encoding="utf-8") as f:
            text2 = f.read()
    elif args.text1 and args.text2:
        text1, text2 = args.text1, args.text2
    else:
        text1 = (
            "I think this is not actually as complicated as people make it out to be. "
            "The fundamental issue is that we are not tracking the right variables. "
            "If you look at the data carefully, you will notice that the patterns are "
            "actually quite clear. But people keep insisting on looking at the wrong "
            "things and then drawing the wrong conclusions."
        )
        text2 = (
            "The basic idea is actually quite simple once you strip away all the jargon. "
            "People tend to overcomplicate things because they want to sound smart. "
            "But the truth is that most of these problems have straightforward solutions "
            "if you just take the time to think about them clearly and systematically."
        )
        print("(Dùng text mẫu)")

    print("\nĐang train model...")
    df = load_data(LARGE_CSV)
    model, scaler, feature_cols = train_model(df)

    print("Đang dự đoán...")
    pred, prob_same = predict(text1, text2, model, scaler, feature_cols)

    # In kết quả
    print("\n" + "=" * 60)
    print("VĂN BẢN 1:")
    print("-" * 60)
    print(text1[:300] + ("..." if len(text1) > 300 else ""))
    print("=" * 60)
    print("\nVĂN BẢN 2:")
    print("-" * 60)
    print(text2[:300] + ("..." if len(text2) > 300 else ""))
    print("=" * 60)

    print("\n" + "-" * 60)
    if pred == 1:
        print(f"  ✅ CÙNG TÁC GIẢ (confidence: {prob_same:.2%})")
    else:
        print(f"  ❌ KHÁC TÁC GIẢ (confidence: {1 - prob_same:.2%})")
    print("-" * 60)


if __name__ == "__main__":
    main()
