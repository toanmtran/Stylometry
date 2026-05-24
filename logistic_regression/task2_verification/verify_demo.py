"""
Demo: Verification — check whether two texts share the same author.
Uses Logistic Regression trained on 35-author pairwise features.

Usage:
  python verify_demo.py                                    # sample texts
  python verify_demo.py --text1 "text 1" --text2 "text 2"  # inline
  python verify_demo.py --file1 a.txt --file2 b.txt        # from files
"""
import argparse
import os
import random
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ── bootstrap project path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_ROOT = os.path.join(_HERE, "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from _config import (
    RANDOM_STATE, configure_stdout_utf8,
    generate_pair_features, PARAM_GRID_LR_SIMPLE,
)
from _config import LESSWRONG_LARGE as LARGE_CSV
from src.features import extract_features

configure_stdout_utf8()

# ── Config ─────────────────────────────────────────────────────────────────────
METADATA_COLS = {"author", "passage_id", "_label"}
TRAIN_FRAC = 0.85  # fraction of authors used for training


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["author"] = df["author"].str.strip().str.lower()
    return df


def train_model(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    authors = df["author"].values
    unique_authors = sorted(set(authors))

    # Author-level train/test split
    random.seed(RANDOM_STATE)
    n_train = max(2, int(len(unique_authors) * TRAIN_FRAC))
    train_authors, _ = train_test_split(
        unique_authors, train_size=n_train, random_state=RANDOM_STATE,
    )
    mask = np.isin(authors, train_authors)
    X_train = X[mask]
    y_train = authors[mask]

    # Generate pairs
    pos_pairs, neg_pairs = [], []
    for auth in train_authors:
        idx = np.where(y_train == auth)[0]
        if len(idx) < 2:
            continue
        for i, j in combinations(idx, 2):
            pos_pairs.append(generate_pair_features(X_train[i], X_train[j]))

    num_pos = len(pos_pairs)
    selected_list = list(train_authors)
    for _ in range(num_pos):
        a1, a2 = random.sample(selected_list, 2)
        i = random.choice(np.where(y_train == a1)[0])
        j = random.choice(np.where(y_train == a2)[0])
        neg_pairs.append(generate_pair_features(X_train[i], X_train[j]))

    X_pairs = np.vstack((pos_pairs, neg_pairs))
    y_pairs = np.concatenate([np.ones(num_pos), np.zeros(num_pos)])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pairs)

    # Tune hyperparameters
    gs = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID_LR_SIMPLE,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1, refit=True,
    )
    gs.fit(X_scaled, y_pairs)
    model = gs.best_estimator_
    print(f"[OK] Trained on {len(train_authors)} authors, {len(y_pairs)} pairs")
    print(f"     Best params: {gs.best_params_}, CV AUC: {gs.best_score_:.4f}")

    return model, scaler, feature_cols


def extract_feat(text: str, feature_cols: list[str]) -> np.ndarray:
    feats = extract_features(text)
    return np.array([feats.get(c, 0.0) for c in feature_cols])


def predict(text1: str, text2: str, model, scaler, feature_cols: list[str]):
    f1 = extract_feat(text1, feature_cols)
    f2 = extract_feat(text2, feature_cols)
    pair = generate_pair_features(f1, f2).reshape(1, -1)
    pair_scaled = scaler.transform(pair)
    prob_same = model.predict_proba(pair_scaled)[0, 1]
    pred = model.predict(pair_scaled)[0]
    return pred, prob_same


def main():
    parser = argparse.ArgumentParser(description="Authorship Verification Demo")
    parser.add_argument("--text1", type=str, help="First text")
    parser.add_argument("--text2", type=str, help="Second text")
    parser.add_argument("--file1", type=str, help="File containing first text")
    parser.add_argument("--file2", type=str, help="File containing second text")
    args = parser.parse_args()

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
        print("(Using sample texts)")

    print("\nTraining model...")
    df = load_data(LARGE_CSV)
    model, scaler, feature_cols = train_model(df)

    print("Predicting...")
    pred, prob_same = predict(text1, text2, model, scaler, feature_cols)

    print("\n" + "=" * 60)
    print("TEXT 1:")
    print("-" * 60)
    print(text1[:300] + ("..." if len(text1) > 300 else ""))
    print("=" * 60)
    print("\nTEXT 2:")
    print("-" * 60)
    print(text2[:300] + ("..." if len(text2) > 300 else ""))
    print("=" * 60)

    print("\n" + "-" * 60)
    if pred == 1:
        print(f"  SAME AUTHOR (confidence: {prob_same:.2%})")
    else:
        print(f"  DIFFERENT AUTHORS (confidence: {1 - prob_same:.2%})")
    print("-" * 60)


if __name__ == "__main__":
    main()
