"""
Demo: Du doan tac gia cho 1 doan text bat ky
Dung Logistic Regression da train (retrain tren full dataset)

Cach dung:
  python predict_demo.py                              # text mau
  python predict_demo.py --text "Your text here..."   # text tu nhap
  python predict_demo.py --file input.txt             # tu file
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Them project root vao path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from src.features import extract_features
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

# Set encoding cho stdout
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- Config ----------------------------------------------------------------
DATA_DIR  = _HERE
FULL_CSV  = os.path.join(DATA_DIR, "features_25authors.csv")
METADATA_COLS = {"author", "passage_id"}
RANDOM_STATE  = 42

PARAM_GRID = {
    "C":        [0.01, 0.1, 1, 10, 100],
    "penalty":  ["l2"],
    "solver":   ["lbfgs"],
    "max_iter": [1000, 2000, 5000],
}


def remove_outliers(df):
    """Remove outliers per author using Isolation Forest."""
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    kept = []
    for _, group in df.groupby("author"):
        clf = IsolationForest(contamination="auto", random_state=RANDOM_STATE)
        mask = clf.fit_predict(group[feature_cols].values) == 1
        kept.append(group[mask])
    return pd.concat(kept).reset_index(drop=True)


# --- Train model -----------------------------------------------------------
def train_model(use_clean=True):
    df = pd.read_csv(FULL_CSV)
    if use_clean:
        df = remove_outliers(df)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["author"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_clf = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)
    gs = GridSearchCV(base_clf, PARAM_GRID, cv=3, scoring="accuracy", n_jobs=1, refit=True)
    gs.fit(X_scaled, y)

    model = gs.best_estimator_

    print(f"[OK] Model trained on {len(df)} passages, {len(feature_cols)} features")
    print(f"     Best params: {gs.best_params_}")
    print(f"     Authors: {list(le.classes_)}")

    return model, scaler, le, feature_cols


# --- Predict ---------------------------------------------------------------
def predict(text, model, scaler, le, feature_cols):
    feats = extract_features(text)
    feat_dict = {col: feats.get(col, 0.0) for col in feature_cols}
    X_new = pd.DataFrame([feat_dict])[feature_cols].values
    X_new_scaled = scaler.transform(X_new)

    pred_class = model.predict(X_new_scaled)[0]
    pred_proba = model.predict_proba(X_new_scaled)[0]

    author = le.inverse_transform([pred_class])[0]
    probas = {name: round(float(p), 4) for name, p in zip(le.classes_, pred_proba)}

    return author, probas


# --- In ket qua ------------------------------------------------------------
def print_result(text, author, probas):
    print("\n" + "=" * 60)
    print("VAN BAN DAU VAO:")
    print("-" * 60)
    display_text = text[:500] + "..." if len(text) > 500 else text
    print(display_text)
    print("=" * 60)

    print("\nKET QUA DU DOAN TAC GIA:")
    print("-" * 60)

    sorted_authors = sorted(probas.items(), key=lambda x: x[1], reverse=True)

    for i, (auth, prob) in enumerate(sorted_authors):
        bar_len = int(prob * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        marker = "  <-- Du doan" if i == 0 else ""
        print(f"  {auth:25s} |{bar}| {prob:>6.2%}{marker}")

    print("-" * 60)

    top_auth, top_prob = sorted_authors[0]
    second_prob = sorted_authors[1][1] if len(sorted_authors) > 1 else 0
    margin = top_prob - second_prob

    print(f"\n=> Ket luan: **{top_auth}** (confidence: {top_prob:.2%})")

    if margin > 0.5:
        print("   Muc do: Rat chac chan")
    elif margin > 0.2:
        print("   Muc do: Kha chac chan")
    else:
        print("   Muc do: Ranh gioi mong manh")


# --- Main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Du doan tac gia voi Logistic Regression")
    parser.add_argument("--text", type=str, help="Doan text can du doan")
    parser.add_argument("--file", type=str, help="File .txt chua text can du doan")
    parser.add_argument("--with-outliers", action="store_true", help="Dung model co outliers")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = (
            "I think this is not actually as complicated as people make it out to be. "
            "The fundamental issue is that we are not tracking the right variables. "
            "If you look at the data carefully, you will notice that the patterns are "
            "actually quite clear. But people keep insisting on looking at the wrong "
            "things and then drawing the wrong conclusions. This is a pattern that "
            "repeats itself across many different domains and it's frankly frustrating "
            "to watch. The solution is obvious once you stop and think about it."
        )
        print("(Dung text mau vi khong co input --text hoac --file)")

    use_clean = not args.with_outliers
    print("\nDang train model tren full dataset...")
    model, scaler, le, feature_cols = train_model(use_clean=use_clean)
    author, probas = predict(text, model, scaler, le, feature_cols)
    print_result(text, author, probas)


if __name__ == "__main__":
    main()
