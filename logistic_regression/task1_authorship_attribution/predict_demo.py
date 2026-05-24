"""
Demo: Predict author for a single passage using Logistic Regression.

Usage:
  python predict_demo.py                              # sample text
  python predict_demo.py --text "Your text here..."    # inline text
  python predict_demo.py --file input.txt              # from file
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── bootstrap project path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_PROJECT_ROOT = os.path.join(_HERE, "..", "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from _config import (
    METADATA_COLS, PARAM_GRID_LR, RANDOM_STATE,
    remove_outliers, configure_stdout_utf8,
)
from src.features import extract_features

configure_stdout_utf8()

# ── config ─────────────────────────────────────────────────────────────────────
FULL_CSV = os.path.join(_HERE, "features_25authors.csv")


def train_model(use_clean: bool = True):
    """Train Logistic Regression on the full 25-author feature set."""
    df = pd.read_csv(FULL_CSV)
    if use_clean:
        df = remove_outliers(df, contamination=0.05)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["author"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gs = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID_LR, cv=3, scoring="accuracy",
        n_jobs=-1, refit=True,
    )
    gs.fit(X_scaled, y)

    model = gs.best_estimator_
    print(f"[OK] Model trained on {len(df)} passages, {len(feature_cols)} features")
    print(f"     Best params: {gs.best_params_}")
    print(f"     Authors: {list(le.classes_)}")

    return model, scaler, le, feature_cols


def predict(text: str, model, scaler, le, feature_cols: list[str]):
    """Extract features from text and predict author."""
    feats = extract_features(text)
    missing = [col for col in feature_cols if col not in feats]
    if missing:
        print(f"[WARNING] {len(missing)} features missing from input text — "
              f"filled with 0.0: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    feat_vec = np.array([feats.get(col, 0.0) for col in feature_cols]).reshape(1, -1)
    X_scaled = scaler.transform(feat_vec)

    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]
    author = le.inverse_transform([pred_class])[0]
    probas = {name: round(float(p), 4) for name, p in zip(le.classes_, pred_proba)}
    return author, probas


def print_result(text: str, author: str, probas: dict):
    """Pretty-print prediction result with bar chart."""
    print("\n" + "=" * 60)
    print("INPUT TEXT:")
    print("-" * 60)
    display_text = text[:500] + "..." if len(text) > 500 else text
    print(display_text)
    print("=" * 60)

    print("\nPREDICTION RESULT:")
    print("-" * 60)

    sorted_authors = sorted(probas.items(), key=lambda x: x[1], reverse=True)
    for i, (auth, prob) in enumerate(sorted_authors):
        bar_len = int(prob * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        marker = "  <-- Predicted" if i == 0 else ""
        print(f"  {auth:25s} |{bar}| {prob:>6.2%}{marker}")

    print("-" * 60)

    top_auth, top_prob = sorted_authors[0]
    second_prob = sorted_authors[1][1] if len(sorted_authors) > 1 else 0
    margin = top_prob - second_prob

    print(f"\n=> Conclusion: **{top_auth}** (confidence: {top_prob:.2%})")
    if margin > 0.5:
        print("   Level: Very high confidence")
    elif margin > 0.2:
        print("   Level: Fairly confident")
    else:
        print("   Level: Narrow margin — borderline case")


def main():
    parser = argparse.ArgumentParser(description="Predict author with Logistic Regression")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="File containing text")
    parser.add_argument("--with-outliers", action="store_true", help="Use model trained with outliers")
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
        print("(Using sample text — no --text or --file provided)")

    if len(text.split()) < 50:
        print("[WARNING] Input text is very short (<50 words). "
              "Stylometric features may be unreliable.")

    use_clean = not args.with_outliers
    print("\nTraining model on full dataset...")
    model, scaler, le, feature_cols = train_model(use_clean=use_clean)
    author, probas = predict(text, model, scaler, le, feature_cols)
    print_result(text, author, probas)


if __name__ == "__main__":
    main()
