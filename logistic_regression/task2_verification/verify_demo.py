"""
Demo: Verification — check whether two texts share the same author.
Uses Logistic Regression trained on 35-author pairwise features.

Usage:
  python verify_demo.py                                    # sample texts
  python verify_demo.py --text1 "text 1" --text2 "text 2"  # inline
  python verify_demo.py --file1 a.txt --file2 b.txt        # from files

The model must first be trained by logistic_regression_verification.py,
which saves it to saved_model/.
"""
import argparse
import os
import sys

import joblib
import numpy as np

# ── bootstrap project path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_ROOT = os.path.join(_HERE, "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from _config import generate_pair_features, configure_stdout_utf8
from src.features import extract_features

configure_stdout_utf8()

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(_HERE, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "lr_verification.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.joblib")


def load_model():
    """Load pre-trained verification model from disk.

    Model is saved by logistic_regression_verification.py after training.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    print(f"[OK] Loaded model from: {MODEL_DIR}/")
    print(f"     {len(feature_cols)} features → {len(feature_cols) * 3} pair features")
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

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_DIR}")
        print("Run logistic_regression_verification.py first to train and save the model.")
        sys.exit(1)

    print("Loading model...")
    model, scaler, feature_cols = load_model()

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
