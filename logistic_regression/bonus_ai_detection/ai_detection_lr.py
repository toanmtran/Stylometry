"""
AI Detection using Logistic Regression — Multi-Author Generalised Model

Idea:
  - Group ALL author posts into two classes:
      0 = HUMAN (<=2022) — before AI was widespread
      1 = AI    (>=2024) — AI widely available
  - Train a single Logistic Regression to classify
  - Works for ANY new text, no author info needed
  - 80% train / 20% test split for objective evaluation

Usage:
    python ai_detection_lr.py                              # train + save model
    python ai_detection_lr.py --text "Your sample text..."  # classify text
    python ai_detection_lr.py --file input.txt              # classify from file
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))           # logistic_regression/  (for _config)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))      # project root         (for src)

from src.features import extract_features
from src.preprocess import (
    _sanitize_json,
    canonicalize_typography,
    strip_lesswrong_boilerplate,
    word_count,
    truncate_to_words,
)
from _config import RANDOM_STATE

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_WORDS = 500
TEST_SIZE = 0.2
DATA_DIR = os.path.join(_HERE, "..", "..", "dataset", "lesswrong_regular", "cleaned")

PARAM_GRID = {
    "C":        [0.01, 0.1, 1, 10, 100],
    "penalty":  ["l2"],
    "solver":   ["lbfgs"],
    "max_iter": [1000, 2000],
    "class_weight": ["balanced", None],
}


def classify_era(date_str: str) -> int | None:
    """
    0 = HUMAN (<=2022), 1 = AI (>=2024), None = 2023 (ambiguous — skip).
    """
    if not date_str or len(date_str) < 4:
        return None
    try:
        year = int(date_str[:4])
    except ValueError:
        return None
    if year <= 2022:
        return 0
    elif year >= 2024:
        return 1
    return None


def prepare_data(data_dir: str, min_words: int = MIN_WORDS):
    """Load data, classify HUMAN/AI by date, extract features."""
    print(f"[*] Loading passages from {data_dir}...")
    folder = Path(data_dir)
    raw_items = []
    for fp in sorted(folder.glob("*.json")):
        raw_text = fp.read_text(encoding="utf-8-sig")
        items = json.loads(_sanitize_json(raw_text))
        for item in items:
            raw_items.append({
                "author": item.get("author", "Unknown").strip().lower(),
                "text": item.get("text", ""),
                "date": item.get("date", ""),
            })
    print(f"   Total passages loaded: {len(raw_items)}")

    texts, labels = [], []
    for item in raw_items:
        era = classify_era(item["date"])
        if era is None:
            continue

        text = canonicalize_typography(item["text"])
        text = strip_lesswrong_boilerplate(text)
        if word_count(text) < min_words:
            continue

        texts.append(text)
        labels.append(era)

    print(f"\n[Data summary]:")
    human_count = sum(1 for l in labels if l == 0)
    ai_count = sum(1 for l in labels if l == 1)
    print(f"   HUMAN (<=2022): {human_count} passages")
    print(f"   AI (>=2024):    {ai_count} passages")

    # Balance classes
    min_count = min(human_count, ai_count)
    human_idx = [i for i, l in enumerate(labels) if l == 0]
    ai_idx = [i for i, l in enumerate(labels) if l == 1]

    np.random.seed(RANDOM_STATE)
    sampled_human = np.random.choice(human_idx, min_count, replace=False)
    sampled_ai = np.random.choice(ai_idx, min_count, replace=False)
    selected = sorted(list(sampled_human) + list(sampled_ai))

    texts_balanced = [texts[i] for i in selected]
    labels_balanced = [labels[i] for i in selected]

    print(f"   After balancing: {len(texts_balanced)} passages")
    print(f"     HUMAN: {sum(1 for l in labels_balanced if l == 0)}")
    print(f"     AI:    {sum(1 for l in labels_balanced if l == 1)}")

    # Truncate to same word length
    min_wc = min(word_count(t) for t in texts_balanced)
    texts_balanced = [truncate_to_words(t, min_wc) for t in texts_balanced]

    # Extract features
    print(f"\n[Extracting stylometric features...]")
    feature_list, feature_names = [], None
    for i, t in enumerate(texts_balanced):
        feats = extract_features(t)
        if feature_names is None:
            feature_names = list(feats.keys())
        feature_list.append(list(feats.values()))
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(texts_balanced)} passages...")

    X = np.array(feature_list)
    y = np.array(labels_balanced)
    return X, y, feature_names


def train_and_evaluate(X, y, feature_names):
    """Train Logistic Regression with 80/20 split + hyperparameter tuning."""
    print(f"\n{'='*60}")
    print(f"  DATA SPLIT: 80% TRAIN / 20% TEST")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"   Train: {len(X_train)} passages ({sum(y_train==0)} HUMAN + {sum(y_train==1)} AI)")
    print(f"   Test:  {len(X_test)} passages ({sum(y_test==0)} HUMAN + {sum(y_test==1)} AI)")

    # Scale (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning
    print(f"\nTuning hyperparameters via GridSearchCV...")
    print(f"   Features: {X.shape[1]}")

    gs = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1, refit=True,
    )
    gs.fit(X_train_scaled, y_train)
    clf = gs.best_estimator_
    print(f"   Best params: {gs.best_params_}")
    print(f"   CV AUC: {gs.best_score_:.4f}")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"  RESULTS ON TEST SET (20%)")
    print(f"{'='*60}")

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   AUC-ROC:   {test_auc:.4f}")
    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  +----------------------+")
    print(f"                  | Pred HUMAN | Pred AI  |")
    print(f"   ---------------+------------+----------+")
    print(f"   Actual HUMAN   |     {cm[0,0]:3d}      |    {cm[0,1]:3d}     |")
    print(f"   ---------------+------------+----------+")
    print(f"   Actual AI      |     {cm[1,0]:3d}      |    {cm[1,1]:3d}     |")
    print(f"                  +----------------------+")
    print(f"\n{classification_report(y_test, y_pred, target_names=['HUMAN (<=2022)', 'AI (>=2024)'])}")

    # Cross-validation for stability check
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    print(f"\nCROSS-VALIDATION (5-fold, on train only)")
    print(f"   CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Feature importance
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    print(f"\n{'='*60}")
    print(f"  TOP 15 DISCRIMINATIVE FEATURES (HUMAN vs AI)")
    print(f"{'='*60}")
    print(f"   {'Feature':25s} {'Coef':>10s} {'Direction':>20s}")
    print(f"   {'-'*25} {'-'*10} {'-'*20}")
    for _, row in coef_df.head(15).iterrows():
        direction = "AI (>=2024)" if row["coefficient"] > 0 else "HUMAN (<=2022)"
        print(f"   {row['feature']:25s} {row['coefficient']:>+10.4f} {direction:>20s}")

    results = {
        "test_accuracy": test_acc,
        "test_auc_roc": test_auc,
        "best_params": gs.best_params_,
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
    return clf, scaler, coef_df, results


def predict_text(text: str, clf, scaler, feature_names: list[str]):
    """Predict whether a text is HUMAN or AI."""
    text = canonicalize_typography(text)
    text = strip_lesswrong_boilerplate(text)

    if word_count(text) < MIN_WORDS:
        print(f"[!] Warning: Text only has {word_count(text)} words (minimum {MIN_WORDS})")

    feats = extract_features(text)
    X = np.array([[feats.get(f, 0.0) for f in feature_names]])
    X_scaled = scaler.transform(X)

    prob = clf.predict_proba(X_scaled)[0]
    pred = clf.predict(X_scaled)[0]

    return {
        "label": "AI (>=2024)" if pred == 1 else "HUMAN (<=2022)",
        "prob_human": prob[0],
        "prob_ai": prob[1],
        "confidence": max(prob),
    }


def save_model(clf, scaler, coef_df, feature_names, results, output_dir=None):
    if output_dir is None:
        output_dir = _HERE
    """Save model artifacts to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, output_path / "ai_detection_lr_model.pkl")
    joblib.dump(scaler, output_path / "ai_detection_lr_scaler.pkl")
    joblib.dump(feature_names, output_path / "ai_detection_lr_features.pkl")
    coef_df.to_csv(output_path / "ai_detection_lr_feature_importance.csv", index=False)

    # Report
    report_path = output_path / "ai_detection_lr_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AI Detection Report — Logistic Regression\n\n")
        f.write(f"## Best Params\n\n")
        f.write(f"```\n{results.get('best_params', {})}\n```\n\n")
        f.write("## Results on Test Set (20%)\n\n")
        f.write(f"- **Accuracy:** {results['test_accuracy']:.4f}\n")
        f.write(f"- **AUC-ROC:** {results['test_auc_roc']:.4f}\n\n")
        f.write("### Confusion Matrix\n\n")
        f.write("| | Pred HUMAN | Pred AI |\n|---|---|---|\n")
        f.write(f"| Actual HUMAN | {results['confusion_matrix'][0][0]} | {results['confusion_matrix'][0][1]} |\n")
        f.write(f"| Actual AI | {results['confusion_matrix'][1][0]} | {results['confusion_matrix'][1][1]} |\n\n")
        f.write("### Cross-Validation (5-fold, on Train)\n\n")
        f.write(f"- **Mean:** {results['cv_mean']:.4f}\n")
        f.write(f"- **Std:** {results['cv_std']:.4f}\n\n")
        f.write("### Top Features -> HUMAN (<=2022)\n\n")
        f.write("| Feature | Coefficient |\n|---|---|\n")
        for feat in results["top_features_human"]:
            f.write(f"| {feat['feature']} | {feat['coefficient']:.4f} |\n")
        f.write("\n### Top Features -> AI (>=2024)\n\n")
        f.write("| Feature | Coefficient |\n|---|---|\n")
        for feat in results["top_features_ai"]:
            f.write(f"| {feat['feature']} | {feat['coefficient']:.4f} |\n")

    print(f"\n[*] Model saved to {output_path / 'ai_detection_lr_model.pkl'}")
    print(f"[*] Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Detection using Logistic Regression — Multi-Author Generalised Model"
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to cleaned dataset")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS, help="Min words per passage")
    parser.add_argument("--text", type=str, default=None, help="Text to classify")
    parser.add_argument("--file", type=str, default=None, help="File containing text to classify")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI DETECTION USING LOGISTIC REGRESSION")
    print("  Multi-Author Generalised Model (HUMAN vs AI)")
    print("  80% Train / 20% Test Split with Hyperparameter Tuning")
    print("=" * 60)

    # Inference-only mode
    if args.text or args.file:
        here = Path(__file__).parent
        model_path = here / "ai_detection_lr_model.pkl"
        scaler_path = here / "ai_detection_lr_scaler.pkl"
        features_path = here / "ai_detection_lr_features.pkl"

        if not model_path.exists():
            print("[!] Model not yet trained. Run first:")
            print("    python ai_detection_lr.py")
            return

        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)

        if args.text:
            input_text = args.text
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                input_text = f.read()

        print(f"\n[*] Analyzing text ({word_count(input_text)} words)...")
        result = predict_text(input_text, clf, scaler, feature_names)

        print(f"\n  RESULT:")
        print(f"  {'='*40}")
        print(f"  Label:      {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"  Prob HUMAN: {result['prob_human']:.4f} ({result['prob_human']*100:.2f}%)")
        print(f"  Prob AI:    {result['prob_ai']:.4f} ({result['prob_ai']*100:.2f}%)")
        print(f"{'='*60}")
        if result['label'] == "AI (>=2024)":
            print(f"   This text shows signs of AI authorship or AI assistance!")
        else:
            print(f"   This text resembles HUMAN writing style.")
        print(f"{'='*60}")
        return

    # Training mode
    X, y, feature_names = prepare_data(args.data_dir, args.min_words)
    clf, scaler, coef_df, results = train_and_evaluate(X, y, feature_names)

    if not args.no_save:
        save_model(clf, scaler, coef_df, feature_names, results)

    print(f"\n{'='*60}")
    print(f"   DONE!")
    print(f"{'='*60}")
    print(f"  To classify new text:")
    print(f'    python ai_detection_lr.py --text "Your text here..."')
    print(f"    python ai_detection_lr.py --file input.txt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
