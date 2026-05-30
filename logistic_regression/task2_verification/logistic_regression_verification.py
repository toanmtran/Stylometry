"""
Authorship Verification — Binary Classification via Pairwise Features

Given two texts, determine whether they were written by the same author.

Approach:
  1. Load feature CSV with author labels
  2. Generate positive pairs (same author, different passages) and
     negative pairs (different authors)
  3. For each pair, compute: [abs_diff, sq_diff, mult_feat] = 3 * n_features
  4. Train Logistic Regression binary classifier: 1 = same author, 0 = different authors
  5. Evaluate on held-out test AUTHORS (not passages) for honest generalisation

NOTE — Negative-pair groups:
  A negative pair involves TWO authors. GroupKFold cannot track both, so we
  split authors into train/test FIRST, and use StratifiedKFold for CV within
  the training set. The final held-out test set contains only unseen authors
  —no author appears in both train and test.
"""
import argparse
import os
import random
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
)

# ── bootstrap project path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in __import__("sys").path:
    __import__("sys").path.insert(0, _PARENT)

from _config import RANDOM_STATE, generate_pair_features, PARAM_GRID_LR_SIMPLE
from _config import LESSWRONG_LARGE as LARGE_CSV

# ── configuration ──────────────────────────────────────────────────────────────
METADATA_COLS = {"author", "passage_id", "_label"}
MAX_PAIRS_PER_AUTHOR = 500


def load_data(csv_path: str) -> pd.DataFrame:
    """Load feature CSV and normalise author names."""
    df = pd.read_csv(csv_path)
    df["author"] = df["author"].str.strip().str.lower()
    print(f"  Loaded {len(df)} passages, {df['author'].nunique()} authors")
    return df


def generate_pairs(
    features: np.ndarray,
    authors: np.ndarray,
    selected_authors: list[str],
    max_pairs_per_auth: int = MAX_PAIRS_PER_AUTHOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced positive and negative pairs.

    Pair feature = concat(abs_diff, sq_diff, mult) → 3 * n_features.

    Returns X (pair features) and y (1 = same author, 0 = different).
    """
    pos_pairs = []
    neg_pairs = []

    # 1. Positive pairs: same author, different passages
    for auth in selected_authors:
        idx = np.where(authors == auth)[0]
        if len(idx) < 2:
            continue

        all_combos = list(combinations(idx, 2))
        if len(all_combos) > max_pairs_per_auth:
            all_combos = random.sample(all_combos, max_pairs_per_auth)

        for i, j in all_combos:
            pos_pairs.append(generate_pair_features(features[i], features[j]))

    # 2. Negative pairs: different authors (match positive count)
    num_pos = len(pos_pairs)
    selected_list = list(selected_authors)

    if len(selected_list) >= 2:
        for _ in range(num_pos):
            a1, a2 = random.sample(selected_list, 2)
            i = random.choice(np.where(authors == a1)[0])
            j = random.choice(np.where(authors == a2)[0])
            neg_pairs.append(generate_pair_features(features[i], features[j]))

    if len(neg_pairs) == 0:
        return np.array(pos_pairs), np.ones(num_pos)

    X = np.vstack((pos_pairs, neg_pairs))
    y = np.concatenate([np.ones(num_pos), np.zeros(len(neg_pairs))])
    return X, y


def run_verification(df: pd.DataFrame, case_label: str) -> dict:
    """Run authorship verification with Logistic Regression + hyperparameter tuning."""
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X_feat = df[feature_cols].values
    authors = df["author"].values
    unique_authors = sorted(set(authors))

    print(f"\n{'='*62}")
    print(f"  Authorship Verification | {case_label} | Model: Logistic Regression")
    print(f"  {len(df)} passages | {len(feature_cols)} features | {len(unique_authors)} authors")
    print(f"{'='*62}")

    # Split authors: 80% train, 20% test (by author, not passage!)
    random.seed(RANDOM_STATE)
    n_train = max(2, int(len(unique_authors) * 0.8))
    if n_train >= len(unique_authors):
        n_train = len(unique_authors) - 1
    train_authors, test_authors = train_test_split(
        unique_authors, train_size=n_train, random_state=RANDOM_STATE,
    )

    train_mask = np.isin(authors, train_authors)
    test_mask = np.isin(authors, test_authors)

    print(f"  Train authors: {len(train_authors)}, Test authors: {len(test_authors)}")

    # Generate pairs
    print("  Generating pairs...")
    X_train, y_train = generate_pairs(
        X_feat[train_mask], authors[train_mask], train_authors,
    )
    X_test, y_test = generate_pairs(
        X_feat[test_mask], authors[test_mask], test_authors,
    )

    print(f"  Train pairs: {len(y_train)} ({int(y_train.sum())} positive, "
          f"{int(len(y_train) - y_train.sum())} negative)")
    print(f"  Test pairs:  {len(y_test)} ({int(y_test.sum())} positive, "
          f"{int(len(y_test) - y_test.sum())} negative)")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning via GridSearchCV
    print("  Tuning hyperparameters via GridSearchCV (3-fold)...")
    gs = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID_LR_SIMPLE,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1, refit=True,
    )
    gs.fit(X_train_scaled, y_train)
    model = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  CV AUC: {gs.best_score_:.4f}")

    # Cross-validation accuracy for reporting
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=min(5, len(set(y_train))), shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
    )
    print(f"  CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Evaluate on held-out test authors
    print("  Evaluating on held-out test authors...")
    y_pred = model.predict(X_test_scaled)

    n_classes_test = len(set(y_test))
    if n_classes_test < 2:
        print(f"  [!] Test set has only {n_classes_test} class(es) — skipping some metrics")
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, auc = 0.0, 0.0, 0.0, 0.0
    else:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # ── Save model for verify_demo.py ─────────────────────────────────────
    save_dir = os.path.join(_HERE, "saved_model")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "lr_verification.joblib"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(save_dir, "feature_cols.joblib"))
    print(f"\n  Saved model to: {save_dir}/")

    return {
        "case": case_label,
        "model": "Logistic Regression",
        "n_passages": len(df),
        "n_authors": len(unique_authors),
        "best_params": gs.best_params_,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "test_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="Authorship Verification — Logistic Regression")
    parser.add_argument("--output", default="verification_results.md", help="Output markdown file")
    args = parser.parse_args()

    all_results = []

    print("\n" + "=" * 62)
    print("  CASE: 35 Authors (LesswrongLarge.csv) — Logistic Regression")
    print("=" * 62)
    df_35 = load_data(LARGE_CSV)
    r = run_verification(df_35, "35 Authors")
    all_results.append(r)

    print(f"\nSaving report to {args.output}...")
    md = "# Authorship Verification Results\n\n"

    md += "## Summary Table\n\n"
    md += "| Case | Model | Passages | Authors | Best Params | CV Acc | Test Acc | Precision | Recall | F1 | AUC |\n"
    md += "|------|-------|----------|---------|-------------|--------|----------|-----------|--------|----|-----|\n"
    for r in all_results:
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(r["best_params"].items()))
        md += (f"| {r['case']} | {r['model']} | {r['n_passages']} | {r['n_authors']} "
               f"| `{params_str}` "
               f"| {r['cv_accuracy_mean']:.4f}+/-{r['cv_accuracy_std']:.4f} "
               f"| {r['test_accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} "
               f"| {r['f1']:.4f} | {r['auc']:.4f} |\n")

    md += "\n## Detailed Results\n\n"
    for r in all_results:
        md += f"### {r['case']} — {r['model']}\n\n"
        md += f"- **Passages:** {r['n_passages']}\n"
        md += f"- **Authors:** {r['n_authors']}\n"
        md += f"- **Best Params:** `{params_str}`\n"
        md += f"- **CV Accuracy:** {r['cv_accuracy_mean']:.4f} +/- {r['cv_accuracy_std']:.4f}\n"
        md += f"- **Test Accuracy:** {r['test_accuracy']:.4f}\n"
        md += f"- **Precision:** {r['precision']:.4f}\n"
        md += f"- **Recall:** {r['recall']:.4f}\n"
        md += f"- **F1:** {r['f1']:.4f}\n"
        md += f"- **AUC:** {r['auc']:.4f}\n\n"

        cm = r["confusion_matrix"]
        md += "**Confusion Matrix:**\n\n"
        md += "| | Predicted Diff | Predicted Same |\n"
        md += "|---|---:|---:|\n"
        md += f"| Actual Different | {cm[0,0]} | {cm[0,1]} |\n"
        md += f"| Actual Same | {cm[1,0]} | {cm[1,1]} |\n\n"

    out_path = os.path.join(_HERE, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
