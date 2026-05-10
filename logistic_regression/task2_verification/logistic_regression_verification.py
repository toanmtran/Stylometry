"""
Authorship Verification — Binary Classification via Pairwise Features

Given two texts, determine whether they were written by the same author.

Approach:
  1. Load feature CSV with author labels
  2. Generate positive pairs (same author, different passages) and
     negative pairs (different authors)
  3. For each pair, compute: [abs_diff, sq_diff, mult_feat] = 3 * n_features
  4. Train Logistic Regression binary classifier: 1 = same author, 0 = different authors
  5. Use GroupKFold to prevent author-level data leakage

Usage:
    python logistic_regression_verification.py
"""
import argparse
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
)


# ── paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.join(_HERE, "..", "..")

LARGE_CSV  = os.path.join(ROOT, "SVM", "LesswrongLarge.csv")  # 35 authors

# ── configuration ──────────────────────────────────────────────────────────
METADATA_COLS = {"author", "passage_id", "_label"}
RANDOM_STATE  = 42
MAX_PAIRS_PER_AUTHOR = 500  # Limit to prevent class imbalance from exploding


def load_data(csv_path: str) -> pd.DataFrame:
    """Load feature CSV."""
    df = pd.read_csv(csv_path)
    # Normalize author names
    df["author"] = df["author"].str.strip().str.lower()
    print(f"  Loaded {len(df)} passages, {df['author'].nunique()} authors")
    return df


def generate_pairs(
    features: np.ndarray,
    authors: np.ndarray,
    selected_authors: list[str],
    max_pairs_per_auth: int = MAX_PAIRS_PER_AUTHOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate positive and negative pairs.

    Positive: same author, different passages
    Negative: different authors, one passage each

    Pair feature vector = concatenation of:
      abs_diff(A, B), sq_diff(A, B), mult(A, B)

    Returns:
        X: (n_pairs, 3 * n_features) array
        y: (n_pairs,) labels (1 = same, 0 = different)
        groups: (n_pairs,) author IDs for GroupKFold
    """
    author_to_id = {auth: idx for idx, auth in enumerate(selected_authors)}
    n_features = features.shape[1]

    pos_pairs, pos_groups = [], []
    neg_pairs, neg_groups = [], []

    # 1. Positive pairs: same author
    for auth in selected_authors:
        idx = np.where(authors == auth)[0]
        if len(idx) < 2:
            continue

        all_combos = list(combinations(idx, 2))
        if len(all_combos) > max_pairs_per_auth:
            all_combos = random.sample(all_combos, max_pairs_per_auth)

        for i, j in all_combos:
            f1, f2 = features[i], features[j]
            combined = np.concatenate([
                np.abs(f1 - f2),
                (f1 - f2) ** 2,
                f1 * f2,
            ])
            pos_pairs.append(combined)
            pos_groups.append(author_to_id[auth])

    # 2. Negative pairs: different authors (same count as positives)
    num_pos = len(pos_pairs)
    selected_list = list(selected_authors)

    if len(selected_list) >= 2:
        for _ in range(num_pos):
            a1, a2 = random.sample(selected_list, 2)
            idx1 = random.choice(np.where(authors == a1)[0])
            idx2 = random.choice(np.where(authors == a2)[0])

            f1, f2 = features[idx1], features[idx2]
            combined = np.concatenate([
                np.abs(f1 - f2),
                (f1 - f2) ** 2,
                f1 * f2,
            ])
            neg_pairs.append(combined)
            neg_groups.append(author_to_id[a1])

    if len(neg_pairs) == 0:
        X = np.array(pos_pairs)
        y = np.ones(num_pos)
        groups = np.array(pos_groups)
    else:
        X = np.vstack((pos_pairs, neg_pairs))
        y = np.concatenate([np.ones(num_pos), np.zeros(len(neg_pairs))])
        groups = np.array(pos_groups + neg_groups)

    return X, y, groups


def run_verification(
    df: pd.DataFrame,
    case_label: str,
) -> dict:
    """Run authorship verification with Logistic Regression."""
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X_feat = df[feature_cols].values
    authors = df["author"].values
    unique_authors = sorted(set(authors))

    print(f"\n{'='*62}")
    print(f"  Authorship Verification | {case_label} | Model: Logistic Regression")
    print(f"  {len(df)} passages | {len(feature_cols)} features | {len(unique_authors)} authors")
    print(f"{'='*62}")

    # Split authors: 80% train, 20% test (by author, not by passage!)
    random.seed(RANDOM_STATE)
    n_train = max(2, int(len(unique_authors) * 0.8))
    if n_train >= len(unique_authors):
        n_train = len(unique_authors) - 1
    train_authors, test_authors = train_test_split(
        unique_authors, train_size=n_train,
        random_state=RANDOM_STATE,
    )

    train_mask = np.isin(authors, train_authors)
    test_mask = np.isin(authors, test_authors)

    X_train_feat = X_feat[train_mask]
    y_train_auth = authors[train_mask]
    X_test_feat = X_feat[test_mask]
    y_test_auth = authors[test_mask]

    print(f"  Train authors: {len(train_authors)}, Test authors: {len(test_authors)}")

    # Generate pairs
    print(f"  Generating pairs...")
    X_train, y_train, groups_train = generate_pairs(
        X_train_feat, y_train_auth, train_authors,
    )
    X_test, y_test, _ = generate_pairs(
        X_test_feat, y_test_auth, test_authors,
    )

    print(f"  Train pairs: {len(y_train)} ({int(y_train.sum())} positive, "
          f"{int(len(y_train) - y_train.sum())} negative)")
    print(f"  Test pairs:  {len(y_test)} ({int(y_test.sum())} positive, "
          f"{int(len(y_test) - y_test.sum())} negative)")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)

    # Cross-validation with GroupKFold
    n_groups = len(set(groups_train))
    n_splits = min(5, n_groups)
    print(f"  Running GroupKFold CV ({n_splits} folds, {n_groups} groups)...")
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=gkf,
                                groups=groups_train, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Train on full training set, evaluate on test
    print(f"  Training on full train set, evaluating on held-out test authors...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # If test set has only one class, some metrics are meaningless
    n_classes_test = len(set(y_test))
    if n_classes_test < 2:
        print(f"  [!] Test set has only {n_classes_test} class(es) — skipping some metrics")
        acc = accuracy_score(y_test, y_pred)
        prec = 0.0
        rec = 0.0
        f1 = 0.0
        auc = 0.0
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

    return {
        "case": case_label,
        "model": "Logistic Regression",
        "n_passages": len(df),
        "n_authors": len(unique_authors),
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
    parser.add_argument("--output", default="verification_results.md",
                        help="Output markdown file")
    args = parser.parse_args()

    all_results = []

    # ── 35 Authors (LesswrongLarge.csv) — Logistic Regression ───────────
    print("\n" + "=" * 62)
    print("  CASE: 35 Authors (LesswrongLarge.csv) — Logistic Regression")
    print("=" * 62)
    df_35 = load_data(LARGE_CSV)
    r = run_verification(df_35, "35 Authors")
    all_results.append(r)

    # ── Write report ────────────────────────────────────────────────────
    print(f"\nSaving report to {args.output}...")
    md = "# Authorship Verification Results\n\n"

    md += "## Summary Table\n\n"
    md += "| Case | Model | Passages | Authors | CV Acc | Test Acc | Precision | Recall | F1 | AUC |\n"
    md += "|------|-------|----------|---------|--------|----------|-----------|--------|----|-----|\n"
    for r in all_results:
        md += (f"| {r['case']} | {r['model']} | {r['n_passages']} | {r['n_authors']} "
               f"| {r['cv_accuracy_mean']:.4f}+/-{r['cv_accuracy_std']:.4f} "
               f"| {r['test_accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} "
               f"| {r['f1']:.4f} | {r['auc']:.4f} |\n")

    md += "\n## Detailed Results\n\n"
    for r in all_results:
        md += f"### {r['case']} — {r['model']}\n\n"
        md += f"- **Passages:** {r['n_passages']}\n"
        md += f"- **Authors:** {r['n_authors']}\n"
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
