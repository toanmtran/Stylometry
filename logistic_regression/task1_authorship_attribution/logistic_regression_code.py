"""
Logistic Regression Authorship Classifier — Nested Cross-Validation

Outer 5-fold CV estimates generalisation performance.
Inner 3-fold GridSearchCV tunes hyperparameters on each outer training split.
Runs both cases (with / without outliers) and writes one MD file per case.

Key differences from tree-based models:
  - Requires feature scaling (StandardScaler)
  - Uses multinomial (softmax) for multi-class
  - Can extract coefficient-based feature importance
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = _HERE
FULL_CSV = os.path.join(DATA_DIR, "features_25authors.csv")

# ── configuration ──────────────────────────────────────────────────────────
METADATA_COLS = {"author", "passage_id"}
OUTER_FOLDS   = 5
INNER_FOLDS   = 3
RANDOM_STATE  = 42

PARAM_GRID = {
    "C":        [0.01, 0.1, 1, 10, 100],
    "penalty":  ["l2"],
    "solver":   ["lbfgs"],
    "max_iter": [1000, 2000, 5000],
}


# ── helpers ────────────────────────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers per author using Isolation Forest."""
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    kept = []
    for _, group in df.groupby("author"):
        clf = IsolationForest(contamination="auto", random_state=RANDOM_STATE)
        mask = clf.fit_predict(group[feature_cols].values) == 1
        kept.append(group[mask])
    return pd.concat(kept).reset_index(drop=True)


def fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def extract_feature_importance(
    model: LogisticRegression,
    feature_names: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    """Extract per-class top features from Logistic Regression coefficients."""
    coef_df = pd.DataFrame(
        model.coef_,
        columns=feature_names,
        index=class_names,
    )
    # Top 5 features with highest positive coefficient per author
    top_features = coef_df.apply(
        lambda row: row.nlargest(5).to_dict(),
        axis=1,
    )
    # Top 5 features with most negative coefficient per author
    bottom_features = coef_df.apply(
        lambda row: row.nsmallest(5).to_dict(),
        axis=1,
    )
    return coef_df, top_features, bottom_features


# ── nested CV ──────────────────────────────────────────────────────────────
def run_nested_cv(df: pd.DataFrame, case_label: str) -> str:
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    le = LabelEncoder()
    y  = le.fit_transform(df["author"])

    # Logistic Regression requires scaled features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_results: list[dict] = []
    fold_reports: list[pd.DataFrame] = []

    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)

    print(f"\n{'='*62}")
    print(f"  Logistic Regression | {case_label}")
    print(f"  {len(df)} passages | {len(feature_cols)} features | "
          f"{n_combos} param combos x {INNER_FOLDS} inner folds x {OUTER_FOLDS} outer folds")
    print(f"{'='*62}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        base_clf = LogisticRegression(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        gs = GridSearchCV(
            base_clf, PARAM_GRID,
            cv=inner_cv, scoring="accuracy",
            n_jobs=1, refit=True,
        )
        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro",    zero_division=0)
        rec  = recall_score(  y_test, y_pred, average="macro",    zero_division=0)
        wf1  = f1_score(      y_test, y_pred, average="weighted", zero_division=0)

        print(f"  Fold {fold_idx}: acc={acc:.4f}  prec={prec:.4f}  "
              f"rec={rec:.4f}  wF1={wf1:.4f}")
        print(f"          best: {fmt_params(gs.best_params_)}")

        fold_results.append({
            "fold":        fold_idx,
            "best_params": fmt_params(gs.best_params_),
            "accuracy":    acc,
            "precision":   prec,
            "recall":      rec,
            "w_f1":        wf1,
        })

        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True, zero_division=0,
        )
        fold_reports.append(pd.DataFrame(report).T)

    # ── aggregate ──────────────────────────────────────────────────────────
    res = pd.DataFrame(fold_results)
    stats = {
        m: (res[m].mean(), res[m].std())
        for m in ("accuracy", "precision", "recall", "w_f1")
    }

    print(f"\n  Mean  acc={stats['accuracy'][0]:.4f}+/-{stats['accuracy'][1]:.4f}  "
          f"prec={stats['precision'][0]:.4f}+/-{stats['precision'][1]:.4f}  "
          f"rec={stats['recall'][0]:.4f}+/-{stats['recall'][1]:.4f}  "
          f"wF1={stats['w_f1'][0]:.4f}+/-{stats['w_f1'][1]:.4f}")

    avg_report   = pd.concat(fold_reports).groupby(level=0).mean()
    display_rows = list(le.classes_) + ["macro avg", "weighted avg"]
    avg_table    = avg_report.loc[display_rows]

    # ── Retrain on full data for feature importance analysis ────────────────
    print("\n  Retraining on full dataset for feature importance...")
    final_gs = GridSearchCV(
        LogisticRegression(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        PARAM_GRID,
        cv=inner_cv, scoring="accuracy",
        n_jobs=1, refit=True,
    )
    final_gs.fit(X, y)
    final_model: LogisticRegression = final_gs.best_estimator_
    coef_df, top_feats, bottom_feats = extract_feature_importance(
        final_model, feature_cols, list(le.classes_),
    )

    # ── markdown ───────────────────────────────────────────────────────────
    md  = f"# Logistic Regression Authorship Classification — {case_label}\n\n"

    md += "## Configuration\n\n"
    md += f"- **Classifier:** Logistic Regression (multinomial / softmax)\n"
    md += f"- **Scaler:** StandardScaler\n"
    md += f"- **Outer folds:** {OUTER_FOLDS} (performance estimation)\n"
    md += f"- **Inner folds:** {INNER_FOLDS} (hyperparameter tuning via GridSearchCV)\n"
    md += f"- **Param combinations:** {n_combos}\n"
    md += f"- **Passages:** {len(df)}\n"
    md += f"- **Features:** all {len(feature_cols)}\n\n"
    md += "**Search grid:**\n\n"
    md += "| Hyperparameter | Values |\n|----------------|--------|\n"
    for k, v in PARAM_GRID.items():
        md += f"| `{k}` | {v} |\n"
    md += "\n"

    md += "## Per-Fold Results\n\n"
    md += "| Fold | Accuracy | Precision (macro) | Recall (macro) | Weighted F1 | Best Params |\n"
    md += "|------|----------|-------------------|----------------|-------------|-------------|\n"
    for row in fold_results:
        md += (f"| {row['fold']} | {row['accuracy']:.4f} | {row['precision']:.4f} "
               f"| {row['recall']:.4f} | {row['w_f1']:.4f} | `{row['best_params']}` |\n")

    md += "\n## Summary\n\n"
    md += "| Metric | Mean | Std |\n|--------|------|-----|\n"
    md += f"| Accuracy           | {stats['accuracy'][0]:.4f}  | {stats['accuracy'][1]:.4f}  |\n"
    md += f"| Precision (macro)  | {stats['precision'][0]:.4f} | {stats['precision'][1]:.4f} |\n"
    md += f"| Recall (macro)     | {stats['recall'][0]:.4f}    | {stats['recall'][1]:.4f}    |\n"
    md += f"| Weighted F1        | {stats['w_f1'][0]:.4f}      | {stats['w_f1'][1]:.4f}      |\n"

    md += "\n## Average Classification Report\n\n"
    md += "_Per-class metrics averaged across all outer folds._\n\n"
    md += avg_table.to_markdown() + "\n"

    # ── Feature importance section ──────────────────────────────────────────
    md += "\n## Feature Importance (Coefficient Analysis)\n\n"
    md += "Trained on full dataset with best hyperparameters found via inner CV.\n\n"
    md += "### Top 5 Positive Features Per Author\n\n"
    md += "_Features with highest positive coefficient → strongly associated with this author._\n\n"
    for author in le.classes_:
        md += f"**{author}:**\n\n"
        md += "| Feature | Coefficient |\n|---------|-------------|\n"
        for feat, coef_val in top_feats[author].items():
            md += f"| `{feat}` | {coef_val:.4f} |\n"
        md += "\n"

    md += "### Top 5 Negative Features Per Author\n\n"
    md += "_Features with most negative coefficient → strongly disassociated from this author._\n\n"
    for author in le.classes_:
        md += f"**{author}:**\n\n"
        md += "| Feature | Coefficient |\n|---------|-------------|\n"
        for feat, coef_val in bottom_feats[author].items():
            md += f"| `{feat}` | {coef_val:.4f} |\n"
        md += "\n"

    md += "### Most Discriminative Features Overall\n\n"
    md += "_Features with highest absolute coefficient range across authors._\n\n"
    coef_range = coef_df.max() - coef_df.min()
    top_discriminative = coef_range.nlargest(10)
    md += "| Feature | Max Coef | Min Coef | Range |\n|---------|----------|----------|-------|\n"
    for feat in top_discriminative.index:
        max_val = coef_df.loc[:, feat].max()
        min_val = coef_df.loc[:, feat].min()
        md += f"| `{feat}` | {max_val:.4f} | {min_val:.4f} | {top_discriminative[feat]:.4f} |\n"

    return md


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data...")
    df_full = pd.read_csv(FULL_CSV)
    print(f"  Full dataset: {len(df_full)} rows")

    print("  Computing outlier removal...")
    df_clean = remove_outliers(df_full)
    print(f"  Clean dataset: {len(df_clean)} rows")

    cases = [
        (df_full,  "With Outliers",    "results_with_outliers.md"),
        (df_clean, "Without Outliers", "results_without_outliers.md"),
    ]
    for df, label, outfile in cases:
        md = run_nested_cv(df, label)
        out_path = os.path.join(_HERE, outfile)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
