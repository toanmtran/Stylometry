"""
XGBoost Authorship Classifier — Nested Cross-Validation

Outer 5-fold CV estimates generalisation performance.
Inner 3-fold GridSearchCV tunes hyperparameters on each outer training split.
Runs both cases (with / without outliers) and writes one MD file per case.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, "..", "neural_network")
FULL_CSV  = os.path.join(DATA_DIR, "author_features_extracted_full.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "feature_extracted_without_outliers.csv")

# ── configuration ──────────────────────────────────────────────────────────
METADATA_COLS = {"author", "passage_id"}
OUTER_FOLDS   = 5
INNER_FOLDS   = 3
RANDOM_STATE  = 42

PARAM_GRID = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
}


# ── helpers ────────────────────────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    kept = []
    for _, group in df.groupby("author"):
        clf = IsolationForest(contamination="auto", random_state=RANDOM_STATE)
        mask = clf.fit_predict(group[feature_cols].values) == 1
        kept.append(group[mask])
    return pd.concat(kept).reset_index(drop=True)


def fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


# ── nested CV ──────────────────────────────────────────────────────────────
def run_nested_cv(df: pd.DataFrame, case_label: str) -> str:
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    le = LabelEncoder()
    y  = le.fit_transform(df["author"])

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_results: list[dict] = []
    fold_reports: list[pd.DataFrame] = []

    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)

    print(f"\n{'='*62}")
    print(f"  XGBoost | {case_label}")
    print(f"  {len(df)} passages | {len(feature_cols)} features | "
          f"{n_combos} param combos x {INNER_FOLDS} inner folds x {OUTER_FOLDS} outer folds")
    print(f"{'='*62}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        base_clf = XGBClassifier(
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )
        gs = GridSearchCV(
            base_clf, PARAM_GRID,
            cv=inner_cv, scoring="accuracy",
            n_jobs=1, refit=True,   # loky multiprocessing crashes XGBoost on Windows
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
            "fold":       fold_idx,
            "best_params": fmt_params(gs.best_params_),
            "accuracy":   acc,
            "precision":  prec,
            "recall":     rec,
            "w_f1":       wf1,
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

    # ── markdown ───────────────────────────────────────────────────────────
    md  = f"# XGBoost Authorship Classification — {case_label}\n\n"

    md += "## Configuration\n\n"
    md += f"- **Classifier:** XGBoost\n"
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

    return md


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data...")
    df_full = pd.read_csv(FULL_CSV)
    print(f"  Full dataset: {len(df_full)} rows")

    if os.path.exists(CLEAN_CSV):
        df_clean = pd.read_csv(CLEAN_CSV)
        print(f"  Clean dataset: {len(df_clean)} rows (loaded from file)")
    else:
        print("  Clean CSV not found — computing outlier removal...")
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
