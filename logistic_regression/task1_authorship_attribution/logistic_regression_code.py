"""
Logistic Regression Authorship Classifier — Nested Cross-Validation

Outer 5-fold CV estimates generalisation performance.
Inner 3-fold GridSearchCV tunes hyperparameters on each outer training split.
Runs both cases (with / without outliers) and writes one MD file per case.

Key differences from tree-based models:
  - Requires feature scaling (StandardScaler)
  - Uses multinomial (softmax) for multi-class
  - Can extract coefficient-based feature importance

IMPORTANT — data-leakage safeguards:
  - Outlier removal (IsolationForest) is re-fit on each outer training split
    so test passages never influence which training samples are kept.
  - StandardScaler is re-fit on each outer training split; test folds are
    only transformed.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── bootstrap imports ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)  # logistic_regression/
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from _config import (INNER_FOLDS, METADATA_COLS, OUTER_FOLDS, PARAM_GRID_LR,
                     RANDOM_STATE, fmt_params, n_param_combos)

# ── paths ──────────────────────────────────────────────────────────────────────
FULL_CSV = os.path.join(_HERE, "features_25authors.csv")


# ── helpers ────────────────────────────────────────────────────────────────────

def _remove_outliers_on_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    contamination: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove outliers from *training* data only (per-class), keeping test untouched."""
    if len(X_train) == 0:
        return X_train, y_train

    kept_mask = np.ones(len(X_train), dtype=bool)
    for cls in np.unique(y_train):
        idx = np.where(y_train == cls)[0]
        if len(idx) < 3:
            continue
        iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
        fold_mask = iso.fit_predict(X_train[idx]) == 1
        kept_mask[idx] = fold_mask

    return X_train[kept_mask], y_train[kept_mask]


def extract_feature_importance(
    model: LogisticRegression,
    feature_names: list[str],
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract per-class top features from Logistic Regression coefficients."""
    coef_df = pd.DataFrame(model.coef_, columns=feature_names, index=class_names)
    top_features = coef_df.apply(lambda row: row.nlargest(5).to_dict(), axis=1)
    bottom_features = coef_df.apply(lambda row: row.nsmallest(5).to_dict(), axis=1)
    return coef_df, top_features, bottom_features


# ── nested CV ──────────────────────────────────────────────────────────────────

def run_nested_cv(df: pd.DataFrame, case_label: str) -> str:
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X_raw = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["author"])

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_results: list[dict] = []
    fold_reports: list[pd.DataFrame] = []

    n_combos = n_param_combos(PARAM_GRID_LR)
    remove_outliers_flag = "without outliers" in case_label.lower()

    print(f"\n{'='*62}")
    print(f"  Logistic Regression | {case_label}")
    print(f"  {len(df)} passages | {len(feature_cols)} features | "
          f"{n_combos} param combos x {INNER_FOLDS} inner folds x {OUTER_FOLDS} outer folds")
    print(f"{'='*62}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw, y), 1):
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── per-fold outlier removal (fit on train only) ──────────────────
        if remove_outliers_flag:
            X_train, y_train = _remove_outliers_on_train(X_train, y_train)

        # ── per-fold scaling (fit on train, transform test) ───────────────
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        base_clf = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)
        gs = GridSearchCV(
            base_clf, PARAM_GRID_LR,
            cv=inner_cv, scoring="accuracy",
            n_jobs=-1, refit=True,
        )
        gs.fit(X_train_scaled, y_train)

        y_pred = gs.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        wf1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  Fold {fold_idx}: acc={acc:.4f}  prec={prec:.4f}  "
              f"rec={rec:.4f}  wF1={wf1:.4f}")
        print(f"          best: {fmt_params(gs.best_params_)}")

        fold_results.append({
            "fold": fold_idx,
            "best_params": fmt_params(gs.best_params_),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "w_f1": wf1,
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

    avg_report = pd.concat(fold_reports).groupby(level=0).mean()
    display_rows = list(le.classes_) + ["macro avg", "weighted avg"]
    avg_table = avg_report.loc[display_rows]

    # ── Retrain on full data for feature importance (single best model) ────
    print("\n  Retraining on full dataset for feature importance...")

    if remove_outliers_flag:
        X_final, y_final = _remove_outliers_on_train(X_raw, y)
    else:
        X_final, y_final = X_raw, y

    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_final)

    final_gs = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID_LR,
        cv=inner_cv, scoring="accuracy",
        n_jobs=-1, refit=True,
    )
    final_gs.fit(X_final_scaled, y_final)
    final_model: LogisticRegression = final_gs.best_estimator_
    coef_df, top_feats, bottom_feats = extract_feature_importance(
        final_model, feature_cols, list(le.classes_),
    )

    # ── Save model for predict_demo.py ─────────────────────────────────────
    case_slug = "clean" if remove_outliers_flag else "outliers"
    save_dir = os.path.join(_HERE, "saved_model", case_slug)
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(save_dir, "lr_attribution.joblib"))
    joblib.dump(final_scaler, os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(save_dir, "feature_cols.joblib"))
    print(f"  Saved model to: {save_dir}/")

    # ── markdown ───────────────────────────────────────────────────────────
    md = f"# Logistic Regression Authorship Classification — {case_label}\n\n"

    md += "## Configuration\n\n"
    md += f"- **Classifier:** Logistic Regression (multinomial / softmax)\n"
    md += f"- **Scaler:** StandardScaler (re-fit per outer fold — no leakage)\n"
    md += f"- **Outlier removal:** {'per-fold IsolationForest (contamination=0.05)' if remove_outliers_flag else 'none'}\n"
    md += f"- **Outer folds:** {OUTER_FOLDS} (performance estimation)\n"
    md += f"- **Inner folds:** {INNER_FOLDS} (hyperparameter tuning via GridSearchCV)\n"
    md += f"- **Param combinations:** {n_combos}\n"
    md += f"- **Passages:** {len(df)}\n"
    md += f"- **Features:** all {len(feature_cols)}\n\n"
    md += "**Search grid:**\n\n"
    md += "| Hyperparameter | Values |\n|----------------|--------|\n"
    for k, v in PARAM_GRID_LR.items():
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
    md += "_Features with highest positive coefficient — strongly associated with this author._\n\n"
    for author in le.classes_:
        md += f"**{author}:**\n\n"
        md += "| Feature | Coefficient |\n|---------|-------------|\n"
        for feat, coef_val in top_feats[author].items():
            md += f"| `{feat}` | {coef_val:.4f} |\n"
        md += "\n"

    md += "### Top 5 Negative Features Per Author\n\n"
    md += "_Features with most negative coefficient — strongly disassociated from this author._\n\n"
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


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    df_full = pd.read_csv(FULL_CSV)
    print(f"  Full dataset: {len(df_full)} rows")

    # We no longer pre-compute outlier removal — each CV fold handles it.
    # But we still need both cases: with and without outliers.
    # For "with outliers": use df_full directly, no per-fold removal.
    # For "without outliers": remove outliers per-fold (inside run_nested_cv).

    cases = [
        (df_full, "With Outliers", "results_with_outliers.md"),
        (df_full, "Without Outliers", "results_without_outliers.md"),
    ]
    for df, label, outfile in cases:
        md = run_nested_cv(df, label)
        out_path = os.path.join(_HERE, outfile)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
