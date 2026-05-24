"""
Random Forest Authorship Classifier — Nested Cross-Validation

Outer 5-fold CV estimates generalisation performance.
Inner 3-fold GridSearchCV tunes hyperparameters on each outer training split.
Runs both cases (with / without outliers) and writes one MD file per case.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

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
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "max_features":      ["sqrt", "log2"],
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


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str,
) -> None:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))

    fpr_all, tpr_all = {}, {}
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr_all[i], tpr_all[i] = fpr, tpr
        ax.plot(fpr, tpr, lw=1.5, label=f"{name}  (AUC={auc(fpr, tpr):.3f})")

    all_fpr = np.unique(np.concatenate(list(fpr_all.values())))
    mean_tpr = np.mean(
        [np.interp(all_fpr, fpr_all[i], tpr_all[i]) for i in range(n_classes)], axis=0
    )
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k--", lw=2, label=f"Macro avg  (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: confidence-accuracy gap weighted by bin size."""
    confidences = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def confusion_matrix_md(cm: np.ndarray, names: list[str]) -> str:
    short = [n[:14] for n in names]
    header = "| Actual \\ Pred | " + " | ".join(f"**{s}**" for s in short) + " |"
    sep    = "|" + "---|" * (len(names) + 1)
    rows   = [header, sep]
    for i, row_vals in enumerate(cm):
        rows.append(f"| **{short[i]}** | " + " | ".join(str(v) for v in row_vals) + " |")
    return "\n".join(rows) + "\n"


# ── nested CV ──────────────────────────────────────────────────────────────
def run_nested_cv(df: pd.DataFrame, case_label: str, plot_path: str) -> str:
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    le = LabelEncoder()
    y  = le.fit_transform(df["author"])

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_results: list[dict] = []
    fold_reports: list[pd.DataFrame] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_y_prob: list[np.ndarray] = []

    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)

    print(f"\n{'='*62}")
    print(f"  Random Forest | {case_label}")
    print(f"  {len(df)} passages | {len(feature_cols)} features | "
          f"{n_combos} param combos x {INNER_FOLDS} inner folds x {OUTER_FOLDS} outer folds")
    print(f"{'='*62}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        base_clf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        gs = GridSearchCV(
            base_clf, PARAM_GRID,
            cv=inner_cv, scoring="accuracy",
            n_jobs=1, refit=True,   # loky multiprocessing OOMs on Windows; RF uses threading internally
        )
        gs.fit(X_train, y_train)

        y_pred  = gs.predict(X_test)
        y_prob  = gs.predict_proba(X_test)
        acc     = accuracy_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred, average="macro",    zero_division=0)
        rec     = recall_score(  y_test, y_pred, average="macro",    zero_division=0)
        wf1     = f1_score(      y_test, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score( y_test, y_prob, multi_class="ovr",  average="macro")

        print(f"  Fold {fold_idx}: acc={acc:.4f}  prec={prec:.4f}  "
              f"rec={rec:.4f}  wF1={wf1:.4f}  ROC-AUC={roc_auc:.4f}")
        print(f"          best: {fmt_params(gs.best_params_)}")

        fold_results.append({
            "fold":        fold_idx,
            "best_params": fmt_params(gs.best_params_),
            "accuracy":    acc,
            "precision":   prec,
            "recall":      rec,
            "w_f1":        wf1,
            "roc_auc":     roc_auc,
        })
        fold_reports.append(pd.DataFrame(classification_report(
            y_test, y_pred, target_names=le.classes_,
            output_dict=True, zero_division=0,
        )).T)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        all_y_prob.append(y_prob)

    # ── aggregate ──────────────────────────────────────────────────────────
    agg_y_true = np.concatenate(all_y_true)
    agg_y_pred = np.concatenate(all_y_pred)
    agg_y_prob = np.vstack(all_y_prob)
    agg_cm     = confusion_matrix(agg_y_true, agg_y_pred)
    agg_ece    = compute_ece(agg_y_true, agg_y_prob)

    res = pd.DataFrame(fold_results)
    stats = {
        m: (res[m].mean(), res[m].std())
        for m in ("accuracy", "precision", "recall", "w_f1", "roc_auc")
    }

    print(f"\n  Mean  acc={stats['accuracy'][0]:.4f}+/-{stats['accuracy'][1]:.4f}  "
          f"prec={stats['precision'][0]:.4f}+/-{stats['precision'][1]:.4f}  "
          f"rec={stats['recall'][0]:.4f}+/-{stats['recall'][1]:.4f}  "
          f"wF1={stats['w_f1'][0]:.4f}+/-{stats['w_f1'][1]:.4f}  "
          f"ROC-AUC={stats['roc_auc'][0]:.4f}+/-{stats['roc_auc'][1]:.4f}  "
          f"ECE={agg_ece:.4f}")

    avg_report   = pd.concat(fold_reports).groupby(level=0).mean()
    display_rows = list(le.classes_) + ["macro avg", "weighted avg"]
    avg_table    = avg_report.loc[display_rows]

    # ── markdown ───────────────────────────────────────────────────────────
    md  = f"# Random Forest Authorship Classification — {case_label}\n\n"

    md += "## Configuration\n\n"
    md += f"- **Classifier:** Random Forest\n"
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
    md += "| Fold | Accuracy | Precision (macro) | Recall (macro) | Weighted F1 | ROC-AUC | Best Params |\n"
    md += "|------|----------|-------------------|----------------|-------------|---------|-------------|\n"
    for row in fold_results:
        md += (f"| {row['fold']} | {row['accuracy']:.4f} | {row['precision']:.4f} "
               f"| {row['recall']:.4f} | {row['w_f1']:.4f} | {row['roc_auc']:.4f} "
               f"| `{row['best_params']}` |\n")

    md += "\n## Summary\n\n"
    md += "| Metric | Mean | Std |\n|--------|------|-----|\n"
    md += f"| Accuracy            | {stats['accuracy'][0]:.4f}  | {stats['accuracy'][1]:.4f}  |\n"
    md += f"| Precision (macro)   | {stats['precision'][0]:.4f} | {stats['precision'][1]:.4f} |\n"
    md += f"| Recall (macro)      | {stats['recall'][0]:.4f}    | {stats['recall'][1]:.4f}    |\n"
    md += f"| Weighted F1         | {stats['w_f1'][0]:.4f}      | {stats['w_f1'][1]:.4f}      |\n"
    md += f"| ROC-AUC (macro OvR) | {stats['roc_auc'][0]:.4f}   | {stats['roc_auc'][1]:.4f}   |\n"
    md += f"| ECE (aggregated)    | {agg_ece:.4f}               | —                           |\n"

    md += "\n## Average Classification Report\n\n"
    md += "_Per-class metrics averaged across all outer folds._\n\n"
    md += avg_table.to_markdown() + "\n\n"

    md += "## Confusion Matrix\n\n"
    md += "_Aggregated across all outer folds. Rows = actual, Columns = predicted._\n\n"
    md += confusion_matrix_md(agg_cm, list(le.classes_)) + "\n"

    plot_roc_curves(
        agg_y_true, agg_y_prob, list(le.classes_),
        title=f"ROC Curves — Random Forest ({case_label})",
        save_path=plot_path,
    )
    plot_fname = os.path.basename(plot_path)
    md += f"## ROC Curves\n\n![ROC Curves]({plot_fname})\n"

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
        (df_full,  "With Outliers",    "results_with_outliers.md",    "roc_with_outliers.png"),
        (df_clean, "Without Outliers", "results_without_outliers.md", "roc_without_outliers.png"),
    ]
    for df, label, outfile, plotfile in cases:
        plot_path = os.path.join(_HERE, plotfile)
        md        = run_nested_cv(df, label, plot_path=plot_path)
        out_path  = os.path.join(_HERE, outfile)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nSaved: {out_path}  |  {plot_path}")


if __name__ == "__main__":
    main()
