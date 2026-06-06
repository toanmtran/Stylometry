"""
MLP Authorship Classifier — Dev/Test Evaluation

Workflow:
  1. Split 60 / 20 / 20  (train / dev / test), stratified by author.
  2. Train every (feature-subset x depth) combination on the train split.
  3. Display dev accuracies as a table: rows = feature subsets, cols = depth.
  4. Retrain the best combo on train+dev.
  5. Evaluate on the held-out test split (touched once).

Runs on the full feature matrix (no outlier removal). Writes results.md + roc.png.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

# ==========================================
# CONFIGURATION
# ==========================================
FEATURE_RANKING = [
    "punct_apost_rate", "n_tokens", "punct_semicolon_rate", "contraction_rate",
    "punct_paren_rate", "punct_colon_rate", "pos_verb", "fw_which", "honore_r",
    "fw_his", "pos_det", "pos_adj", "n_vocab", "avg_paragraph_len", "fw_the",
    "fw_you", "fw_what", "fw_but", "simpson_d", "fw_who", "n_sentences",
    "fw_not", "std_word_len", "word_len_1_frac", "brunet_w", "pos_adv",
    "cat_amplifier_rate", "hapax_ratio", "pos_prep", "fw_this",
    "punct_dquote_rate", "yule_k", "word_len_3_frac", "fw_he", "fw_about",
    "uppercase_ratio", "pos_pron", "punct_comma_rate", "fw_we",
    "punct_period_rate", "pronoun_start_rate", "fw_me", "median_sent_len",
    "pos_conj", "fw_if", "fw_or", "fw_i", "fw_and", "word_len_4_frac",
    "fw_that", "fw_her", "fw_on", "fw_by", "fw_would", "fw_at", "char_i_rate",
    "fw_from", "fw_their", "fw_do", "char_a_rate", "word_len_2_frac",
    "fw_they", "fw_will", "punct_question_rate", "char_e_rate", "fw_to",
    "fw_of", "cat_conj_rate", "punct_excl_rate", "fw_up", "fw_one",
    "fw_have", "avg_sent_len", "fw_out",
]

NETWORK_CONFIGS: dict[str, tuple[int, ...]] = {
    "Depth 1 (64,)":      (64,),
    "Depth 3 (64,64,64)": (64, 64, 64),
    "Depth 10":           tuple([64] * 10),
    "Depth 50":           tuple([64] * 50),
}

FEATURE_SUBSETS = [30, 50]        # top-k from FEATURE_RANKING
PATIENCE        = 15              # n_iter_no_change for early stopping
BATCH_SIZE      = 32
METADATA_COLS   = {"author", "passage_id"}

TRAIN_RATIO  = 0.60
DEV_RATIO    = 0.20
TEST_RATIO   = 0.20
RANDOM_STATE = 42
MAX_ITER     = 500


# ==========================================
# DATA HELPERS
# ==========================================
def get_ordered_features(df: pd.DataFrame) -> list[str]:
    available = set(df.columns) - METADATA_COLS
    return [f for f in FEATURE_RANKING if f in available]


def make_eval_tasks(df: pd.DataFrame, ordered: list[str]) -> list[tuple[str, list[str]]]:
    all_cols = [c for c in df.columns if c not in METADATA_COLS]
    tasks = [(f"Top {k} features", ordered[:k]) for k in FEATURE_SUBSETS]
    tasks.append((f"All {len(all_cols)} features", all_cols))
    return tasks


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


def confusion_matrix_md(cm: np.ndarray, names: list[str]) -> str:
    short = [n[:14] for n in names]
    header = "| Actual \\ Pred | " + " | ".join(f"**{s}**" for s in short) + " |"
    sep    = "|" + "---|" * (len(names) + 1)
    rows   = [header, sep]
    for i, row_vals in enumerate(cm):
        rows.append(f"| **{short[i]}** | " + " | ".join(str(v) for v in row_vals) + " |")
    return "\n".join(rows) + "\n"


def dev_table_md(selection_rows: list[dict], best: dict) -> str:
    """2-D markdown table: rows = feature subsets, cols = architectures."""
    subset_labels = list(dict.fromkeys(r["subset_label"] for r in selection_rows))
    cfg_labels    = list(NETWORK_CONFIGS.keys())
    grid = {(r["subset_label"], r["cfg_label"]): r["dev_acc"] for r in selection_rows}
    best_key = (best["subset_label"], best["cfg_label"])

    header = "| Feature Subset | " + " | ".join(cfg_labels) + " |"
    sep    = "|---|" + "---|" * len(cfg_labels)
    lines  = [header, sep]
    for sub in subset_labels:
        cells = []
        for cfg in cfg_labels:
            val    = grid.get((sub, cfg), float("nan"))
            marker = " ✓" if (sub, cfg) == best_key else ""
            cells.append(f"{val:.4f}{marker}")
        lines.append("| " + sub + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ==========================================
# EXPERIMENT
# ==========================================
def run_experiment(df: pd.DataFrame, plot_path: str) -> str:
    le = LabelEncoder()
    y  = le.fit_transform(df["author"])
    ordered = get_ordered_features(df)
    tasks   = make_eval_tasks(df, ordered)

    # 60 / 20 / 20 stratified split
    idx = np.arange(len(df))
    idx_traindev, idx_test, y_traindev, y_test = train_test_split(
        idx, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_STATE
    )
    idx_train, idx_dev, y_train, y_dev = train_test_split(
        idx_traindev, y_traindev,
        test_size=DEV_RATIO / (TRAIN_RATIO + DEV_RATIO),
        stratify=y_traindev, random_state=RANDOM_STATE,
    )

    n_total    = len(df)
    n_train    = len(idx_train)
    n_dev      = len(idx_dev)
    n_test     = len(idx_test)
    n_traindev = len(idx_traindev)

    print(f"\n{'='*60}")
    print(f"  MLP Authorship Classification")
    print(f"  Total={n_total}  Train={n_train}  Dev={n_dev}  Test={n_test}")
    print(f"{'='*60}")

    # Dev-set model selection
    selection_rows: list[dict] = []
    for subset_label, cols in tasks:
        X_all_sub = df[cols].values
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_all_sub[idx_train])
        X_dev_s   = scaler.transform(X_all_sub[idx_dev])

        for cfg_label, cfg in NETWORK_CONFIGS.items():
            print(f"  [DEV]  {subset_label:20s}  {cfg_label}  patience={PATIENCE}")
            clf = MLPClassifier(
                hidden_layer_sizes=cfg,
                max_iter=MAX_ITER,
                batch_size=BATCH_SIZE,
                random_state=RANDOM_STATE,
                solver="adam",
                early_stopping=True,
                n_iter_no_change=PATIENCE,
            )
            clf.fit(X_train_s, y_train)
            selection_rows.append({
                "subset_label": subset_label,
                "cols":         cols,
                "cfg_label":    cfg_label,
                "cfg":          cfg,
                "train_acc":    clf.score(X_train_s, y_train),
                "dev_acc":      clf.score(X_dev_s,   y_dev),
            })

    selection_rows.sort(key=lambda r: r["dev_acc"], reverse=True)
    best = selection_rows[0]
    print(f"\n  Best on dev: {best['subset_label']} | {best['cfg_label']}"
          f"  (dev={best['dev_acc']:.4f})")

    # Retrain winner on train+dev, evaluate on test
    print(f"  Retraining on train+dev ({n_traindev} passages)...")
    X_best_sub   = df[best["cols"]].values
    scaler_final = StandardScaler()
    X_traindev_s = scaler_final.fit_transform(X_best_sub[idx_traindev])
    X_test_s     = scaler_final.transform(X_best_sub[idx_test])

    final_clf = MLPClassifier(
        hidden_layer_sizes=best["cfg"],
        max_iter=MAX_ITER,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        solver="adam",
        early_stopping=True,
        n_iter_no_change=PATIENCE,
    )
    final_clf.fit(X_traindev_s, y_traindev)
    y_pred_test = final_clf.predict(X_test_s)
    y_prob_test = final_clf.predict_proba(X_test_s)
    test_acc    = final_clf.score(X_test_s, y_test)

    test_report = classification_report(
        y_test, y_pred_test,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    test_df    = pd.DataFrame(test_report).T
    display_rows_order = list(le.classes_) + ["macro avg", "weighted avg"]
    test_table = test_df.loc[display_rows_order]

    roc_auc = roc_auc_score(y_test, y_prob_test, multi_class="ovr", average="macro")
    wf1     = test_report["weighted avg"]["f1-score"]
    cm      = confusion_matrix(y_test, y_pred_test)

    plot_roc_curves(
        y_test, y_prob_test, list(le.classes_),
        title="ROC Curves - MLP",
        save_path=plot_path,
    )
    print(f"  Test accuracy: {test_acc:.4f}  ROC-AUC: {roc_auc:.4f}  WF1: {wf1:.4f}")
    print(test_table.to_string())

    # Build markdown
    md  = "# MLP Authorship Classification\n\n"

    md += "## Data Split\n\n"
    md += "| Set | Passages | Proportion |\n"
    md += "|-----|----------|------------|\n"
    md += f"| Train     | {n_train}    | {n_train/n_total:.0%} |\n"
    md += f"| Dev       | {n_dev}      | {n_dev/n_total:.0%}   |\n"
    md += f"| Test      | {n_test}     | {n_test/n_total:.0%}  |\n"
    md += f"| **Total** | **{n_total}**| 100%      |\n\n"

    md += "## Dev Set - Model Selection\n\n"
    md += ("Dev accuracy for every feature-subset x architecture combination "
           f"(patience={PATIENCE}, batch_size={BATCH_SIZE}). "
           "Best cell marked with checkmark.\n\n")
    md += dev_table_md(selection_rows, best)
    md += "\n"
    md += (f"**Best model:** {best['subset_label']} x {best['cfg_label']} "
           f"- Dev accuracy: **{best['dev_acc']:.4f}**\n\n")

    md += "## Final Test Set Results\n\n"
    md += (f"Retrained on train+dev ({n_traindev} passages) using "
           f"**{best['subset_label']}**, **{best['cfg_label']}**.\n\n")

    md += "### Key Metrics\n\n"
    md += "| Metric | Value |\n|--------|-------|\n"
    md += f"| Accuracy            | {test_acc:.4f} |\n"
    md += f"| Weighted F1         | {wf1:.4f} |\n"
    md += f"| ROC-AUC (macro OvR) | {roc_auc:.4f} |\n\n"

    md += "### Per-Class Report\n\n"
    md += test_table.to_markdown() + "\n\n"

    md += "### Confusion Matrix\n\n"
    md += "_Rows = actual, Columns = predicted._\n\n"
    md += confusion_matrix_md(cm, list(le.classes_)) + "\n"

    plot_fname = os.path.basename(plot_path)
    md += f"## ROC Curves\n\n![ROC Curves]({plot_fname})\n"

    return md


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    input_csv = "author_features_extracted_full.csv"
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"Loading {input_csv}...")
    df_full = pd.read_csv(input_csv)
    print(f"  {len(df_full)} rows loaded")

    md = run_experiment(df_full, plot_path="roc.png")
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\nSaved: results.md  |  roc.png")


if __name__ == "__main__":
    main()
