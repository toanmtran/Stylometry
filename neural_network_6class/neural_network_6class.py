"""
MLP Authorship Classifier - 6-class variant (adds none_of_the_5_authors).

Extra class construction:
  - Randomly pick N_NONE_AUTHORS (15) authors from cleaned_35 that are NOT
    among the 5 training authors.
  - Sample N_ARTICLES_PER_NONE_AUTHOR (10) articles from each -> 150 total.
  - Extract stylometric features and label every article "none_of_the_5_authors".
  - Merge with the existing 5-author feature CSV.
  - Remove per-author outliers via Isolation Forest.
  - Split 60/20/20, run dev-set model selection, evaluate on test set.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import extract_features

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

NONE_LABEL = "none_of_the_5_authors"

FIVE_AUTHORS_STEMS = {
    "eliezer_yudkowsky", "johnswentworth", "raemon", "scottalexander", "zvi",
}

CLEANED_35_DIR = (
    Path(__file__).resolve().parent.parent
    / "dataset" / "lesswrong_large" / "cleaned_35"
)

N_NONE_AUTHORS             = 15
N_ARTICLES_PER_NONE_AUTHOR = 10   # 15 x 10 = 150 total


# ==========================================
# NONE-CLASS CONSTRUCTION
# ==========================================
def build_none_class_df(seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    rng = random.Random(seed)
    candidates = [
        fp.stem
        for fp in sorted(CLEANED_35_DIR.glob("*.json"))
        if fp.stem not in FIVE_AUTHORS_STEMS
    ]
    chosen_stems = rng.sample(candidates, N_NONE_AUTHORS)
    print(f"\n  None-class authors ({N_NONE_AUTHORS}):")
    for s in chosen_stems:
        print(f"    {s}")

    rows: list[dict] = []
    passage_id = 0
    for stem in chosen_stems:
        fp = CLEANED_35_DIR / f"{stem}.json"
        all_articles: list[dict] = json.loads(fp.read_text(encoding="utf-8"))
        picked = rng.sample(all_articles, min(N_ARTICLES_PER_NONE_AUTHOR, len(all_articles)))
        for art in picked:
            feats = extract_features(art["text"])
            feats["author"]     = NONE_LABEL
            feats["passage_id"] = passage_id
            rows.append(feats)
            passage_id += 1
        print(f"    {stem}: extracted {len(picked)} articles")

    df = pd.DataFrame(rows).fillna(0.0)
    return df, chosen_stems


# ==========================================
# DATA HELPERS
# ==========================================
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    kept = []
    for _, group in df.groupby("author"):
        clf = IsolationForest(contamination="auto", random_state=RANDOM_STATE)
        mask = clf.fit_predict(group[feature_cols].values) == 1
        kept.append(group[mask])
    return pd.concat(kept).reset_index(drop=True)


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
def run_experiment(df: pd.DataFrame, none_authors: list[str], plot_path: str, case_label: str = "Without Outliers") -> str:
    le = LabelEncoder()
    y  = le.fit_transform(df["author"])
    ordered = get_ordered_features(df)
    tasks   = make_eval_tasks(df, ordered)

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

    none_total = int(df["author"].value_counts().get(NONE_LABEL, 0))
    print(f"\n{'='*60}")
    print(f"  {case_label}")
    print(f"  Total={n_total}  Train={n_train}  Dev={n_dev}  Test={n_test}")
    print(f"  none_of_the_5_authors rows: {none_total}")
    print(f"{'='*60}")

    # Dev-set model selection
    selection_rows: list[dict] = []
    for subset_label, cols in tasks:
        X_sub  = df[cols].values
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_sub[idx_train])
        X_dv   = scaler.transform(X_sub[idx_dev])
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
            clf.fit(X_tr, y_train)
            selection_rows.append({
                "subset_label": subset_label,
                "cols":         cols,
                "cfg_label":    cfg_label,
                "cfg":          cfg,
                "train_acc":    clf.score(X_tr, y_train),
                "dev_acc":      clf.score(X_dv, y_dev),
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

    five_classes  = [c for c in le.classes_ if c != NONE_LABEL]
    display_names = five_classes + [NONE_LABEL] + ["macro avg", "weighted avg"]

    test_report = classification_report(
        y_test, y_pred_test,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    test_df    = pd.DataFrame(test_report).T
    test_table = test_df.loc[display_names]

    roc_auc = roc_auc_score(y_test, y_prob_test, multi_class="ovr", average="macro")
    wf1     = test_report["weighted avg"]["f1-score"]
    cm      = confusion_matrix(y_test, y_pred_test)

    plot_roc_curves(
        y_test, y_prob_test, list(le.classes_),
        title=f"ROC Curves - MLP 6-class ({case_label})",
        save_path=plot_path,
    )
    print(f"  Test accuracy: {test_acc:.4f}  ROC-AUC: {roc_auc:.4f}  WF1: {wf1:.4f}")
    print(test_table.to_string())

    # Markdown
    md  = f"# MLP 6-Class Authorship Classification - {case_label}\n\n"

    md += "## None-of-the-5-Authors Class Construction\n\n"
    md += (f"**{N_NONE_AUTHORS} authors x {N_ARTICLES_PER_NONE_AUTHOR} articles = "
           f"{N_NONE_AUTHORS * N_ARTICLES_PER_NONE_AUTHOR} total passages** "
           f"labelled `{NONE_LABEL}`.\n\n")
    md += "Authors used for the none class:\n\n"
    for s in none_authors:
        md += f"- `{s}`\n"
    md += "\n"

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

    plot_fname = Path(plot_path).name
    md += f"## ROC Curves\n\n![ROC Curves]({plot_fname})\n"

    return md


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    input_csv = str(
        Path(__file__).resolve().parent.parent
        / "neural_network"
        / "author_features_extracted_full.csv"
    )
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")

    print(f"Loading 5-author features from {input_csv} ...")
    df_5 = pd.read_csv(input_csv)
    print(f"  {len(df_5)} rows loaded")

    print("\nBuilding none-of-the-5-authors class (seed=42) ...")
    df_none, none_authors = build_none_class_df(seed=42)
    print(f"  {len(df_none)} none-class rows extracted")

    all_cols = sorted(set(df_5.columns) | set(df_none.columns))
    for col in all_cols:
        if col not in df_5.columns:
            df_5[col] = 0.0
        if col not in df_none.columns:
            df_none[col] = 0.0

    df_full = pd.concat([df_5[all_cols], df_none[all_cols]], ignore_index=True)
    print(f"\nMerged dataset: {len(df_full)} rows, {len(df_full.columns)} columns")
    print(df_full["author"].value_counts().to_string())

    print("\nRemoving outliers ...")
    df_clean = remove_outliers(df_full)
    print(f"  {len(df_full)} -> {len(df_clean)} rows after outlier removal")

    output_dir = Path(__file__).parent

    plot_with = str(output_dir / "roc_with_outliers.png")
    md_with   = run_experiment(df_full, none_authors, plot_path=plot_with, case_label="With Outliers")
    out_with  = output_dir / "results_with_outliers.md"
    out_with.write_text(md_with, encoding="utf-8")
    print(f"\nSaved: {out_with}  |  {plot_with}")

    plot_without = str(output_dir / "roc_without_outliers.png")
    md_without   = run_experiment(df_clean, none_authors, plot_path=plot_without, case_label="Without Outliers")
    out_without  = output_dir / "results_without_outliers.md"
    out_without.write_text(md_without, encoding="utf-8")
    print(f"\nSaved: {out_without}  |  {plot_without}")


if __name__ == "__main__":
    main()
