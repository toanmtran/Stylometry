"""
MLP Authorship Classifier — Dev/Test Evaluation

Workflow (run independently for each case):
  1. Split data 60 / 20 / 20  (train / dev / test), stratified by author.
  2. Train every (feature-subset × depth) combination on the train split.
  3. Select the best combination by dev accuracy.
  4. Retrain the winner on train+dev.
  5. Evaluate on the held-out test split (touched once per case).

Cases:
  - With outliers    → uses author_features_extracted_full.csv as-is
  - Without outliers → applies per-author Isolation Forest before splitting
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    "Depth 2 (64, 32)":   (64, 32),
    "Depth 3 (64,64,64)": (64, 64, 64),
    "Depth 10":           tuple([64] * 10),
    "Depth 50":           tuple([64] * 50),
}

FEATURE_SUBSETS = [15, 30, 50, 74]   # top-k from FEATURE_RANKING; 74 = full ranked set
METADATA_COLS   = {"author", "passage_id"}

TRAIN_RATIO  = 0.60   # of total
DEV_RATIO    = 0.20   # of total  → test_size=0.25 of the 80% traindev pool
TEST_RATIO   = 0.20   # of total
RANDOM_STATE = 42


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


def make_eval_tasks(
    df: pd.DataFrame, ordered: list[str]
) -> list[tuple[str, list[str]]]:
    """Return (label, column_list) pairs for each feature subset."""
    all_cols = [c for c in df.columns if c not in METADATA_COLS]
    tasks = [(f"Top {k} features", ordered[:k]) for k in FEATURE_SUBSETS]
    tasks.append((f"All {len(all_cols)} features", all_cols))
    return tasks


# ==========================================
# EXPERIMENT
# ==========================================
def run_experiment(df: pd.DataFrame, case_label: str, early_stopping: bool) -> str:
    le = LabelEncoder()
    y  = le.fit_transform(df["author"])
    ordered = get_ordered_features(df)
    tasks   = make_eval_tasks(df, ordered)

    # ── 60 / 20 / 20 stratified split ──────────────────────────────────────
    idx = np.arange(len(df))
    idx_traindev, idx_test, y_traindev, y_test = train_test_split(
        idx, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_STATE
    )
    idx_train, idx_dev, y_train, y_dev = train_test_split(
        idx_traindev, y_traindev,
        test_size=DEV_RATIO / (TRAIN_RATIO + DEV_RATIO),   # 0.25 of 80% = 20%
        stratify=y_traindev, random_state=RANDOM_STATE,
    )

    n_total    = len(df)
    n_train    = len(idx_train)
    n_dev      = len(idx_dev)
    n_test     = len(idx_test)
    n_traindev = len(idx_traindev)

    print(f"\n{'='*60}")
    print(f"  {case_label}")
    print(f"  Total={n_total}  Train={n_train}  Dev={n_dev}  Test={n_test}")
    print(f"{'='*60}")

    # ── Dev-set model selection ─────────────────────────────────────────────
    selection_rows: list[dict] = []

    for subset_label, cols in tasks:
        X_all_sub = df[cols].values
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_all_sub[idx_train])
        X_dev_s   = scaler.transform(X_all_sub[idx_dev])

        for cfg_label, cfg in NETWORK_CONFIGS.items():
            print(f"  [DEV]  {subset_label:20s}  {cfg_label}")
            clf = MLPClassifier(
                hidden_layer_sizes=cfg,
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="adam",
                early_stopping=early_stopping,
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

    # Sort by dev accuracy descending
    selection_rows.sort(key=lambda r: r["dev_acc"], reverse=True)
    best = selection_rows[0]

    print(f"\n  Best on dev: {best['subset_label']} | {best['cfg_label']}"
          f"  (dev={best['dev_acc']:.4f})")

    # ── Retrain winner on train+dev, evaluate on test ───────────────────────
    print(f"  Retraining on train+dev ({n_traindev} passages)...")
    X_best_sub  = df[best["cols"]].values
    scaler_final = StandardScaler()
    X_traindev_s = scaler_final.fit_transform(X_best_sub[idx_traindev])
    X_test_s     = scaler_final.transform(X_best_sub[idx_test])

    final_clf = MLPClassifier(
        hidden_layer_sizes=best["cfg"],
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="adam",
        early_stopping=early_stopping,
    )
    final_clf.fit(X_traindev_s, y_traindev)
    test_acc = final_clf.score(X_test_s, y_test)

    test_report = classification_report(
        y_test,
        final_clf.predict(X_test_s),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    test_df = pd.DataFrame(test_report).T
    display_rows = list(le.classes_) + ["macro avg", "weighted avg"]
    test_table = test_df.loc[display_rows]

    print(f"  Test accuracy: {test_acc:.4f}")
    print(test_table.to_string())

    # ── Build markdown ──────────────────────────────────────────────────────
    md  = f"# MLP Authorship Classification — {case_label}\n\n"

    md += "## Data Split\n\n"
    md += "| Set | Passages | Proportion |\n"
    md += "|-----|----------|------------|\n"
    md += f"| Train     | {n_train}    | {n_train/n_total:.0%} |\n"
    md += f"| Dev       | {n_dev}      | {n_dev/n_total:.0%}   |\n"
    md += f"| Test      | {n_test}     | {n_test/n_total:.0%}  |\n"
    md += f"| **Total** | **{n_total}**| 100%      |\n\n"

    md += "## Dev Set — Model Selection\n\n"
    md += ("All feature-subset × architecture combinations ranked by dev accuracy. "
           "Best configuration is retrained on train+dev and evaluated on the test set.\n\n")
    md += "| Rank | Feature Subset | Architecture | Train Acc | Dev Acc |\n"
    md += "|------|----------------|-------------|-----------|----------|\n"
    for i, row in enumerate(selection_rows, 1):
        marker = " ✓" if i == 1 else ""
        md += (f"| {i} | {row['subset_label']} | {row['cfg_label']} "
               f"| {row['train_acc']:.4f} | {row['dev_acc']:.4f}{marker} |\n")

    md += f"\n**Best model:** {best['subset_label']} · {best['cfg_label']}"
    md += f" — Dev accuracy: **{best['dev_acc']:.4f}**\n\n"

    md += "## Final Test Set Results\n\n"
    md += (f"Retrained on train+dev ({n_traindev} passages) using "
           f"**{best['subset_label']}**, **{best['cfg_label']}**.\n\n")
    md += f"**Test Accuracy: {test_acc:.4f}**\n\n"
    md += test_table.to_markdown() + "\n"

    return md


# ==========================================
# MAIN
# ==========================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP authorship classifier — dev/test evaluation")
    p.add_argument("--input",  default="author_features_extracted_full.csv",
                   help="Full feature CSV (default: author_features_extracted_full.csv)")
    p.add_argument("--early-stopping", action="store_true",
                   help="Enable early stopping during MLP training")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading {args.input}...")
    df_full = pd.read_csv(args.input)
    print(f"  {len(df_full)} rows loaded")

    print("\nRemoving outliers for second case...")
    df_clean = remove_outliers(df_full)
    print(f"  {len(df_full)} -> {len(df_clean)} rows after outlier removal")

    cases = [
        (df_full,  "With Outliers",    "results_with_outliers.md"),
        (df_clean, "Without Outliers", "results_without_outliers.md"),
    ]

    for df, label, outfile in cases:
        md = run_experiment(df, label, early_stopping=args.early_stopping)
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
