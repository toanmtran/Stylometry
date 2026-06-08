"""
MLP Authorship Classifier - open-set (6-class) attribution.

Five known LessWrong authors plus a synthetic sixth class,
``none_of_the_5_authors``, so the model can say "written by none of the five"
instead of being forced to pick one.

none-class construction:
  - Randomly pick N_NONE_AUTHORS (15) authors from cleaned_35 that are NOT
    among the five known authors.
  - Sample N_ARTICLES_PER_NONE_AUTHOR (10) articles from each -> 150 passages.
  - Extract stylometric features and label every article "none_of_the_5_authors".
  - Merge with the five-author feature CSV.

Split 60/20/20, grid-search feature subset x depth on the validation set, retrain the
winner on train+validation, evaluate once on the test set. Writes results.md.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

NN_DIR       = Path(__file__).resolve().parent.parent   # src/neural_network/
DATA         = NN_DIR / "data"
OUTPUTS      = NN_DIR / "outputs"
PROJECT_ROOT = NN_DIR.parents[1]                         # Stylometry/
OUTPUTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from src.features import extract_features

# ==========================================
# CONFIGURATION
# ==========================================
# Features ranked by Random Forest permutation importance, most informative
# first. Produced by rf_align_check.py: RandomForestClassifier(n_estimators=500,
# max_depth=5, random_state=0) fit on an 80/20 stratified split, then
# permutation_importance on the held-out test split (random_state=0). The top-30
# and top-50 feature subsets are the first 30 / 50 entries of this list.
FEATURE_RANKING = [
    "punct_paren_rate", "punct_semicolon_rate", "honore_r", "fw_which",
    "fw_this", "punct_colon_rate", "fw_would", "fw_that", "n_tokens", "fw_but",
    "avg_paragraph_len", "word_len_2_frac", "max_sent_len", "n_vocab",
    "std_word_len", "pos_pron", "fw_their", "pos_conj", "pos_adv", "fw_you",
    "fw_for", "fw_his", "cat_discourse_rate", "fw_on", "fw_up", "hapax_ratio",
    "fw_all", "fw_who", "fw_me", "fw_out", "type_token_ratio", "word_len_1_frac",
    "fw_a", "hapax_dis_ratio", "fw_as", "punct_excl_rate", "punct_dquote_rate",
    "pos_prep", "char_i_rate", "cat_conj_rate", "n_sentences", "fw_to",
    "word_len_6_frac", "fw_by", "fw_we", "fw_what", "fw_at", "fw_so",
    "punct_period_rate", "word_len_3_frac", "word_len_5_frac", "pos_noun",
    "char_a_rate", "fw_go", "fw_get", "fw_have", "fw_in", "fw_if", "fw_about",
    "word_len_4_frac", "avg_sent_len", "min_sent_len", "max_word_len",
    "median_word_len", "avg_word_len", "fw_it", "fw_with", "fw_and",
    "median_sent_len", "std_sent_len", "brunet_w", "fw_the", "fw_be", "fw_i",
    "fw_he", "uppercase_ratio", "pos_det", "pos_verb", "punct_dash_rate",
    "fw_there", "fw_one", "fw_from", "fw_do", "fw_they", "fw_say", "fw_her",
    "fw_she", "fw_or", "fw_an", "fw_will", "fw_my", "char_u_rate", "char_o_rate",
    "pos_modal", "punct_comma_rate", "punct_question_rate", "cat_amplifier_rate",
    "cat_hedge_rate", "punct_apost_rate", "char_e_rate", "yule_k", "simpson_d",
    "fw_not", "pos_adj", "fw_of", "pronoun_start_rate", "contraction_rate",
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
VAL_RATIO    = 0.20
TEST_RATIO   = 0.20
RANDOM_STATE = 42
MAX_ITER     = 500

NONE_LABEL = "none_of_the_5_authors"

FIVE_AUTHORS_STEMS = {
    "eliezer_yudkowsky", "johnswentworth", "raemon", "scottalexander", "zvi",
}

CLEANED_35_DIR = DATA / "cleaned_35"

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
def get_ordered_features(df: pd.DataFrame) -> list[str]:
    available = set(df.columns) - METADATA_COLS
    return [f for f in FEATURE_RANKING if f in available]


def make_eval_tasks(df: pd.DataFrame, ordered: list[str]) -> list[tuple[str, list[str]]]:
    all_cols = [c for c in df.columns if c not in METADATA_COLS]
    tasks = [(f"Top {k} features", ordered[:k]) for k in FEATURE_SUBSETS]
    tasks.append((f"All {len(all_cols)} features", all_cols))
    return tasks


def confusion_matrix_md(cm: np.ndarray, names: list[str]) -> str:
    short = [n[:14] for n in names]
    header = "| Actual \\ Pred | " + " | ".join(f"**{s}**" for s in short) + " |"
    sep    = "|" + "---|" * (len(names) + 1)
    rows   = [header, sep]
    for i, row_vals in enumerate(cm):
        rows.append(f"| **{short[i]}** | " + " | ".join(str(v) for v in row_vals) + " |")
    return "\n".join(rows) + "\n"


def val_table_md(selection_rows: list[dict], best: dict) -> str:
    """2-D markdown table: rows = feature subsets, cols = architectures."""
    subset_labels = list(dict.fromkeys(r["subset_label"] for r in selection_rows))
    cfg_labels    = list(NETWORK_CONFIGS.keys())
    grid = {(r["subset_label"], r["cfg_label"]): r["val_acc"] for r in selection_rows}
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
def run_experiment(df: pd.DataFrame, none_authors: list[str]) -> str:
    le = LabelEncoder()
    y  = le.fit_transform(df["author"])
    ordered = get_ordered_features(df)
    tasks   = make_eval_tasks(df, ordered)

    idx = np.arange(len(df))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_STATE
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval, y_trainval,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        stratify=y_trainval, random_state=RANDOM_STATE,
    )

    n_total    = len(df)
    n_train    = len(idx_train)
    n_val      = len(idx_val)
    n_test     = len(idx_test)
    n_trainval = len(idx_trainval)

    none_total = int(df["author"].value_counts().get(NONE_LABEL, 0))
    print(f"\n{'='*60}")
    print(f"  MLP 6-Class Authorship Classification")
    print(f"  Total={n_total}  Train={n_train}  Val={n_val}  Test={n_test}")
    print(f"  none_of_the_5_authors rows: {none_total}")
    print(f"{'='*60}")

    # Validation-set model selection
    selection_rows: list[dict] = []
    for subset_label, cols in tasks:
        X_sub  = df[cols].values
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_sub[idx_train])
        X_val   = scaler.transform(X_sub[idx_val])
        for cfg_label, cfg in NETWORK_CONFIGS.items():
            print(f"  [VAL]  {subset_label:20s}  {cfg_label}  patience={PATIENCE}")
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
                "val_acc":      clf.score(X_val, y_val),
            })

    selection_rows.sort(key=lambda r: r["val_acc"], reverse=True)
    best = selection_rows[0]
    print(f"\n  Best on validation: {best['subset_label']} | {best['cfg_label']}"
          f"  (val={best['val_acc']:.4f})")

    # Retrain winner on train+validation, evaluate on test
    print(f"  Retraining on train+validation ({n_trainval} passages)...")
    X_best_sub   = df[best["cols"]].values
    scaler_final = StandardScaler()
    X_trainval_s = scaler_final.fit_transform(X_best_sub[idx_trainval])
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
    final_clf.fit(X_trainval_s, y_trainval)
    y_pred_test = final_clf.predict(X_test_s)
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

    wf1 = test_report["weighted avg"]["f1-score"]
    cm  = confusion_matrix(y_test, y_pred_test)

    print(f"  Test accuracy: {test_acc:.4f}  WF1: {wf1:.4f}")
    print(test_table.to_string())

    # Markdown
    md  = "# MLP 6-Class Authorship Classification\n\n"

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
    md += f"| Validation | {n_val}      | {n_val/n_total:.0%}   |\n"
    md += f"| Test      | {n_test}     | {n_test/n_total:.0%}  |\n"
    md += f"| **Total** | **{n_total}**| 100%      |\n\n"

    md += "## Validation Set - Model Selection\n\n"
    md += ("Validation accuracy for every feature-subset x architecture combination "
           f"(patience={PATIENCE}, batch_size={BATCH_SIZE}). "
           "Best cell marked with checkmark.\n\n")
    md += val_table_md(selection_rows, best)
    md += "\n"
    md += (f"**Best model:** {best['subset_label']} x {best['cfg_label']} "
           f"- Validation accuracy: **{best['val_acc']:.4f}**\n\n")

    md += "## Final Test Set Results\n\n"
    md += (f"Retrained on train+validation ({n_trainval} passages) using "
           f"**{best['subset_label']}**, **{best['cfg_label']}**.\n\n")

    md += "### Key Metrics\n\n"
    md += "| Metric | Value |\n|--------|-------|\n"
    md += f"| Accuracy    | {test_acc:.4f} |\n"
    md += f"| Weighted F1 | {wf1:.4f} |\n\n"

    md += "### Per-Class Report\n\n"
    md += test_table.to_markdown() + "\n\n"

    md += "### Confusion Matrix\n\n"
    md += "_Rows = actual, Columns = predicted._\n\n"
    md += confusion_matrix_md(cm, list(le.classes_)) + "\n"

    return md


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    input_csv = str(DATA / "author_features_extracted_full.csv")
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

    md     = run_experiment(df_full, none_authors)
    out_md = OUTPUTS / "results.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\nSaved: {out_md}")


if __name__ == "__main__":
    main()
