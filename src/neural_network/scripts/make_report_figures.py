"""
Generate report figures for the open-set NN authorship-attribution write-up.

No outlier removal: every experiment uses the FULL feature matrix.
Reproduces the validation-set grid search for the 6-class (open-set) task, then saves
to ../outputs/:

  validation_selection.png   depth_effect.png   confusion.png

It also prints the validation grid and final test metrics used in the .tex report.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

NN_DIR       = Path(__file__).resolve().parent.parent   # src/neural_network/
DATA         = NN_DIR / "data"
PROJECT_ROOT = NN_DIR.parents[1]                         # Stylometry/
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import extract_features  # noqa: E402

OUT = NN_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette (matches kmeans_report.tex) ───────────────────────────────────────
ACCENT    = "#2E5A88"
ACCENTRED = "#C44E52"
SOFTGRAY  = "#F5F5F7"
BLUES = LinearSegmentedColormap.from_list("accentblues", ["#FFFFFF", ACCENT])

# ── Config (mirrors neural_network_code.py) ───────────────────────────────────
# Features ranked by Random Forest permutation importance, most informative
# first (see rf_align_check.py and neural_network_code.py). The top-30 / top-50
# subsets are the first 30 / 50 entries.
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

NETWORK_CONFIGS = {
    "Depth 1":  (64,),
    "Depth 3":  (64, 64, 64),
    "Depth 10": tuple([64] * 10),
    "Depth 50": tuple([64] * 50),
}
DEPTH_OF = {"Depth 1": 1, "Depth 3": 3, "Depth 10": 10, "Depth 50": 50}

FEATURE_SUBSETS = [30, 50]
PATIENCE        = 15
BATCH_SIZE      = 32
MAX_ITER        = 500
RANDOM_STATE    = 42
METADATA_COLS   = {"author", "passage_id"}
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.60, 0.20, 0.20

NONE_LABEL = "none_of_the_5_authors"
FIVE_AUTHORS_STEMS = {
    "eliezer_yudkowsky", "johnswentworth", "raemon", "scottalexander", "zvi",
}
CLEANED_35_DIR = DATA / "cleaned_35"
N_NONE_AUTHORS, N_ARTICLES_PER_NONE_AUTHOR = 15, 10


# ── Data helpers ──────────────────────────────────────────────────────────────
def get_ordered_features(df):
    available = set(df.columns) - METADATA_COLS
    return [f for f in FEATURE_RANKING if f in available]


def make_eval_tasks(df, ordered):
    all_cols = [c for c in df.columns if c not in METADATA_COLS]
    tasks = [(f"Top {k}", ordered[:k]) for k in FEATURE_SUBSETS]
    tasks.append((f"All {len(all_cols)}", all_cols))
    return tasks


def build_none_class_df(seed=42):
    rng = random.Random(seed)
    candidates = [
        fp.stem for fp in sorted(CLEANED_35_DIR.glob("*.json"))
        if fp.stem not in FIVE_AUTHORS_STEMS
    ]
    chosen = rng.sample(candidates, N_NONE_AUTHORS)
    rows, pid = [], 0
    for stem in chosen:
        arts = json.loads((CLEANED_35_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        for art in rng.sample(arts, min(N_ARTICLES_PER_NONE_AUTHOR, len(arts))):
            feats = extract_features(art["text"])
            feats["author"], feats["passage_id"] = NONE_LABEL, pid
            rows.append(feats); pid += 1
    return pd.DataFrame(rows).fillna(0.0)


def load_dataset():
    """Five known authors + the synthetic ``none`` class -> 873 passages, 6 classes."""
    df5 = pd.read_csv(DATA / "author_features_extracted_full.csv")
    dfn = build_none_class_df(seed=42)
    cols = sorted(set(df5.columns) | set(dfn.columns))
    for c in cols:
        if c not in df5.columns:  df5[c] = 0.0
        if c not in dfn.columns:  dfn[c] = 0.0
    return pd.concat([df5[cols], dfn[cols]], ignore_index=True)


# ── Experiment ────────────────────────────────────────────────────────────────
def run_grid(df):
    le = LabelEncoder()
    y = le.fit_transform(df["author"])
    ordered = get_ordered_features(df)
    tasks = make_eval_tasks(df, ordered)

    idx = np.arange(len(df))
    idx_td, idx_te, y_td, y_te = train_test_split(
        idx, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_STATE)
    idx_tr, idx_val, y_tr, y_val = train_test_split(
        idx_td, y_td, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        stratify=y_td, random_state=RANDOM_STATE)

    grid, rows = {}, []
    for sub_label, cols in tasks:
        Xs = df[cols].values
        sc = StandardScaler()
        Xtr, Xval = sc.fit_transform(Xs[idx_tr]), sc.transform(Xs[idx_val])
        for cfg_label, cfg in NETWORK_CONFIGS.items():
            clf = MLPClassifier(hidden_layer_sizes=cfg, max_iter=MAX_ITER,
                                batch_size=BATCH_SIZE, random_state=RANDOM_STATE,
                                solver="adam", early_stopping=True,
                                n_iter_no_change=PATIENCE)
            clf.fit(Xtr, y_tr)
            acc = clf.score(Xval, y_val)
            grid[(sub_label, cfg_label)] = acc
            rows.append({"sub": sub_label, "cfg": cfg_label, "cols": cols,
                         "cfgt": cfg, "val": acc})
    best = max(rows, key=lambda r: r["val"])

    # Retrain winner on train+validation
    Xs = df[best["cols"]].values
    sc = StandardScaler()
    Xtd, Xte = sc.fit_transform(Xs[idx_td]), sc.transform(Xs[idx_te])
    clf = MLPClassifier(hidden_layer_sizes=best["cfgt"], max_iter=MAX_ITER,
                        batch_size=BATCH_SIZE, random_state=RANDOM_STATE,
                        solver="adam", early_stopping=True,
                        n_iter_no_change=PATIENCE)
    clf.fit(Xtd, y_td)
    y_pred = clf.predict(Xte)
    test_acc = clf.score(Xte, y_te)
    rep = classification_report(y_te, y_pred, target_names=le.classes_,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n=== open-set 6-class (full) ===")
    print(f"  splits: train={len(idx_tr)} val={len(idx_val)} test={len(idx_te)} total={len(df)}")
    print(f"  best: {best['sub']} x {best['cfg']}  val={best['val']:.4f}")
    print(f"  test acc={test_acc:.4f}  wf1={rep['weighted avg']['f1-score']:.4f}")
    print("  validation grid:")
    for sub_label, _ in tasks:
        cells = "  ".join(f"{cfg}={grid[(sub_label, cfg)]:.4f}" for cfg in NETWORK_CONFIGS)
        print(f"    {sub_label:8s} {cells}")

    return {
        "le": le, "grid": grid, "best": best, "tasks": tasks,
        "test_acc": test_acc, "wf1": rep["weighted avg"]["f1-score"],
        "cm": cm, "report": rep,
        "n_total": len(df), "n_train": len(idx_tr), "n_val": len(idx_val),
        "n_test": len(idx_te),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_val_heatmap(res, path):
    tasks = res["tasks"]
    cfgs = list(NETWORK_CONFIGS.keys())
    row_labels = [t[0] for t in tasks][::-1]   # All on top
    M = np.array([[res["grid"][(r, c)] for c in cfgs] for r in row_labels])
    best_key = (res["best"]["sub"], res["best"]["cfg"])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(M, cmap=BLUES, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cfgs)), cfgs)
    ax.set_yticks(range(len(row_labels)), row_labels)
    for i, r in enumerate(row_labels):
        for j, c in enumerate(cfgs):
            star = "★" if (r, c) == best_key else ""
            txt = f"{M[i, j]:.3f}{star}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.55 else "#333333",
                    fontweight="bold" if (r, c) == best_key else "normal")
    ax.set_xlabel("Network depth"); ax.set_ylabel("Feature subset")
    ax.set_title("Validation accuracy", color=ACCENT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Validation accuracy", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_depth_effect(res, path):
    cfgs = list(NETWORK_CONFIGS.keys())
    depths = [DEPTH_OF[c] for c in cfgs]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for sub_label, _ in res["tasks"]:
        accs = [res["grid"][(sub_label, c)] for c in cfgs]
        ax.plot(depths, accs, marker="o", lw=1.8, label=sub_label)
    ax.set_xscale("log"); ax.set_xticks(depths, [str(d) for d in depths])
    ax.set_xlabel("Network depth (hidden layers, log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Effect of network depth", color=ACCENT, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", title="Features")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confusion(res, path):
    cm = res["cm"]
    names = [n[:16] for n in res["le"].classes_]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm, cmap=BLUES)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > vmax * 0.5 else "#333333")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset (no outlier removal) ...")
    df = load_dataset()

    res = run_grid(df)

    plot_val_heatmap(res, OUT / "validation_selection.png")
    plot_depth_effect(res, OUT / "depth_effect.png")
    plot_confusion(res, OUT / "confusion.png")

    print(f"\nSaved 3 figures to {OUT}")


if __name__ == "__main__":
    main()
