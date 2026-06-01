"""
Generate report figures for the MLP authorship-attribution write-up.

No outlier removal: every experiment uses the FULL feature matrix.
Reproduces the dev-set grid search for both the 5-class and 6-class
(open-set) tasks, then saves to outputs/neural_network/:

  dev_selection_5class.png   dev_selection_6class.png
  depth_effect.png
  confusion_5class.png       confusion_6class.png
  roc_5class.png             roc_6class.png

It also prints the dev grids and final test metrics used in the .tex report.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import extract_features  # noqa: E402

OUT = ROOT / "outputs" / "neural_network"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette (matches kmeans_report.tex) ───────────────────────────────────────
ACCENT    = "#2E5A88"
ACCENTRED = "#C44E52"
SOFTGRAY  = "#F5F5F7"
BLUES = LinearSegmentedColormap.from_list("accentblues", ["#FFFFFF", ACCENT])

# ── Config (mirrors neural_network_code.py) ───────────────────────────────────
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
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.60, 0.20, 0.20

NONE_LABEL = "none_of_the_5_authors"
FIVE_AUTHORS_STEMS = {
    "eliezer_yudkowsky", "johnswentworth", "raemon", "scottalexander", "zvi",
}
CLEANED_35_DIR = ROOT / "dataset" / "lesswrong_large" / "cleaned_35"
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


def load_5class():
    return pd.read_csv(Path(__file__).resolve().parent / "author_features_extracted_full.csv")


def load_6class():
    df5 = load_5class()
    dfn = build_none_class_df(seed=42)
    cols = sorted(set(df5.columns) | set(dfn.columns))
    for c in cols:
        if c not in df5.columns:  df5[c] = 0.0
        if c not in dfn.columns:  dfn[c] = 0.0
    return pd.concat([df5[cols], dfn[cols]], ignore_index=True)


# ── Experiment ────────────────────────────────────────────────────────────────
def run_grid(df, task_label):
    le = LabelEncoder()
    y = le.fit_transform(df["author"])
    ordered = get_ordered_features(df)
    tasks = make_eval_tasks(df, ordered)

    idx = np.arange(len(df))
    idx_td, idx_te, y_td, y_te = train_test_split(
        idx, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_STATE)
    idx_tr, idx_dv, y_tr, y_dv = train_test_split(
        idx_td, y_td, test_size=DEV_RATIO / (TRAIN_RATIO + DEV_RATIO),
        stratify=y_td, random_state=RANDOM_STATE)

    grid, rows = {}, []
    for sub_label, cols in tasks:
        Xs = df[cols].values
        sc = StandardScaler()
        Xtr, Xdv = sc.fit_transform(Xs[idx_tr]), sc.transform(Xs[idx_dv])
        for cfg_label, cfg in NETWORK_CONFIGS.items():
            clf = MLPClassifier(hidden_layer_sizes=cfg, max_iter=MAX_ITER,
                                batch_size=BATCH_SIZE, random_state=RANDOM_STATE,
                                solver="adam", early_stopping=True,
                                n_iter_no_change=PATIENCE)
            clf.fit(Xtr, y_tr)
            acc = clf.score(Xdv, y_dv)
            grid[(sub_label, cfg_label)] = acc
            rows.append({"sub": sub_label, "cfg": cfg_label, "cols": cols,
                         "cfgt": cfg, "dev": acc})
    best = max(rows, key=lambda r: r["dev"])

    # Retrain winner on train+dev
    Xs = df[best["cols"]].values
    sc = StandardScaler()
    Xtd, Xte = sc.fit_transform(Xs[idx_td]), sc.transform(Xs[idx_te])
    clf = MLPClassifier(hidden_layer_sizes=best["cfgt"], max_iter=MAX_ITER,
                        batch_size=BATCH_SIZE, random_state=RANDOM_STATE,
                        solver="adam", early_stopping=True,
                        n_iter_no_change=PATIENCE)
    clf.fit(Xtd, y_td)
    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)
    test_acc = clf.score(Xte, y_te)
    rep = classification_report(y_te, y_pred, target_names=le.classes_,
                                output_dict=True, zero_division=0)
    roc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n=== {task_label} ===")
    print(f"  splits: train={len(idx_tr)} dev={len(idx_dv)} test={len(idx_te)} total={len(df)}")
    print(f"  best: {best['sub']} x {best['cfg']}  dev={best['dev']:.4f}")
    print(f"  test acc={test_acc:.4f}  wf1={rep['weighted avg']['f1-score']:.4f}  roc-auc={roc:.4f}")
    print("  dev grid:")
    for sub_label, _ in tasks:
        cells = "  ".join(f"{cfg}={grid[(sub_label, cfg)]:.4f}" for cfg in NETWORK_CONFIGS)
        print(f"    {sub_label:8s} {cells}")

    return {
        "le": le, "grid": grid, "best": best, "tasks": tasks,
        "test_acc": test_acc, "wf1": rep["weighted avg"]["f1-score"],
        "roc": roc, "cm": cm, "report": rep, "y_te": y_te, "y_prob": y_prob,
        "n_total": len(df), "n_train": len(idx_tr), "n_dev": len(idx_dv),
        "n_test": len(idx_te),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_dev_heatmap(res, title, path):
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
    ax.set_title(title, color=ACCENT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Dev accuracy", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_depth_effect(res5, res6, path):
    cfgs = list(NETWORK_CONFIGS.keys())
    depths = [DEPTH_OF[c] for c in cfgs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, res, title in ((axes[0], res5, "5-class"), (axes[1], res6, "6-class")):
        for sub_label, _ in res["tasks"]:
            accs = [res["grid"][(sub_label, c)] for c in cfgs]
            ax.plot(depths, accs, marker="o", lw=1.8, label=sub_label)
        ax.set_xscale("log"); ax.set_xticks(depths, [str(d) for d in depths])
        ax.set_xlabel("Network depth (hidden layers, log scale)")
        ax.set_title(title, color=ACCENT, fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Dev accuracy")
    axes[1].legend(fontsize=8, loc="lower left", title="Features")
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


def plot_roc(res, title, path):
    y_true, y_prob = res["y_te"], res["y_prob"]
    names = list(res["le"].classes_)
    n = len(names)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(7, 5.6))
    fpr_all, tpr_all = {}, {}
    for i, name in enumerate(names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr_all[i], tpr_all[i] = fpr, tpr
        ax.plot(fpr, tpr, lw=1.5, label=f"{name}  (AUC={auc(fpr, tpr):.3f})")
    all_fpr = np.unique(np.concatenate(list(fpr_all.values())))
    mean_tpr = np.mean([np.interp(all_fpr, fpr_all[i], tpr_all[i]) for i in range(n)], axis=0)
    ax.plot(all_fpr, mean_tpr, "k--", lw=2, label=f"Macro avg  (AUC={auc(all_fpr, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title, color=ACCENT, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading datasets (no outlier removal) ...")
    df5 = load_5class()
    df6 = load_6class()

    res5 = run_grid(df5, "5-class (full)")
    res6 = run_grid(df6, "6-class (full)")

    plot_dev_heatmap(res5, "Dev accuracy - 5-class MLP", OUT / "dev_selection_5class.png")
    plot_dev_heatmap(res6, "Dev accuracy - 6-class MLP", OUT / "dev_selection_6class.png")
    plot_depth_effect(res5, res6, OUT / "depth_effect.png")
    plot_confusion(res5, OUT / "confusion_5class.png")
    plot_confusion(res6, OUT / "confusion_6class.png")
    plot_roc(res5, "ROC Curves - 5-class MLP", OUT / "roc_5class.png")
    plot_roc(res6, "ROC Curves - 6-class MLP", OUT / "roc_6class.png")
    print(f"\nSaved 7 figures to {OUT}")


if __name__ == "__main__":
    main()
