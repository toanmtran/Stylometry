"""
Effect of N on clustering quality (ARI vs document length).
Two experiments with K=5, M=15 fixed throughout.

  cleaned_10: 10 authors × 40 articles each (≥1500 words).
              N in {100, 200, 500, 1000, 1500}.
              Each trial randomly samples K=5 authors from 10, then M=15
              articles per selected author.

  cleaned_5:  5 authors × 29–62 articles each (≥3000 words).
              N in {100, 200, 500, 1000, 1500, 2000, 2500, 3000}.
              K=5 equals the full author set, so each trial only samples
              M=15 articles per author (no author selection).

Conceptual illustration — why does ARI drop at short N?

  As N shrinks, each document is too small a sample of its author's style —
  feature frequencies become noisy estimates of the author's true rate, so
  documents from different authors overlap in feature space and K-means
  struggles to separate them.

  We illustrate this with one author (zvi) and two features (fw_you,
  word_len_1_frac):

    1. Concatenate all of zvi's articles in `cleaned_5` into one long text.
    2. For each window size N in CONCEPT_N_VALUES, sample CONCEPT_N_WINDOWS
       random windows of exactly N whitespace-separated tokens.
    3. Compute the two features on each window, record mean and std.
    4. Plot feature value (mean ± std) vs N — one panel per feature.

  Expected shape: the mean converges to zvi's long-run rate while the std
  shrinks roughly like 1/sqrt(N). Short windows = high variance = authors
  bleed into each other's feature distributions.

Outputs:
  outputs/kmeans/effect_of_n/ari_vs_n_cleaned10.png
  outputs/kmeans/effect_of_n/ari_vs_n_cleaned5.png
  outputs/kmeans/effect_of_n/feature_stability_zvi.png
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.clustering import build_feature_matrix
from src.corpus import load_corpus
from src.features import extract_features_batch
from src.preprocess import canonicalize_typography
from src.viz import plot_ari_sweep


# ── Sweep experiment constants ────────────────────────────────────────────────
K = 5
M = 15
N_TRIALS = 50
N_INIT = 10
GLOBAL_SEED = 42

N_VALUES_10 = [100, 200, 500, 1000, 1500]
N_VALUES_5 = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]

# ── Concept illustration constants ────────────────────────────────────────────
CONCEPT_AUTHOR = "zvi"
CONCEPT_FEATURES = ["fw_you", "word_len_1_frac"]
CONCEPT_N_VALUES = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
CONCEPT_N_WINDOWS = 20

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR = _PROJECT_ROOT / "outputs" / "kmeans" / "effect_of_n"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH_10 = OUT_DIR / "ari_vs_n_10authors.png"
FIG_PATH_5 = OUT_DIR / "ari_vs_n_5authors.png"
FIG_PATH_CONCEPT = OUT_DIR / "feature_stability_zvi.png"


def _build_author_index(authors: list[str]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for i, a in enumerate(authors):
        idx.setdefault(a, []).append(i)
    return idx


def run_ari_sweep(
    corpus,
    *,
    k: int,
    m: int,
    n_values: list[int],
    n_trials: int,
    sample_authors: bool,
    seed: int,
) -> dict[int, list[float]]:
    """Sweep N values, returning per-N ARI lists.

    For each N:
      1. Truncate all texts to N words, build the full feature matrix once.
      2. For n_trials trials:
         - If sample_authors: randomly pick k unique authors.
         - Else: use all authors (requires exactly k authors in corpus).
         - Randomly pick m docs per selected author.
         - Run K-means++ on that subset, record ARI.
    """
    unique_authors = sorted(set(corpus.authors))
    if not sample_authors and len(unique_authors) != k:
        raise ValueError(
            f"sample_authors=False but corpus has {len(unique_authors)} authors, "
            f"expected exactly {k}."
        )

    results: dict[int, list[float]] = {}
    for n_words in n_values:
        sub = corpus.truncate_to(n_words)
        author_to_rows = _build_author_index(sub.authors)
        X, _ = build_feature_matrix(
            sub.texts, sub.labels,
            source=sub.source, cleaned_dir=sub.cleaned_dir,
        )

        scores: list[float] = []
        for t in range(n_trials):
            trial_rng = random.Random(seed * 10_000_000 + n_words * 10_000 + t)

            if sample_authors:
                group = tuple(trial_rng.sample(unique_authors, k))
            else:
                group = tuple(unique_authors)

            rows: list[int] = []
            for a in group:
                rows.extend(trial_rng.sample(author_to_rows[a], m))
            rows.sort()

            sub_X = X[rows]
            a_to_id = {a: i for i, a in enumerate(group)}
            true_ids = [a_to_id[sub.authors[r]] for r in rows]

            km_seed = seed * 1_000_000 + n_words * 1_000 + t
            pred = KMeans(
                n_clusters=k, init="k-means++",
                n_init=N_INIT, random_state=km_seed,
            ).fit_predict(sub_X)
            ari = adjusted_rand_score(true_ids, pred)
            scores.append(ari)
            print(f"  N={n_words:5d}  trial {t+1:2d}/{n_trials}  "
                  f"ARI={ari:+.3f}  authors={list(group)}")
        results[n_words] = scores
    return results


def _print_summary(results: dict[int, list[float]], n_values: list[int]) -> None:
    print(f"  {'N':>5}  {'n':>3}  {'mean':>6}  {'std':>6}  {'min':>6}  {'max':>6}")
    for n_words in n_values:
        s = results[n_words]
        print(f"  {n_words:>5d}  {len(s):>3d}  "
              f"{np.mean(s):>+6.3f}  {np.std(s):>6.3f}  "
              f"{min(s):>+6.3f}  {max(s):>+6.3f}")


def sample_windows(tokens: list[str], n: int, n_windows: int,
                   rng: random.Random) -> list[str]:
    """Draw `n_windows` random contiguous slices of exactly `n` tokens from
    `tokens`, joined back into whitespace-separated strings."""
    max_start = len(tokens) - n
    if max_start < 0:
        raise ValueError(
            f"sample_windows: need {n} tokens but only {len(tokens)} available"
        )
    windows: list[str] = []
    for _ in range(n_windows):
        start = rng.randint(0, max_start)
        windows.append(" ".join(tokens[start : start + n]))
    return windows


def run_cleaned10() -> None:
    print("=" * 60)
    print("Experiment 1 — cleaned_10  (10 authors × 40 articles, ≥1500w)")
    print("=" * 60)
    corpus = load_corpus("lesswrong_regular", cleaned_dir="cleaned_10")
    n_authors = len(set(corpus.authors))
    author_to_rows = _build_author_index(corpus.authors)
    print(f"  {len(corpus)} docs, {n_authors} authors")
    for a, rows in sorted(author_to_rows.items()):
        print(f"    {a}: {len(rows)} docs")

    if min(corpus.word_counts) < max(N_VALUES_10):
        print(f"ERROR: shortest doc = {min(corpus.word_counts)}w "
              f"but max N = {max(N_VALUES_10)}")
        sys.exit(1)
    for a, rows in author_to_rows.items():
        if len(rows) < M:
            print(f"ERROR: {a} has {len(rows)} docs, need ≥{M}")
            sys.exit(1)

    print(f"\nSweeping N in {N_VALUES_10}  "
          f"(K={K}, M={M}, {N_TRIALS} trials/N, n_init={N_INIT})")
    print(f"  Sampling K={K} authors from {n_authors} per trial\n")

    results = run_ari_sweep(
        corpus, k=K, m=M,
        n_values=N_VALUES_10, n_trials=N_TRIALS,
        sample_authors=True, seed=GLOBAL_SEED,
    )

    print("\n── ARI vs N summary (cleaned_10) ──")
    _print_summary(results, N_VALUES_10)

    plot_ari_sweep(
        results, FIG_PATH_10,
        title=f"K-means — Effects of Article Length (LessWrong {n_authors} Authors, authors = {K}, articles = {M})",
        xlabel="N  (words per article)",
        x_symbol="N",
        xscale="log",
    )
    print(f"\nSaved -> {FIG_PATH_10.relative_to(_PROJECT_ROOT)}")


def run_cleaned5() -> None:
    print("=" * 60)
    print("Experiment 2 — cleaned_5  (5 authors × 29–62 articles, ≥3000w)")
    print("=" * 60)
    corpus = load_corpus("lesswrong_regular", cleaned_dir="cleaned_5")
    n_authors = len(set(corpus.authors))
    author_to_rows = _build_author_index(corpus.authors)
    print(f"  {len(corpus)} docs, {n_authors} authors")
    for a, rows in sorted(author_to_rows.items()):
        print(f"    {a}: {len(rows)} docs")

    if n_authors != K:
        print(f"ERROR: expected exactly {K} authors in cleaned_5, got {n_authors}")
        sys.exit(1)
    if min(corpus.word_counts) < max(N_VALUES_5):
        print(f"ERROR: shortest doc = {min(corpus.word_counts)}w "
              f"but max N = {max(N_VALUES_5)}")
        sys.exit(1)
    for a, rows in author_to_rows.items():
        if len(rows) < M:
            print(f"ERROR: {a} has {len(rows)} docs, need ≥{M}")
            sys.exit(1)

    print(f"\nSweeping N in {N_VALUES_5}  "
          f"(K={K}, M={M}, {N_TRIALS} trials/N, n_init={N_INIT})")
    print(f"  All {K} authors used; only article sampling randomised\n")

    results = run_ari_sweep(
        corpus, k=K, m=M,
        n_values=N_VALUES_5, n_trials=N_TRIALS,
        sample_authors=False, seed=GLOBAL_SEED,
    )

    print("\n── ARI vs N summary (cleaned_5) ──")
    _print_summary(results, N_VALUES_5)

    plot_ari_sweep(
        results, FIG_PATH_5,
        title=f"K-means — Effects of Article Length (LessWrong {n_authors} Authors, authors = {K}, articles = {M})",
        xlabel="N  (words per document)",
        x_symbol="N",
        xscale="log",
    )
    print(f"\nSaved -> {FIG_PATH_5.relative_to(_PROJECT_ROOT)}")


def run_concept() -> None:
    print("=" * 60)
    print("Feature stability vs window size  (author: zvi)")
    print("=" * 60)
    corpus = load_corpus("lesswrong_regular", cleaned_dir="cleaned_5")
    docs = [t for t, a in zip(corpus.texts, corpus.authors) if a == CONCEPT_AUTHOR]
    if not docs:
        print(f"ERROR: no documents found for author {CONCEPT_AUTHOR!r}")
        sys.exit(1)
    print(f"  {len(docs)} articles for {CONCEPT_AUTHOR}")

    # Canonicalise typography before sampling so feature rates don't depend
    # on whether zvi's publisher converts ' → '.
    concat = canonicalize_typography(" ".join(docs))
    tokens = concat.split()
    print(f"  concatenated text: {len(tokens):,} whitespace-separated tokens")
    if len(tokens) < max(CONCEPT_N_VALUES):
        print(f"ERROR: only {len(tokens)} tokens, need ≥ {max(CONCEPT_N_VALUES)}")
        sys.exit(1)

    rng = random.Random(GLOBAL_SEED)
    results: dict[str, dict[int, list[float]]] = {f: {} for f in CONCEPT_FEATURES}

    print(f"\nSampling {CONCEPT_N_WINDOWS} windows per N, computing features …")
    for n_words in CONCEPT_N_VALUES:
        windows = sample_windows(tokens, n_words, CONCEPT_N_WINDOWS, rng)
        labels = [f"{CONCEPT_AUTHOR}_N{n_words}_{i:02d}" for i in range(CONCEPT_N_WINDOWS)]
        df = extract_features_batch(windows, labels)
        for feat in CONCEPT_FEATURES:
            if feat not in df.columns:
                print(f"ERROR: feature {feat!r} missing from extractor output")
                sys.exit(1)
            vals = df[feat].to_numpy()
            results[feat][n_words] = vals.tolist()
        print(f"  N={n_words:5d}  " +
              "  ".join(f"{feat}: {np.mean(results[feat][n_words]):.4f} "
                        f"± {np.std(results[feat][n_words]):.4f}"
                        for feat in CONCEPT_FEATURES))

    fig, axes = plt.subplots(len(CONCEPT_FEATURES), 1,
                             figsize=(9.0, 4.5 * len(CONCEPT_FEATURES)),
                             sharex=True)
    if len(CONCEPT_FEATURES) == 1:
        axes = [axes]
    jitter_rng = np.random.RandomState(GLOBAL_SEED)

    for ax, feat in zip(axes, CONCEPT_FEATURES):
        means = [float(np.mean(results[feat][n])) for n in CONCEPT_N_VALUES]
        stds = [float(np.std(results[feat][n])) for n in CONCEPT_N_VALUES]

        for n in CONCEPT_N_VALUES:
            ys = results[feat][n]
            jitter = jitter_rng.uniform(0.92, 1.08, size=len(ys))
            xs = n * jitter
            ax.scatter(xs, ys, color="#C44E52", alpha=0.35, s=24, zorder=2,
                       label="sample" if n == CONCEPT_N_VALUES[0] else None)
        ax.errorbar(CONCEPT_N_VALUES, means, yerr=stds, fmt="o-", capsize=5,
                    linewidth=2, color="#4C72B0",
                    markeredgecolor="black", markersize=7,
                    label="mean ± std", zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(CONCEPT_N_VALUES)
        ax.set_xticklabels([str(n) for n in CONCEPT_N_VALUES], rotation=30, ha="right")
        ax.minorticks_off()
        ax.set_xlim(CONCEPT_N_VALUES[0] * 0.8, CONCEPT_N_VALUES[-1] * 1.25)
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("N  (window size, words)")
    fig.suptitle(
        f"Feature Stability — Author: {CONCEPT_AUTHOR}\n"
        f"({CONCEPT_N_WINDOWS} random windows per N)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_PATH_CONCEPT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {FIG_PATH_CONCEPT.relative_to(_PROJECT_ROOT)}")


def main() -> None:
    run_cleaned10()
    print()
    run_cleaned5()
    print()
    run_concept()


if __name__ == "__main__":
    main()
