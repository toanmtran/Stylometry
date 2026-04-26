"""
Effect of M (docs per author) on clustering quality (ARI vs M).

K is fixed at 4. Doc length is fixed at DOC_LENGTH = 1526 words (the min
across the entire cleaned_10 corpus) for both datasets, so length is not
a confound when comparing across M values or datasets.

Two datasets are tested:

  cleaned_10 — 10 authors, 40 docs/author available after filtering.
    M in {5, 10, 15, 20, 25, 30}.
    Each trial randomly samples K=4 authors from 10, then M docs per author.
    Author-group sampling adds extra variance across trials.

  cleaned_4 — 4 authors (= K), 76-79 docs/author available after filtering.
    M in {5, 10, 15, 20, 25, 30, 40, 50, 60}.
    C(4,4)=1 so every trial uses all 4 authors; only docs are randomly
    sampled. Variance comes purely from doc selection + KMeans init.

Each M uses its own independent RNG seeded by (GLOBAL_SEED, m). Adding
or removing M values does not perturb the samples drawn for other M's.

Outputs:
  - outputs/kmeans/effect_of_m/ari_vs_m_10authors.png
  - outputs/kmeans/effect_of_m/ari_vs_m_4authors.png
"""

from __future__ import annotations

import itertools
import math
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
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.clustering import build_feature_matrix
from src.corpus import load_corpus
from src.viz import plot_ari_sweep


DOC_LENGTH = 1526  # min word count across cleaned_10; fixed for both datasets
K = 4
N_TRIALS = 50
N_INIT = 10
GLOBAL_SEED = 42

M_VALUES_10 = [5, 10, 15, 20, 25, 30]               # cleaned_10: 40 docs/author
M_VALUES_4  = [5, 10, 15, 20, 25, 30, 40, 50, 60]  # cleaned_4:  76-79 docs/author

OUT_DIR = _PROJECT_ROOT / "outputs" / "kmeans" / "effect_of_m"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH_10 = OUT_DIR / "ari_vs_m_10authors.png"
FIG_PATH_4  = OUT_DIR / "ari_vs_m_4authors.png"


def _sample_groups(
    unique_authors: list[str],
    *,
    k: int,
    n_trials: int,
    seed: int,
) -> list[tuple[str, ...]]:
    """Sample K-author groups for one M value.

    Without replacement when C(n_authors, K) ≥ n_trials; otherwise uses
    every unique group once and fills the remainder with replacement.
    """
    subsets = list(itertools.combinations(unique_authors, k))
    rng = random.Random(seed)
    if len(subsets) >= n_trials:
        return rng.sample(subsets, n_trials)
    groups = list(subsets)
    groups.extend(rng.choices(subsets, k=n_trials - len(subsets)))
    rng.shuffle(groups)
    return groups


def ari_sweep_m(
    X: np.ndarray,
    authors: list[str],
    k: int,
    m_values: list[int],
    *,
    n_trials: int = 30,
    n_init: int = 10,
    seed: int = 42,
) -> dict[int, list[float]]:
    """For each M, run `n_trials` sampled trials and return per-M ARI lists."""
    unique_authors = sorted(set(authors))
    author_to_rows: dict[str, list[int]] = {}
    for i, a in enumerate(authors):
        author_to_rows.setdefault(a, []).append(i)

    min_pool = min(len(author_to_rows[a]) for a in unique_authors)
    if max(m_values) > min_pool:
        raise ValueError(
            f"max M={max(m_values)} exceeds smallest author pool ({min_pool}). "
            f"Lower M or use a corpus with more docs per author."
        )

    results: dict[int, list[float]] = {}
    for m in m_values:
        chosen = _sample_groups(
            unique_authors, k=k, n_trials=n_trials, seed=seed * 1000 + m
        )
        scores: list[float] = []
        for t, group in enumerate(chosen):
            trial_rng = random.Random(seed * 10_000_000 + m * 10_000 + t)
            rows = []
            for a in group:
                rows.extend(trial_rng.sample(author_to_rows[a], m))
            rows.sort()

            sub_X = X[rows]
            a_to_id = {a: i for i, a in enumerate(group)}
            sub_ids = [a_to_id[authors[r]] for r in rows]
            labels = KMeans(
                n_clusters=k, init="k-means++",
                n_init=n_init,
                random_state=seed * 1_000_000 + m * 10_000 + t,
            ).fit_predict(sub_X)
            ari = adjusted_rand_score(sub_ids, labels)
            scores.append(ari)
            print(
                f"  M={m:2d}  trial {t+1:3d}/{n_trials}  ARI={ari:+.3f}  "
                f"n_docs={len(rows):3d}  authors={list(group)}"
            )
        results[m] = scores
    return results


def _print_summary(results: dict[int, list[float]], m_values: list[int]) -> None:
    print(f"  {'M':>3}  {'n':>3}  {'mean':>6}  {'std':>6}  {'min':>6}  {'max':>6}")
    for m in m_values:
        s = results[m]
        print(f"  {m:>3d}  {len(s):>3d}  "
              f"{np.mean(s):>+6.3f}  {np.std(s):>6.3f}  "
              f"{min(s):>+6.3f}  {max(s):>+6.3f}")


def _run_dataset(
    cleaned_dir: str,
    m_values: list[int],
    fig_path: Path,
    label: str,
) -> None:
    print(f"Loading {cleaned_dir} corpus …")
    corpus = load_corpus("lesswrong_regular", cleaned_dir=cleaned_dir)
    corpus = (
        corpus.filter(min_words=DOC_LENGTH)
              .truncate_to(DOC_LENGTH)
    )
    n_authors = len(set(corpus.authors))
    print(f"  {len(corpus)} docs × {corpus.truncated_to}w  "
          f"({n_authors} authors)")

    print("Building stylometric feature matrix …")
    X, _ = build_feature_matrix(
        corpus.texts, corpus.labels,
        source=corpus.source, cleaned_dir=corpus.cleaned_dir,
    )
    print(f"  X shape: {X.shape}")

    group_pool = math.comb(n_authors, K)
    print(f"\nARI sweep: K={K}, M in {m_values}, {N_TRIALS} trials per M")
    print(f"  group pool = C({n_authors},{K}) = {group_pool}")
    for m in m_values:
        print(f"    M={m:2d}: {m} docs/author, total={K * m} docs/trial")
    print(f"  KMeans: init='k-means++', n_init={N_INIT}")

    results = ari_sweep_m(
        X, corpus.authors, K, m_values,
        n_trials=N_TRIALS, n_init=N_INIT, seed=GLOBAL_SEED,
    )

    print(f"\n── ARI vs M summary ({label}) ──")
    _print_summary(results, m_values)

    plot_ari_sweep(
        results, fig_path,
        title=f"K-means — Effects of No. of Articles per Author (LessWrong {n_authors} Authors, authors = {K}, length = {DOC_LENGTH})",
        xlabel="M  (articles per author)",
        x_symbol="M",
    )
    print(f"Saved -> {fig_path.relative_to(_PROJECT_ROOT)}\n")


def main() -> None:
    print("=== [1/2] cleaned_10: 10 authors, K=4, sample 4 authors + M docs ===\n")
    _run_dataset("cleaned_10", M_VALUES_10, FIG_PATH_10, "10 authors")

    print("=== [2/2] cleaned_4: 4 authors = K, sample M docs only ===\n")
    _run_dataset("cleaned_4", M_VALUES_4, FIG_PATH_4, "4 authors")


if __name__ == "__main__":
    main()
