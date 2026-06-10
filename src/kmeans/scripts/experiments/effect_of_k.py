"""ARI vs number of clusters K on cleaned_35, with docs/author held constant."""

from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_KMEANS_DIR = _EXPERIMENT_DIR.parents[1]
_PROJECT_ROOT = _KMEANS_DIR.parents[1]
_DATA_DIR = _KMEANS_DIR / "data"
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

from src.kmeans.scripts.clustering import build_feature_matrix
from src.kmeans.scripts.corpus import load_corpus
from src.kmeans.scripts.viz import plot_ari_sweep


MIN_WORDS = 1500
DOC_LENGTH = 1500
K_VALUES = [2, 4, 6, 8, 10, 12]
N_TRIALS = 100
N_INIT = 10
M_PER_AUTHOR = 15
GLOBAL_SEED = 42

OUT_DIR = _KMEANS_DIR / "outputs" / "effect_of_k"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUT_DIR / "ari_vs_k.png"


_ENUM_THRESHOLD = 2_000_000


def _sample_groups(
    unique_authors: list[str],
    *,
    k: int,
    n_trials: int,
    seed: int,
) -> list[tuple[str, ...]]:
    """Sample n_trials K-author groups (enumerate when feasible, else direct sampling)."""
    n = len(unique_authors)
    if n < k:
        return []

    pool_size = math.comb(n, k)
    rng = random.Random(seed)

    if pool_size <= n_trials:
        subsets = list(itertools.combinations(unique_authors, k))
        groups = list(subsets)
        groups.extend(rng.choices(subsets, k=n_trials - len(subsets)))
        rng.shuffle(groups)
        return groups

    if pool_size <= _ENUM_THRESHOLD:
        subsets = list(itertools.combinations(unique_authors, k))
        return rng.sample(subsets, n_trials)

    seen: set[tuple[str, ...]] = set()
    groups: list[tuple[str, ...]] = []
    while len(groups) < n_trials:
        group = tuple(sorted(rng.sample(unique_authors, k)))
        if group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def ari_sweep(
    X: np.ndarray,
    authors: list[str],
    k_values: list[int],
    *,
    n_trials: int = 30,
    n_init: int = 10,
    docs_per_author: int,
    seed: int = 42,
) -> dict[int, list[float]]:
    """For each K, run n_trials author-group trials and return per-K ARI lists."""
    unique_authors = sorted(set(authors))
    author_to_rows: dict[str, list[int]] = {}
    for i, a in enumerate(authors):
        author_to_rows.setdefault(a, []).append(i)

    min_pool = min(len(author_to_rows[a]) for a in unique_authors)
    if docs_per_author > min_pool:
        raise ValueError(
            f"docs_per_author={docs_per_author} exceeds smallest author "
            f"pool ({min_pool}). Lower the budget or rebalance the corpus "
            f"so every author has ≥{docs_per_author} docs."
        )

    results: dict[int, list[float]] = {}
    for k in k_values:
        chosen = _sample_groups(
            unique_authors, k=k, n_trials=n_trials, seed=seed * 1000 + k
        )

        scores: list[float] = []
        for t, group in enumerate(chosen):
            trial_rng = random.Random(seed * 10_000_000 + k * 10_000 + t)
            rows = []
            for a in group:
                rows.extend(
                    trial_rng.sample(author_to_rows[a], docs_per_author)
                )
            rows.sort()

            sub_X = X[rows]
            a_to_id = {a: i for i, a in enumerate(group)}
            sub_ids = [a_to_id[authors[r]] for r in rows]
            labels = KMeans(
                n_clusters=k, init="k-means++",
                n_init=n_init,
                random_state=seed * 1_000_000 + k * 10_000 + t,
            ).fit_predict(sub_X)
            ari = adjusted_rand_score(sub_ids, labels)
            scores.append(ari)
            print(
                f"  K={k:2d}  trial {t+1:3d}/{len(chosen)}  ARI={ari:+.3f}  "
                f"n_docs={len(rows):3d}  authors={list(group)}"
            )
        results[k] = scores
    return results


def _print_summary(results: dict[int, list[float]], k_values: list[int]) -> None:
    print(f"  {'K':>3}  {'n':>3}  {'mean':>6}  {'std':>6}  {'min':>6}  {'max':>6}")
    for k in k_values:
        s = results[k]
        print(f"  {k:>3d}  {len(s):>3d}  "
              f"{np.mean(s):>+6.3f}  {np.std(s):>6.3f}  "
              f"{min(s):>+6.3f}  {max(s):>+6.3f}")


def main() -> None:
    print("Loading cleaned corpus …")
    corpus = load_corpus("", cleaned_dir="cleaned_35", dataset_root=_DATA_DIR)
    corpus = (
        corpus.filter(min_words=MIN_WORDS)
              .truncate_to(DOC_LENGTH)
    )
    print(f"  {len(corpus)} docs × {corpus.truncated_to}w  "
          f"({len(set(corpus.authors))} authors)")

    print("\nBuilding stylometric feature matrix …")
    X, _ = build_feature_matrix(
        corpus.texts, corpus.labels,
        source=corpus.source, cleaned_dir=corpus.cleaned_dir,
    )

    n_auth = len(set(corpus.authors))

    print(f"\nARI sweep: {M_PER_AUTHOR} docs/author at every K")
    print(f"  K in {K_VALUES}, {N_TRIALS} Trials per K")
    for k in K_VALUES:
        pool = math.comb(n_auth, k)
        print(f"    K={k:2d}: group pool={pool}, "
              f"{M_PER_AUTHOR} docs/author (total={M_PER_AUTHOR * k})")
    print(f"  KMeans: init='k-means++', n_init={N_INIT}")

    results = ari_sweep(
        X=X, authors=corpus.authors, k_values=K_VALUES,
        n_trials=N_TRIALS, n_init=N_INIT,
        docs_per_author=M_PER_AUTHOR,
        seed=GLOBAL_SEED,
    )

    print("\n── ARI sweep summary ──")
    _print_summary(results, K_VALUES)

    plot_ari_sweep(
        results, FIG_PATH,
        title=(
            f"K-means — Effects of No. of Authors (K)  "
            f"(Lesswrong 35 Authors, articles = {M_PER_AUTHOR}, length = {DOC_LENGTH})"
        ),
    )
    print(f"\nSaved -> {FIG_PATH.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
