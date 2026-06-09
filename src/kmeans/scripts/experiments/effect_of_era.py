"""
Effect of era on clustering quality (ARI vs publication period).

Tests whether stylometric clustering generalises across publication eras
on the cleaned_10 corpus, where every author has exactly 40 articles
balanced 20|20 between pre (year <= 2022) and post (year >= 2024).

Fixed parameters across both eras:
  K = 4   (authors per trial, sampled from the 10 in cleaned_10)
  M = 10  (docs per author, sampled from that author's 20 in this era)
  N = 1500  (words per doc)

For each era we run N_TRIALS = 50 independent trials. Each trial:
  1. Randomly samples K authors from the 10 available.
  2. Randomly samples M of that author's 20 era-restricted docs.
  3. Runs K-means++ (n_init=10) and records ARI vs true author labels.

Per-era RNG keying mirrors the other sweep scripts: each era gets its
own seed branch so adding/removing an era leaves the other untouched.

Output: src/kmeans/outputs/effect_of_era/ari_vs_era.png
"""

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
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.kmeans.scripts.clustering import build_feature_matrix
from src.kmeans.scripts.corpus import load_corpus


DOC_LENGTH = 1500        # fixed N for both eras
K = 4
M = 10
N_TRIALS = 50
N_INIT = 10
GLOBAL_SEED = 42

ERAS = ["pre", "post"]   # x-axis order, left -> right
ERA_LABELS = {"pre": "pre  (year ≤ 2022)", "post": "post  (year ≥ 2024)"}

OUT_DIR = _KMEANS_DIR / "outputs" / "effect_of_era"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUT_DIR / "ari_vs_era.png"


def _sample_groups(
    unique_authors: list[str],
    *,
    k: int,
    n_trials: int,
    seed: int,
) -> list[tuple[str, ...]]:
    """Sample K-author groups for one era.

    Without replacement when C(n_authors, K) >= n_trials; otherwise uses
    every unique group once and fills the remainder with replacement.
    Mirrors the helper used by effect_of_m / effect_of_k.
    """
    subsets = list(itertools.combinations(unique_authors, k))
    rng = random.Random(seed)
    if len(subsets) >= n_trials:
        return rng.sample(subsets, n_trials)
    groups = list(subsets)
    groups.extend(rng.choices(subsets, k=n_trials - len(subsets)))
    rng.shuffle(groups)
    return groups


def run_era(corpus, era: str) -> list[float]:
    """Filter to `era`, build the era-restricted feature matrix once,
    then run N_TRIALS sampled (K, M) trials and return the ARI list."""
    sub = corpus.filter(era=era).truncate_to(DOC_LENGTH)
    n_authors = len(set(sub.authors))
    print(f"  era={era}: {len(sub)} docs × {sub.truncated_to}w  "
          f"({n_authors} authors)")

    author_to_rows: dict[str, list[int]] = {}
    for i, a in enumerate(sub.authors):
        author_to_rows.setdefault(a, []).append(i)

    min_pool = min(len(author_to_rows[a]) for a in author_to_rows)
    if M > min_pool:
        raise ValueError(
            f"M={M} exceeds smallest era-restricted author pool "
            f"({min_pool}) for era={era!r}."
        )

    print("  Building stylometric feature matrix …")
    X, _ = build_feature_matrix(
        sub.texts, sub.labels,
        source=sub.source, cleaned_dir=sub.cleaned_dir,
    )
    print(f"  X shape: {X.shape}")

    unique_authors = sorted(author_to_rows)
    era_seed_offset = 1 if era == "pre" else 2
    chosen = _sample_groups(
        unique_authors, k=K, n_trials=N_TRIALS,
        seed=GLOBAL_SEED * 1000 + era_seed_offset,
    )

    scores: list[float] = []
    for t, group in enumerate(chosen):
        trial_rng = random.Random(
            GLOBAL_SEED * 10_000_000 + era_seed_offset * 100_000 + t
        )
        rows: list[int] = []
        for a in group:
            rows.extend(trial_rng.sample(author_to_rows[a], M))
        rows.sort()

        sub_X = X[rows]
        a_to_id = {a: i for i, a in enumerate(group)}
        sub_ids = [a_to_id[sub.authors[r]] for r in rows]
        labels = KMeans(
            n_clusters=K, init="k-means++",
            n_init=N_INIT,
            random_state=GLOBAL_SEED * 1_000_000 + era_seed_offset * 10_000 + t,
        ).fit_predict(sub_X)
        ari = adjusted_rand_score(sub_ids, labels)
        scores.append(ari)
        print(
            f"  [{era:>4s}] trial {t+1:3d}/{N_TRIALS}  ARI={ari:+.3f}  "
            f"authors={list(group)}"
        )
    return scores


def _print_summary(results: dict[str, list[float]]) -> None:
    print(f"  {'era':>5}  {'n':>3}  {'mean':>6}  {'std':>6}  {'min':>6}  {'max':>6}")
    for era in ERAS:
        s = results[era]
        print(f"  {era:>5s}  {len(s):>3d}  "
              f"{np.mean(s):>+6.3f}  {np.std(s):>6.3f}  "
              f"{min(s):>+6.3f}  {max(s):>+6.3f}")


def plot_ari_vs_era(results: dict[str, list[float]], out_path: Path) -> None:
    """Two-position categorical version of the ARI sweep plot — same look
    as plot_ari_sweep but with era labels on the x-axis."""
    xs = list(range(len(ERAS)))
    means = [float(np.mean(results[e])) for e in ERAS]
    stds = [float(np.std(results[e])) for e in ERAS]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    jitter_rng = np.random.RandomState(0)
    for x, era in zip(xs, ERAS):
        ys = results[era]
        xj = x + jitter_rng.uniform(-0.10, 0.10, size=len(ys))
        ax.scatter(xj, ys, color="#C44E52", alpha=0.35, s=22, zorder=2,
                   label="trial" if era == ERAS[0] else None)
    ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=5, linewidth=2,
                color="#4C72B0", markeredgecolor="black", markersize=8,
                label="mean ± std", zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([ERA_LABELS[e] for e in ERAS])
    ax.set_xlim(-0.5, len(ERAS) - 0.5)
    ax.set_xlabel("era")
    ax.set_ylabel("ARI")
    trials_str = "  ".join(f"{e}: {len(results[e])}" for e in ERAS)
    ax.set_title(
        f"K-means — Effects of Era  "
        f"(Lesswrong 10 Authors, authors={K}, articles={M}, length={DOC_LENGTH})\n"
        f"Trials per era — {trials_str}"
    )
    ax.axhline(0, linestyle=":", color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    for x, m in zip(xs, means):
        ax.annotate(f"{m:.2f}", (x, m), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color="#4C72B0")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Loading cleaned_10 corpus …")
    corpus = load_corpus("", cleaned_dir="cleaned_10", dataset_root=_DATA_DIR)
    n_authors = len(set(corpus.authors))
    print(f"  {len(corpus)} docs total, {n_authors} authors  "
          f"(20 pre + 20 post per author)")

    if min(corpus.word_counts) < DOC_LENGTH:
        print(f"ERROR: shortest doc = {min(corpus.word_counts)}w, "
              f"need ≥ {DOC_LENGTH}")
        sys.exit(1)

    group_pool = math.comb(n_authors, K)
    print(f"\nARI sweep: K={K}, M={M}, N={DOC_LENGTH}, "
          f"{N_TRIALS} trials per era")
    print(f"  group pool = C({n_authors},{K}) = {group_pool}  per era")
    print(f"  KMeans: init='k-means++', n_init={N_INIT}\n")

    results: dict[str, list[float]] = {}
    for era in ERAS:
        print(f"=== era = {era} ===")
        results[era] = run_era(corpus, era)
        print()

    print("── ARI vs era summary ──")
    _print_summary(results)

    plot_ari_vs_era(results, FIG_PATH)
    print(f"\nSaved -> {FIG_PATH.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
