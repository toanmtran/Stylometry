"""K=10 clustering on cleaned_10: main PCA/confusion diagnostic plus a seed-stability sweep."""

from __future__ import annotations

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
from src.kmeans.scripts.viz import plot_diagnostics


MIN_WORDS = 1500
N_PER_ERA = 20
K = 10
N_INIT = 10
SAMPLE_SEED = 42

OUT_DIR = _KMEANS_DIR / "outputs" / "cluster_10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUT_DIR / "clustering.png"
STABILITY_PATH = OUT_DIR / "ari_vs_seed.png"


def main() -> None:
    print("Loading cleaned corpus …")
    base_corpus = load_corpus("", cleaned_dir="cleaned_10", dataset_root=_DATA_DIR)
    print(f"  total: {len(base_corpus)} docs, "
          f"{len(set(base_corpus.authors))} authors")

    base_corpus = base_corpus.filter(min_words=MIN_WORDS)
    print(f"  after min_words={MIN_WORDS}: {len(base_corpus)} docs")

    corpus = base_corpus.balance_per_author_era(
        n_per_era=N_PER_ERA, seed=SAMPLE_SEED,
    )
    print(f"  after per-author era balancing ({N_PER_ERA}/era): "
          f"{len(corpus)} docs")

    corpus = corpus.truncate_to(MIN_WORDS)
    print(f"  truncated to N={corpus.truncated_to} words")

    unique_authors = sorted(set(corpus.authors))
    a2i = {a: i for i, a in enumerate(unique_authors)}
    true_ids = [a2i[a] for a in corpus.authors]

    print(f"\nBuilding stylometric feature matrix …")
    X, feature_names = build_feature_matrix(
        corpus.texts, corpus.labels,
        source=corpus.source, cleaned_dir=corpus.cleaned_dir,
    )
    print(f"  X shape: {X.shape}  ({len(feature_names)} features)")

    print(f"\nRunning K-means (k-means++ init, n_init={N_INIT}), K={K} …")
    pred = KMeans(
        n_clusters=K, init="k-means++",
        n_init=N_INIT, random_state=SAMPLE_SEED,
    ).fit_predict(X)
    ari = adjusted_rand_score(true_ids, pred)
    print(f"  ARI = {ari:.3f}")

    settings_lines = [
        f"Authors (K) = {len(unique_authors)}",
        f"Articles/author = {2 * N_PER_ERA}",
        f"Length = {corpus.truncated_to}",
        "Initialization = k-means++",
        f"Starts = {N_INIT}",
        "Choice = Lowest Cost",
    ]

    plot_diagnostics(
        X=X, pred=pred,
        true_ids=true_ids, author_names=unique_authors,
        ari=ari, out_path=FIG_PATH,
        title_prefix="K-means Clustering — LessWrong 10 Authors",
        settings_lines=settings_lines,
    )

    print(f"\nSaved -> {FIG_PATH.relative_to(_PROJECT_ROOT)}")

    print(
        f"\nRunning stability analysis: 10 K-means runs "
        f"(n_init={N_INIT} attempts each) on the same {len(corpus)} docs, "
        f"varying random_state…", flush=True,
    )
    test_seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    plt.figure(figsize=(10, 6))
    best_aris = []

    for i, test_seed in enumerate(test_seeds):
        shared_rs = np.random.RandomState(test_seed)
        distortions = []
        aris = []
        for _ in range(N_INIT):
            km = KMeans(
                n_clusters=K, init="k-means++",
                n_init=1, random_state=shared_rs,
            )
            test_pred = km.fit_predict(X)
            distortions.append(km.inertia_)
            aris.append(adjusted_rand_score(true_ids, test_pred))

        best_idx = int(np.argmin(distortions))
        best_aris.append(aris[best_idx])

        print(
            f"  random_state={test_seed:>3d}: best ARI={aris[best_idx]:+.3f}  "
            f"(attempt {best_idx + 1}/{N_INIT}, "
            f"distortion={distortions[best_idx]:.1f})",
            flush=True,
        )

        x_vals = [i] * N_INIT
        plt.scatter(
            x_vals, distortions, color="blue", alpha=0.4, s=20,
            label="Attempts" if i == 0 else "",
        )
        plt.scatter(
            [i], [distortions[best_idx]], color="red", marker="*", s=100,
            label="Lowest Distortion" if i == 0 else "",
        )
        plt.annotate(
            f"{aris[best_idx]:.2f}", (i, distortions[best_idx]),
            textcoords="offset points", xytext=(0, -15),
            ha="center", fontsize=9, color="red",
        )

    mean_ari = float(np.mean(best_aris))
    plt.title(
        f"K-means Stability — {len(corpus)} docs, 10 trials "
        f"(Mean ARI: {mean_ari:.3f})"
    )
    plt.xlabel("KMeans random_state")
    plt.ylabel("Distortion")
    plt.xticks(range(len(test_seeds)), test_seeds)
    plt.legend()
    plt.tight_layout()
    plt.savefig(STABILITY_PATH)
    print(f"\nSaved -> {STABILITY_PATH.relative_to(_PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
