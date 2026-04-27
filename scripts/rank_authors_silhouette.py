"""
Rank cleaned_25 authors by how well they cluster relative to their peers.

Uses the per-author mean silhouette score in the globally z-scored feature
space (the same space K-means operates in). Authors at the top of the
ranking are the most "blurry" — their articles are stylometrically closer
to some other author than to themselves — and are the best candidates for
removal to improve ARI.

For each author the table shows:
  mean_sil   mean silhouette across all the author's articles
             (negative = more articles misplaced than placed correctly)
  neg_pct    % of articles with silhouette < 0
  a          mean within-author distance (cohesion; lower = tighter cluster)
  b          mean nearest-cluster distance (separation; higher = safer)
  b/a        separation-to-cohesion ratio (higher = better)
  confused_with   the other author whose centroid is geometrically closest
                  to this author's centroid (their most likely K-means mix-up)

Output:
  stdout          ranked table
  outputs/corpus_stats/lesswrong_large/cleaned_25/author_silhouette_rank.csv

Usage:
  python scripts/rank_authors_silhouette.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import silhouette_samples

from src.corpus import load_corpus
from src.clustering import build_feature_matrix

MIN_WORDS  = 1526
DOC_LENGTH = 1526

OUT_DIR = _PROJECT_ROOT / "outputs" / "corpus_stats" / "lesswrong_large" / "cleaned_25"
OUT_CSV = OUT_DIR / "author_silhouette_rank.csv"


def main() -> None:
    print("Loading corpus …")
    corpus = (
        load_corpus("lesswrong_large", cleaned_dir="cleaned_25")
        .filter(min_words=MIN_WORDS)
        .equal_length_truncate(floor=MIN_WORDS)
    )
    print(f"  {len(corpus)} docs  {len(set(corpus.authors))} authors  "
          f"{corpus.truncated_to}w each")

    print("\nBuilding feature matrix …")
    X, _ = build_feature_matrix(
        corpus.texts, corpus.labels,
        source=corpus.source, cleaned_dir=corpus.cleaned_dir,
    )

    authors     = corpus.authors                       # list[str], length N
    unique      = sorted(set(authors))
    a2i         = {a: i for i, a in enumerate(unique)}
    label_ids   = np.array([a2i[a] for a in authors])

    print("\nComputing silhouette scores …")
    sil = silhouette_samples(X, label_ids)             # (N,)

    # Per-author centroids for confusion-partner lookup
    centroids = np.stack([
        X[label_ids == i].mean(axis=0) for i in range(len(unique))
    ])

    rows: list[dict] = []
    authors_arr = np.array(authors)
    for author in unique:
        aid    = a2i[author]
        mask   = label_ids == aid
        a_sil  = sil[mask]
        a_X    = X[mask]
        n      = int(mask.sum())

        mean_sil = float(a_sil.mean())
        neg_pct  = float((a_sil < 0).mean()) * 100

        # a_i: mean within-author distance (cohesion)
        diffs_in  = a_X - a_X.mean(axis=0)
        a_score   = float(np.linalg.norm(diffs_in, axis=1).mean())

        # b_i: mean distance to nearest other-author centroid (separation)
        other_ids  = [i for i in range(len(unique)) if i != aid]
        cent_dists = np.linalg.norm(centroids[other_ids] - centroids[aid], axis=1)
        nearest_id = other_ids[int(np.argmin(cent_dists))]
        b_score    = float(cent_dists.min())
        confused   = unique[nearest_id]

        rows.append({
            "author":        author,
            "n":             n,
            "mean_sil":      mean_sil,
            "neg_pct":       neg_pct,
            "a":             a_score,
            "b":             b_score,
            "b_over_a":      b_score / a_score if a_score > 0 else float("inf"),
            "confused_with": confused,
        })

    rows.sort(key=lambda r: r["mean_sil"])

    # ── Print table ──────────────────────────────────────────────────────────
    hdr = (f"{'Rank':>4}  {'Author':<28}  {'N':>3}  "
           f"{'mean_sil':>8}  {'neg%':>5}  {'a':>6}  {'b':>6}  "
           f"{'b/a':>5}  confused_with")
    print(f"\n{hdr}")
    print("─" * len(hdr))
    for rank, r in enumerate(rows, 1):
        marker = " ←" if rank <= 5 else ""
        print(
            f"{rank:>4}  {r['author']:<28}  {r['n']:>3}  "
            f"{r['mean_sil']:>+8.4f}  {r['neg_pct']:>4.0f}%  "
            f"{r['a']:>6.3f}  {r['b']:>6.3f}  "
            f"{r['b_over_a']:>5.2f}  {r['confused_with']}"
            f"{marker}"
        )

    # ── Save CSV ─────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write("rank,author,n,mean_sil,neg_pct,a,b,b_over_a,confused_with\n")
        for rank, r in enumerate(rows, 1):
            f.write(
                f"{rank},{r['author']},{r['n']},"
                f"{r['mean_sil']:.6f},{r['neg_pct']:.1f},"
                f"{r['a']:.6f},{r['b']:.6f},{r['b_over_a']:.4f},"
                f"{r['confused_with']}\n"
            )
    print(f"\nSaved → {OUT_CSV.relative_to(_PROJECT_ROOT)}")

    # ── Guidance ─────────────────────────────────────────────────────────────
    print("\nTo remove an author from cleaned_25, delete their JSON file:")
    for rank, r in enumerate(rows[:5], 1):
        print(f"  [{rank}] dataset/lesswrong_large/cleaned_25/{r['author']}.json  "
              f"(mean_sil={r['mean_sil']:+.4f})")


if __name__ == "__main__":
    main()
