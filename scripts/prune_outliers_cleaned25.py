"""
Prune off-group articles from dataset/lesswrong_large/cleaned_25/.

For each author, computes PCA-whitened Mahalanobis distances from the author
centroid, then looks for a natural break in the sorted distance distribution:
a spacing between consecutive distances that is abnormally large compared to
the typical inter-article spacing. Everything beyond that break is removed.

Algorithm (per author):
  1. Extract 107 features from each article truncated to DOC_LENGTH=1526
     words (same window as effect_of_k.py) — removes length as a confound.

  2. Z-score each feature within the author group; skip zero-variance ones.

  3. PCA-whiten: reduce to k components capturing >=VAR_THRESHOLD of within-
     author variance (capped at n//3 for well-conditioned covariance), then
     scale each component by 1/sqrt(eigenvalue). The L2 norm of the whitened
     vector is the Mahalanobis distance. Wide-ranging authors have large
     eigenvalues -> larger effective radius; consistent authors get a tighter
     one. The style region is entirely self-referential per author.

  4. Sort the n Mahalanobis distances. Compute the n-1 consecutive spacings.
     Apply the Iglewicz-Hoaglin (1993) Modified Z-score to the spacings:

       MZ_i = 0.6745 * (spacing_i - median(spacings)) / MAD(spacings)

     The key distinction from the previous attempt: MZ is applied to the
     SPACINGS not to the distances. Mahalanobis distances are chi-distributed
     with thin tails, so even extreme distances score low on MZ. Spacings
     are roughly exponential -- a single large gap (an outlier breaking away)
     produces a huge MZ and is reliably detected.

  5. Search for the largest-MZ spacing only within the upper MAX_REMOVE_FRAC
     of the sorted list (default 25%). This keeps the search in the tail so
     the main cluster is never split. If the best spacing MZ > GAP_SIGMA
     (default 2.0), all articles beyond it are flagged as off-group.

Features are extracted inline — not written to cache/.

Usage:
  python scripts/prune_outliers_cleaned25.py
  python scripts/prune_outliers_cleaned25.py --gap-sigma 2.5   # more conservative
  python scripts/prune_outliers_cleaned25.py --gap-sigma 1.5   # more aggressive
  python scripts/prune_outliers_cleaned25.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import extract_features
from src.preprocess import canonicalize_typography, truncate_to_words

_DATASET_DIR     = _PROJECT_ROOT / "dataset" / "lesswrong_large" / "cleaned_25"
_DOC_LENGTH      = 1526
_VAR_THRESHOLD   = 0.90   # fraction of within-author variance PCA must retain
_GAP_SIGMA       = 1.8    # Modified Z-score threshold on spacings
_MAX_REMOVE_FRAC = 0.25   # only look for gaps in the top 25% by distance


# ── Feature extraction ────────────────────────────────────────────────────────

def _feature_matrix(articles: list[dict], doc_length: int) -> np.ndarray:
    """Extract raw feature vectors for every article; show inline progress."""
    n = len(articles)
    rows: list[dict] = []
    for i, art in enumerate(articles):
        text = canonicalize_typography(art["text"])
        text = truncate_to_words(text, doc_length)
        rows.append(extract_features(text))
        print(f"    [{i + 1:3d}/{n}]", end="\r", flush=True)
    print()
    keys = list(rows[0].keys())
    return np.array([[r.get(k, 0.0) for k in keys] for r in rows], dtype=np.float64)


# ── Gap detection on Mahalanobis distances ────────────────────────────────────

def _gap_outliers(
    matrix: np.ndarray,
    var_threshold: float,
    gap_sigma: float,
    max_remove_frac: float,
) -> tuple[list[int], dict]:
    """
    Return (outlier_indices, info_dict).

    info_dict contains diagnostic fields: pca_k, distances, best_gap_mz,
    best_gap_pos, fence_distance. Empty outlier list if no gap found.
    """
    n, _ = matrix.shape
    info: dict = {"pca_k": None, "best_gap_mz": None, "fence_dist": None}

    if n < 6:
        return [], info

    # Within-author z-score
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0)
    valid = sigma > 1e-10
    if valid.sum() < 5:
        return [], info
    z_mat = (matrix[:, valid] - mu[valid]) / sigma[valid]

    # Adaptive PCA whitening
    max_k = max(3, min(n // 3, int(valid.sum())))
    pca_probe = PCA(n_components=max_k, random_state=42)
    pca_probe.fit(z_mat)
    cum_var = np.cumsum(pca_probe.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, var_threshold)) + 1
    k = max(3, min(k, max_k))
    info["pca_k"] = k

    pca_w = PCA(n_components=k, whiten=True, random_state=42)
    scores_w = pca_w.fit_transform(z_mat)
    dists = np.sqrt((scores_w ** 2).sum(axis=1))
    info["dists"] = dists

    # Sort by distance
    order = np.argsort(dists)          # ascending
    sorted_d = dists[order]
    spacings = np.diff(sorted_d)       # n-1 non-negative values

    # Modified Z-score on spacings — robust to the very gaps being detected
    med_s = np.median(spacings)
    mad_s = np.median(np.abs(spacings - med_s))
    if mad_s < 1e-10:
        return [], info
    mz_s = 0.6745 * (spacings - med_s) / mad_s

    # Search in the upper MAX_REMOVE_FRAC of the sorted list
    floor = max(3, int(n * (1.0 - max_remove_frac)))
    upper_mz = mz_s[floor:]           # spacings between positions floor..n-1
    if len(upper_mz) == 0:
        return [], info

    best_local = int(np.argmax(upper_mz))
    best_global = floor + best_local   # index in spacings / sorted_d
    best_mz = float(upper_mz[best_local])
    info["best_gap_mz"] = best_mz
    info["fence_dist"] = float(sorted_d[best_global + 1])

    if best_mz <= gap_sigma:
        return [], info

    # Everything after the gap
    outlier_idxs = sorted(int(j) for j in order[best_global + 1:])
    return outlier_idxs, info


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gap-sigma", type=float, default=_GAP_SIGMA,
        help=f"Modified Z-score threshold on spacings (default: {_GAP_SIGMA}; "
             "raise to keep more, lower to remove more)",
    )
    parser.add_argument(
        "--max-remove-frac", type=float, default=_MAX_REMOVE_FRAC,
        help=f"max fraction of articles to consider for removal per author "
             f"(default: {_MAX_REMOVE_FRAC})",
    )
    parser.add_argument(
        "--var-threshold", type=float, default=_VAR_THRESHOLD,
        help=f"PCA variance to retain per author (default: {_VAR_THRESHOLD})",
    )
    parser.add_argument(
        "--doc-length", type=int, default=_DOC_LENGTH,
        help=f"truncation length in words (default: {_DOC_LENGTH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be removed without modifying any files",
    )
    args = parser.parse_args()

    author_files = sorted(_DATASET_DIR.glob("*.json"))
    if not author_files:
        sys.exit(f"No JSON files found in {_DATASET_DIR}")

    print(f"Pruning off-group articles from "
          f"{_DATASET_DIR.relative_to(_PROJECT_ROOT)}")
    print(f"  gap_sigma={args.gap_sigma}  max_remove_frac={args.max_remove_frac}  "
          f"doc_length={args.doc_length}  dry_run={args.dry_run}")
    print(f"  {len(author_files)} authors\n")

    total_before  = 0
    total_removed = 0
    report_entries: list[dict] = []

    for fp in author_files:
        author   = fp.stem
        articles = json.loads(fp.read_text(encoding="utf-8"))
        n        = len(articles)
        total_before += n

        print(f"{author}  ({n} articles)")
        matrix = _feature_matrix(articles, args.doc_length)

        outlier_idxs, info = _gap_outliers(
            matrix,
            var_threshold=args.var_threshold,
            gap_sigma=args.gap_sigma,
            max_remove_frac=args.max_remove_frac,
        )

        k       = info.get("pca_k")
        best_mz = info.get("best_gap_mz")
        dists   = info.get("dists")

        if k is not None:
            mz_str = f"{best_mz:.2f}" if best_mz is not None else "n/a"
            print(f"  PCA k={k}  best_gap_MZ={mz_str}  "
                  f"threshold={args.gap_sigma}")

        if not outlier_idxs:
            print("  → no gap found above threshold\n")
            report_entries.append({
                "author": author, "n_before": n, "n_kept": n,
                "n_removed": 0, "pca_k": k, "removed": [],
            })
            continue

        fence = info.get("fence_dist", float("nan"))
        removed_info: list[dict] = []
        for i in outlier_idxs:
            art   = articles[i]
            d_str = f"{dists[i]:.3f}" if dists is not None else "n/a"
            removed_info.append({
                "index":        i,
                "source_file":  art.get("source_file", ""),
                "source_index": art.get("source_index", i),
                "date":         art.get("date", ""),
                "word_count":   art.get("word_count", 0),
                "mahal_dist":   round(float(dists[i]), 4) if dists is not None else None,
            })
            print(
                f"  REMOVE [{i:3d}]  d={d_str}  (fence={fence:.3f})  "
                f"wc={art.get('word_count', 0):5d}  "
                f"{art.get('date', '?')}  "
                f"{art.get('source_file', '')}:{art.get('source_index', i)}"
            )

        outlier_set = set(outlier_idxs)
        kept = [art for i, art in enumerate(articles) if i not in outlier_set]
        print(f"  → kept {len(kept)}/{n}  (removed {len(outlier_idxs)})\n")

        if not args.dry_run:
            fp.write_text(
                json.dumps(kept, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        total_removed += len(outlier_idxs)
        report_entries.append({
            "author":       author,
            "n_before":     n,
            "n_kept":       len(kept),
            "n_removed":    len(outlier_idxs),
            "pca_k":        k,
            "best_gap_mz":  round(best_mz, 3) if best_mz is not None else None,
            "removed":      removed_info,
        })

    print("─" * 60)
    prefix = "DRY RUN — " if args.dry_run else ""
    print(
        f"{prefix}{total_removed} articles removed, "
        f"{total_before - total_removed} kept "
        f"(of {total_before} total)"
    )

    if not args.dry_run:
        out_path = (
            _PROJECT_ROOT
            / "outputs" / "kmeans" / "effect_of_k"
            / "outlier_pruning_report.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"settings": vars(args), "authors": report_entries},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Report saved → {out_path.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
