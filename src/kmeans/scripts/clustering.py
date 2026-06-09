"""
Clustering building blocks.

`build_feature_matrix` : handcrafted stylometric feature matrix — word-length
stats, sentence/paragraph length, vocabulary richness (Yule K, Simpson D,
Brunét W, Honoré R, hapax ratios, type-token ratio), punctuation rates, POS
tag distribution, function-word frequencies, and category rates (hedges,
amplifiers, discourse markers, conjunctions). Standardised with z-score
(StandardScaler) so every feature has mean 0 and std 1 before clustering.

`f_ratio` : per-feature ANOVA F-statistic (between-cluster variance ÷ within-
cluster variance) — used to pick the features that discriminate clusters best.

`hungarian_map` : one-to-one assignment from predicted cluster ids to true-author
ids that maximises the diagonal of the confusion matrix.

K-means itself is called directly by the problem scripts via sklearn's KMeans
(init="k-means++", n_init=10).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from src.kmeans.scripts.features import extract_features_batch
from src.kmeans.scripts.preprocess import canonicalize_typography


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"


def _cache_key(texts: list[str], labels: list[str]) -> str:
    """16-hex-char SHA-256 prefix that uniquely identifies this (texts, labels)
    invocation. Solutions truncate/filter the corpus before calling
    build_feature_matrix, so the same dataset legitimately produces multiple
    matrices — each gets its own cache entry under the dataset's folder."""
    h = hashlib.sha256()
    for t in texts:
        b = t.encode("utf-8", errors="replace")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    h.update(b"\x00--labels--\x00")
    for label in labels:
        b = label.encode("utf-8", errors="replace")
        h.update(len(b).to_bytes(4, "little"))
        h.update(b)
    return h.hexdigest()[:16]


def build_feature_matrix(
    texts: list[str],
    labels: list[str],
    *,
    source: str | None = None,
    cleaned_dir: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the stylometric feature matrix.

    Each column is one handcrafted stylometric feature; we drop any feature
    with zero variance on this corpus (uninformative) and z-score the rest.

    When `source` and `cleaned_dir` are both provided, the result is cached
    to `cache/features/<source>/<cleaned_dir>/<key>.npz` where `key` is a
    short content hash of (texts, labels). On a hit, the matrix is loaded
    instead of recomputed. Edits to feature extraction code do not auto-
    invalidate the cache — delete `cache/features/` after such changes.

    Returns:
      X              : (n_docs, n_features) dense float matrix
      feature_names  : length-n_features list of column names
    """
    use_cache = source is not None and cleaned_dir is not None
    cache_path: Path | None = None
    if use_cache:
        key = _cache_key(texts, labels)
        cache_path = _CACHE_ROOT / source / cleaned_dir / f"{key}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            X = data["X"].astype(np.float32, copy=False)
            feature_names = data["feature_names"].tolist()
            print(f"  feature cache hit -> {cache_path.relative_to(_PROJECT_ROOT)}")
            return X, feature_names

    # Canonicalise quotes/apostrophes once so that the tokenizer and the
    # contraction regex both see consistent ASCII quotes regardless of the
    # publishing pipeline's smart-quote setting.
    texts = [canonicalize_typography(t) for t in texts]

    style_df = extract_features_batch(texts, labels)
    keep = style_df.var()[style_df.var() > 1e-10].index.tolist()

    if keep:
        X = StandardScaler().fit_transform(style_df[keep].values)
    else:
        X = np.zeros((len(texts), 0), dtype=np.float32)
    feature_names = list(keep)
    X = X.astype(np.float32, copy=False)

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, feature_names=np.array(feature_names))
        print(f"  feature cache miss -> built and saved "
              f"{cache_path.relative_to(_PROJECT_ROOT)}")

    return X, feature_names


def f_ratio(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-feature ANOVA F-statistic for a clustering.

    For each column j of X:
        F_j = between-cluster variance / within-cluster variance
            = (SS_between / (K-1)) / (SS_within / (N-K))

    Higher F means the cluster means of that feature differ more than its
    spread within clusters — i.e. the feature discriminates the clusters.
    Features with zero within-cluster variance get F = 0 (uninformative).
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n, d = X.shape
    clusters = np.unique(labels)
    k = len(clusters)
    overall_mean = X.mean(axis=0)

    ss_between = np.zeros(d)
    ss_within = np.zeros(d)
    for c in clusters:
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        mu_c = X[mask].mean(axis=0)
        ss_between += n_c * (mu_c - overall_mean) ** 2
        ss_within += ((X[mask] - mu_c) ** 2).sum(axis=0)

    df_between = max(k - 1, 1)
    df_within = max(n - k, 1)
    between_var = ss_between / df_between
    within_var = ss_within / df_within
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(within_var > 0, between_var / within_var, 0.0)
    return f


def hungarian_map(true_ids: list[int], pred_ids: np.ndarray) -> np.ndarray:
    """Map predicted cluster ids to true-author ids (maximising diagonal)."""
    true_classes = sorted(set(true_ids))
    pred_classes = sorted(set(pred_ids.tolist()))
    size = max(len(true_classes), len(pred_classes))
    cost = np.zeros((size, size))
    for i, tc in enumerate(true_classes):
        for j, pc in enumerate(pred_classes):
            cost[i, j] = -sum(
                1 for t, p in zip(true_ids, pred_ids) if t == tc and p == pc
            )
    row, col = linear_sum_assignment(cost)
    mapping = {
        pred_classes[c]: true_classes[r]
        for r, c in zip(row, col)
        if r < len(true_classes) and c < len(pred_classes)
    }
    return np.array([mapping.get(int(p), int(p)) for p in pred_ids])
