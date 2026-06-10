"""Stylometric feature matrix builder and Hungarian cluster-to-label mapper."""

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
    """Short content hash that identifies a (texts, labels) input."""
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
    """Build the z-scored stylometric feature matrix, with optional disk caching."""
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
