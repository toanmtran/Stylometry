"""Shared plotting helpers for clustering problems."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src.clustering import f_ratio, hungarian_map


def _cluster_majority(pred: np.ndarray, true_ids: list[int]) -> dict[int, int]:
    """For each predicted cluster id, return the true-author id that is the
    majority inside that cluster (ties broken by smallest author id)."""
    out: dict[int, int] = {}
    for c in set(int(p) for p in pred):
        members = [true_ids[i] for i in range(len(pred)) if pred[i] == c]
        out[c] = Counter(members).most_common(1)[0][0]
    return out


def _pretty_feature_name(name: str) -> str:
    if name.startswith("fw_"):
        return f"FW:{name[3:]}"
    if name.startswith("punct_"):
        return f"P:{name[6:]}"
    if name.startswith("cat_"):
        return f"CAT:{name[4:]}"
    if name.startswith("pos_"):
        return f"POS:{name[4:]}"
    return name


def plot_diagnostics(
    X: np.ndarray,
    pred: np.ndarray,
    true_ids: list[int],
    author_names: list[str],
    feature_names: list[str],
    ari: float,
    out_path: Path,
    *,
    title_prefix: str = "Stylometric Clustering",
    subtitle: str | None = None,
    top_k_features: int = 20,
    settings_lines: list[str] | None = None,
) -> None:
    """4-panel diagnostic: PCA (true + predicted), settings/top features,
    and Hungarian-aligned confusion matrix."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    head = f"{title_prefix}\nARI = {ari:.3f}"
    if subtitle:
        head = f"{title_prefix} — {subtitle}\nARI = {ari:.3f}"
    fig.suptitle(head, fontsize=25, fontweight="bold")

    pca = PCA(n_components=2).fit(X)
    X2 = pca.transform(X)
    vr = pca.explained_variance_ratio_
    colors = cm.tab20(np.linspace(0, 1, 20))

    # 1. PCA coloured by true author
    ax = axes[0, 0]
    for i, name in enumerate(author_names):
        mask = np.array(true_ids) == i
        ax.scatter(X2[mask, 0], X2[mask, 1], s=28, alpha=0.75,
                   color=colors[i % 20], label=name,
                   edgecolors="white", linewidths=0.3)
    ax.set_title("PCA — true authors", fontsize=21)
    ax.set_xlabel(f"PC1 ({vr[0]:.0%} var)", fontsize=18)
    ax.set_ylabel(f"PC2 ({vr[1]:.0%} var)", fontsize=18)
    ax.legend(fontsize=14, loc="best", ncol=2, frameon=True)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.25)

    # 2. PCA coloured by predicted cluster (each cluster -> majority-author colour).
    # Multiple clusters can share a colour when an author's docs split across
    # clusters; some author colours may not appear at all if no cluster has
    # that author as its majority. This reflects where the clusters actually
    # went rather than forcing a 1-to-1 mapping.
    cluster_major = _cluster_majority(pred, true_ids)
    doc_color_id = np.array([cluster_major[int(p)] for p in pred])
    ax = axes[0, 1]
    for i in range(len(author_names)):
        mask = doc_color_id == i
        if not mask.any():
            continue
        ax.scatter(X2[mask, 0], X2[mask, 1], s=28, alpha=0.75,
                   color=colors[i % 20],
                   edgecolors="white", linewidths=0.3)
    ax.set_title("PCA — predicted clusters (majority-author colours)", fontsize=21)
    ax.set_xlabel(f"PC1 ({vr[0]:.0%} var)", fontsize=18)
    ax.set_ylabel(f"PC2 ({vr[1]:.0%} var)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.25)

    # 3. Settings panel OR top-N discriminating features.
    ax = axes[1, 0]
    if settings_lines:
        ax.axis("off")
        # Calculate dimensions based on number of lines
        n_lines = len(settings_lines)
        box_height = min(0.9, 0.13 * n_lines + 0.05)
        box_y = 0.5 - box_height / 2

        panel = FancyBboxPatch(
            (0.0, box_y), 1.0, box_height,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor="#F5F5F5", edgecolor="#999999", linewidth=1.6,
            transform=ax.transAxes,
        )
        ax.add_patch(panel)

        # Center text vertically based on number of lines
        dy = min(0.12, (box_height - 0.1) / max(n_lines - 1, 1)) if n_lines > 1 else 0
        y = 0.5 + (dy * (n_lines - 1) / 2)

        for line in settings_lines:
            if "=" in line:
                key, val = [s.strip() for s in line.split("=", 1)]
                ax.text(
                    0.48, y, key,
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=26, fontweight="bold",
                )
                ax.text(
                    0.50, y, "=",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=26, fontweight="bold",
                )
                ax.text(
                    0.52, y, val,
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=26,
                )
            else:
                ax.text(
                    0.50, y, line,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=26, fontweight="bold",
                )
            y -= dy
    else:
        f = f_ratio(X, pred)
        top = np.argsort(f)[::-1][:top_k_features]
        labels_top = [_pretty_feature_name(feature_names[i]) for i in top]
        y_pos = np.arange(len(top))
        ax.barh(y_pos, f[top], color="#4C72B0", alpha=0.85,
                edgecolor="black", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_top, fontsize=13)
        ax.invert_yaxis()  # largest F at the top
        ax.set_title(f"Top {top_k_features} discriminating features "
                     , fontsize=18)
        ax.set_xlabel("F-ratio (between / within cluster variance)", fontsize=16)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(True, axis="x", alpha=0.25)

    # 4. Confusion matrix (Hungarian 1-to-1 mapping; kept for completeness).
    mapped = hungarian_map(true_ids, pred)
    ax = axes[1, 1]
    cm_ = confusion_matrix(true_ids, mapped, labels=list(range(len(author_names))))
    sns.heatmap(cm_, ax=ax, annot=True, fmt="d", cmap="Blues",
                xticklabels=author_names, yticklabels=author_names,
                cbar=False, square=True, linewidths=0.3,
                annot_kws={"size": 14})
    ax.set_title("Confusion matrix (Hungarian-aligned)", fontsize=21)
    ax.set_xlabel("predicted", fontsize=17)
    ax.set_ylabel("true", fontsize=17)
    ax.tick_params(axis="x", labelrotation=45, labelsize=15)
    ax.tick_params(axis="y", labelrotation=0, labelsize=15)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_features(
    X: np.ndarray,
    pred: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    *,
    top_k_features: int = 20,
    title: str = "Top Discriminating Features",
) -> None:
    """Standalone Top-N discriminating feature chart."""
    f = f_ratio(X, pred)
    top = np.argsort(f)[::-1][:top_k_features]
    labels_top = [_pretty_feature_name(feature_names[i]) for i in top]

    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, f[top], color="#4C72B0", alpha=0.85,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_top, fontsize=17)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("F-ratio (between / within cluster variance)", fontsize=18)
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ari_sweep(
    results: dict[int, list[float]],
    out_path: Path,
    *,
    title: str = "ARI vs K",
    xlabel: str = "K  (number of authors sampled)",
    x_symbol: str = "K",
    xscale: str = "linear",
) -> None:
    """Mean ± std ARI across x values, with individual trial dots overlaid.
    The subtitle shows the trial count at each x explicitly (since different
    x values may run different numbers of trials).

    `x_symbol` is the short name used in the per-x trial-count subtitle
    (e.g. "K" or "N"). `xscale` is passed to matplotlib."""
    xs_sorted = sorted(results)
    means = [float(np.mean(results[x])) for x in xs_sorted]
    stds = [float(np.std(results[x])) for x in xs_sorted]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    jitter_rng = np.random.RandomState(0)
    # Jitter is proportional to the local spacing so it reads correctly on
    # both linear and log axes with unevenly spaced x values.
    for i, x in enumerate(xs_sorted):
        if len(xs_sorted) > 1:
            left = xs_sorted[i - 1] if i > 0 else x
            right = xs_sorted[i + 1] if i < len(xs_sorted) - 1 else x
            if xscale == "log":
                span = min(x / max(left, 1e-9), max(right, 1e-9) / x) ** 0.5
                xj = x * jitter_rng.uniform(1.0 / (span ** 0.15),
                                            span ** 0.15,
                                            size=len(results[x]))
            else:
                span = max(x - left, right - x, 1)
                xj = x + jitter_rng.uniform(-0.15 * span, 0.15 * span,
                                            size=len(results[x]))
        else:
            xj = np.full(len(results[x]), x, dtype=float)
        ys = results[x]
        ax.scatter(xj, ys, color="#C44E52", alpha=0.35, s=22, zorder=2,
                   label="trial" if x == xs_sorted[0] else None)
    ax.errorbar(xs_sorted, means, yerr=stds, fmt="o-", capsize=5, linewidth=2,
                color="#4C72B0", markeredgecolor="black", markersize=8,
                label="mean ± std", zorder=4)
    ax.set_xscale(xscale)
    ax.set_xticks(xs_sorted)
    ax.set_xticklabels([str(x) for x in xs_sorted])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ARI")
    trial_counts = [len(results[x]) for x in xs_sorted]
    if len(set(trial_counts)) == 1:
        trials_str = f"trials per {x_symbol} = {trial_counts[0]}"
    else:
        trials_str = "trials per " + x_symbol + " — " + "  ".join(
            f"{x_symbol}={x}: {len(results[x])}" for x in xs_sorted)
    ax.set_title(f"{title}\n{trials_str}")
    ax.axhline(0, linestyle=":", color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    for x, m in zip(xs_sorted, means):
        ax.annotate(f"{m:.2f}", (x, m), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color="#4C72B0")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
