"""Top feature importances by Random Forest permutation importance.

Reads the rankings dumped by rf_align_check.py (outputs/rf_rankings.json) and
draws a clean horizontal bar chart of the most informative stylometric features
-- the ones that define the top-30 / top-50 subsets used in the grid search.

Figure -> outputs/rf_permutation_importance.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

NN_DIR = Path(__file__).resolve().parent.parent   # src/neural_network/
OUT = NN_DIR / "outputs"
ACCENT = "#2E5A88"
TOP_N = 15


def main():
    data = json.loads((OUT / "rf_rankings.json").read_text())
    pi = data["permutation_importances"]
    items = list(pi.items())[:TOP_N]
    names = [k for k, _ in items][::-1]      # most important on top
    vals = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(names, vals, color=ACCENT, edgecolor="white", height=0.78)
    ax.set_xlabel("Permutation importance (mean test-accuracy drop)")
    ax.set_title(f"Top {TOP_N} features by Random Forest permutation importance",
                 color=ACCENT, fontweight="bold", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = OUT / "rf_permutation_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
