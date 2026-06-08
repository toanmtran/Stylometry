"""
Standalone version of Panel A (left) from gradient_health.png:
per-layer gradient RMS at initialisation for ALL depths (1, 3, 10, 50).

Reuses the exact computation from gradient_diagnostics.py.
Figure -> outputs/neural_network/gradient_perlayer.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from gradient_diagnostics import (layer_grad_rms, CSV, OUT, N_INIT, SEED,
                                  METADATA, DEPTHS, ACCENT)

df = pd.read_csv(CSV)
cols = [c for c in df.columns if c not in METADATA]
X = StandardScaler().fit_transform(df[cols].values)
y = LabelEncoder().fit_transform(df["author"])
K = int(y.max() + 1)
Y = np.eye(K)[y]

per_depth_layers = {}
for depth in DEPTHS:
    rng = np.random.default_rng(SEED)
    stack = [layer_grad_rms(X, Y, depth, rng)[0] for _ in range(N_INIT)]
    per_depth_layers[depth] = np.array(stack).mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 4.8))
cmap = plt.get_cmap("viridis")
for i, d in enumerate(DEPTHS):
    ys = per_depth_layers[d]
    xs = np.arange(1, d + 1)
    ax.plot(xs, ys, marker="o", ms=4, lw=1.6,
            color=cmap(i / (len(DEPTHS) - 1)),
            label=f"{d} layer{'s' if d > 1 else ''}")
ax.set_yscale("log")
ax.set_xlabel("Hidden layer  (1 = input-facing  →  deepest = output-facing)")
ax.set_ylabel("Gradient RMS (log scale)")
ax.set_title("Per-layer gradient magnitude", color=ACCENT, fontweight="bold")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, title="Network depth")

plt.tight_layout()
path = OUT / "gradient_perlayer.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved figure: {path}")
