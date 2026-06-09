"""
Standalone plot of the depth-50 curve from gradient_health.png (Panel A).

Shows ONLY the 50-layer network's per-layer gradient RMS at initialisation:
the input-facing first layer sits ~10 orders of magnitude below the output
layer -- the vanishing-gradient collapse that kills the deep net.

Reuses the exact computation from gradient_diagnostics.py.
Figure -> outputs/neural_network/gradient_depth50.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from gradient_diagnostics import layer_grad_rms, CSV, OUT, N_INIT, SEED, METADATA

DEPTH = 50

df = pd.read_csv(CSV)
cols = [c for c in df.columns if c not in METADATA]
X = StandardScaler().fit_transform(df[cols].values)
y = LabelEncoder().fit_transform(df["author"])
K = int(y.max() + 1)
Y = np.eye(K)[y]

rng = np.random.default_rng(SEED)
stack = []
for _ in range(N_INIT):
    rms, _ = layer_grad_rms(X, Y, DEPTH, rng)
    stack.append(rms)
mean_layers = np.array(stack).mean(axis=0)
xs = np.arange(1, DEPTH + 1)

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(xs, mean_layers, marker="o", ms=4, lw=1.8, color="#E5B300",
        markeredgecolor="#B58A00", label="50-layer network")
ax.set_yscale("log")
ax.set_xlabel("Hidden layer  (1 = input-facing  →  50 = output-facing)")
ax.set_ylabel("Gradient RMS (log scale)")
ax.set_title("Per-layer gradient magnitude — 50-layer MLP",
             color="#2E5A88", fontweight="bold")
ax.grid(alpha=0.3, which="both")

first, last = mean_layers[0], mean_layers[-1]
ax.annotate(f"input layer\n{first:.1e}", xy=(1, first), xytext=(6, first * 6),
            fontsize=9, color="#B58A00",
            arrowprops=dict(arrowstyle="->", color="#B58A00", lw=1))
ax.annotate(f"output-facing\n{last:.1e}", xy=(DEPTH, last), xytext=(DEPTH - 16, last / 8),
            fontsize=9, color="#B58A00", ha="left",
            arrowprops=dict(arrowstyle="->", color="#B58A00", lw=1))
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
path = OUT / "gradient_depth50.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"first={first:.3e}  last={last:.3e}  ratio={last/first:.2e}")
print(f"Saved figure: {path}")
