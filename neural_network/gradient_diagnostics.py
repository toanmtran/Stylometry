"""
Gradient-health diagnostic: why deep plain MLPs fail on this task.

The depth sweep shows accuracy collapsing at 50 layers. This script checks
whether that collapse is a *vanishing-gradient* optimisation failure by measuring
the per-layer gradient magnitude at initialisation -- the moment that decides
whether training can get off the ground.

We re-implement the exact architecture sklearn's MLPClassifier uses (Glorot-
uniform init with bound sqrt(6 / (fan_in + fan_out)) for ReLU, ReLU hidden units,
softmax output, cross-entropy loss) in numpy so we can read off the gradient of
the loss w.r.t. every layer's weights from a single forward+backward pass. We
average over many random initialisations and report, per depth:

  * the RMS gradient reaching the input-facing first layer (the layer that must
    learn feature combinations),
  * the RMS gradient at the output layer,
  * their ratio (the attenuation factor across the network).

Figure -> outputs/neural_network/gradient_health.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "neural_network"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT, ACCENTRED = "#2E5A88", "#C44E52"
DEPTHS   = [1, 3, 10, 50]
HIDDEN   = 64
N_INIT   = 50          # random initialisations to average over
SEED     = 42
METADATA = {"author", "passage_id"}
CSV = Path(__file__).resolve().parent / "author_features_extracted_full.csv"


def glorot_uniform(rng, fan_in, fan_out):
    bound = np.sqrt(6.0 / (fan_in + fan_out))      # sklearn's ReLU init bound
    return rng.uniform(-bound, bound, size=(fan_in, fan_out))


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def layer_grad_rms(X, Y, depth, rng):
    """One forward+backward pass; return (per-hidden-layer RMS list, output RMS).

    Layer 0 is input-facing (D->64); layer depth-1 is the last hidden (64->64);
    the output layer is 64->K. RMS = root-mean-square of the weight-gradient
    entries, so layers of different sizes stay comparable.
    """
    D, K = X.shape[1], Y.shape[1]
    dims = [D] + [HIDDEN] * depth
    Ws = [glorot_uniform(rng, dims[i], dims[i + 1]) for i in range(depth)]
    bs = [rng.uniform(-np.sqrt(6.0 / (dims[i] + dims[i + 1])),
                      np.sqrt(6.0 / (dims[i] + dims[i + 1])), dims[i + 1])
          for i in range(depth)]
    W_out = glorot_uniform(rng, HIDDEN, K)
    b_out = rng.uniform(-np.sqrt(6.0 / (HIDDEN + K)),
                        np.sqrt(6.0 / (HIDDEN + K)), K)

    # forward
    a = X
    acts, zs = [a], []
    for l in range(depth):
        z = a @ Ws[l] + bs[l]
        a = relu(z)
        zs.append(z)
        acts.append(a)
    probs = softmax(a @ W_out + b_out)

    # backward (cross-entropy + softmax)
    n = X.shape[0]
    delta = (probs - Y) / n
    g_out = acts[-1].T @ delta
    delta = (delta @ W_out.T) * (zs[-1] > 0)

    grads = [None] * depth
    for l in range(depth - 1, -1, -1):
        grads[l] = acts[l].T @ delta
        if l > 0:
            delta = (delta @ Ws[l].T) * (zs[l - 1] > 0)

    rms = [float(np.sqrt(np.mean(g ** 2))) for g in grads]
    return rms, float(np.sqrt(np.mean(g_out ** 2)))


def main():
    print(f"Loading {CSV.name} ...")
    df = pd.read_csv(CSV)
    cols = [c for c in df.columns if c not in METADATA]
    X = StandardScaler().fit_transform(df[cols].values)
    y = LabelEncoder().fit_transform(df["author"])
    K = int(y.max() + 1)
    Y = np.eye(K)[y]
    print(f"  {X.shape[0]} passages x {X.shape[1]} features, {K} classes")

    # collect per-layer RMS averaged over N_INIT inits, for each depth
    per_depth_layers = {}   # depth -> mean RMS per hidden layer (input->output)
    summary = {}            # depth -> (first_rms, out_rms, ratio)
    for depth in DEPTHS:
        rng = np.random.default_rng(SEED)
        stack, outs = [], []
        for _ in range(N_INIT):
            rms, rms_out = layer_grad_rms(X, Y, depth, rng)
            stack.append(rms)
            outs.append(rms_out)
        mean_layers = np.array(stack).mean(axis=0)     # length = depth
        mean_out = float(np.mean(outs))
        per_depth_layers[depth] = mean_layers
        first = float(mean_layers[0])
        summary[depth] = (first, mean_out, mean_out / first)

    # ---- report table ----
    print("\nGradient health at initialisation (RMS of weight gradients):")
    print(f"  {'depth':>5} | {'first-layer':>12} | {'output':>10} | {'ratio out/first':>16}")
    print("  " + "-" * 52)
    for d in DEPTHS:
        first, out, ratio = summary[d]
        print(f"  {d:>5} | {first:12.3e} | {out:10.3e} | {ratio:16.1f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: per-layer gradient RMS, input(1) -> output(depth)
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(DEPTHS):
        ys = per_depth_layers[d]
        xs = np.arange(1, d + 1)
        axes[0].plot(xs, ys, marker="o", ms=4, lw=1.6,
                     color=cmap(i / (len(DEPTHS) - 1)), label=f"{d} layer{'s' if d > 1 else ''}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Hidden layer (1 = input-facing  ->  deepest = output-facing)")
    axes[0].set_ylabel("Gradient RMS (log scale)")
    axes[0].set_title("Per-layer gradient magnitude", color=ACCENT, fontweight="bold")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, title="Network depth")

    # Panel B: first-layer gradient RMS vs depth
    firsts = [summary[d][0] for d in DEPTHS]
    axes[1].plot(DEPTHS, firsts, marker="o", color=ACCENTRED, lw=1.8)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xticks(DEPTHS, [str(d) for d in DEPTHS])
    axes[1].set_xlabel("Total network depth (log scale)")
    axes[1].set_ylabel("First-layer gradient RMS (log scale)")
    axes[1].set_title("Gradient reaching the input layer", color=ACCENT, fontweight="bold")
    axes[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    path = OUT / "gradient_health.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved figure: {path}")


if __name__ == "__main__":
    main()
