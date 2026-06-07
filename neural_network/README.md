# Neural Network — Stylometric Author Classification (5-class)

An MLP authorship classifier trained on 107 stylometric features extracted from
LessWrong blog posts. Five target authors: **Eliezer Yudkowsky, Johnswentworth,
Raemon, Scottalexander, Zvi** — 723 passages total.

> The open-set variant (adds a sixth `none_of_the_5_authors` class) lives in
> [`../neural_network_6class/`](../neural_network_6class/). The full written
> report is [`../docs/neural_network_report.tex`](../docs/neural_network_report.tex).

---

## Setup

**Dependencies** — Python ≥ 3.10 plus:

| Package | Used for |
|---------|----------|
| `numpy`, `pandas` | data handling |
| `scikit-learn` | `MLPClassifier`, splitting, scaling, metrics |
| `matplotlib` | ROC curves and report figures |
| `nltk` | tokenisation / POS-tagging (only needed by the 6-class none-class build and figure script; regex fallback otherwise) |

**Install**

```bash
# from the project root; a virtual environment is recommended
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install numpy pandas scikit-learn matplotlib nltk
```

**Run**

```bash
# Train the grid, select on dev, retrain the winner, evaluate on test.
# Writes results.md + roc.png in this folder.
cd neural_network
python neural_network_code.py

# Regenerate every report figure into outputs/neural_network/
# (dev heatmaps, depth-effect, confusion matrices, ROC, seed-stability sweep)
python make_report_figures.py

# Gradient-health diagnostic -> outputs/neural_network/gradient_health.png
python gradient_diagnostics.py
```

---

## Evaluation Design

Data is split **60 / 20 / 20** (train / dev / test), stratified by author. Every
combination of **3 feature subsets** (top 30, top 50, all 107) × **4 network
depths** (1, 3, 10, 50 hidden layers of 64 units) — **12 configs** — is trained
on the train split and ranked by dev accuracy. The winner is retrained on
train+dev, then evaluated **once** on the held-out test set. No outlier removal:
the full feature matrix is used throughout.

```
train (60%) ─► fit all 12 configs ─► rank by dev acc ─► pick best
dev   (20%) ─► score all 12 configs ────────────────────────────┘
                                                         │
train+dev ──────────────────────────── retrain best ─────┘
test  (20%) ───────────────────────── final report (touched once)
```

Training uses **early stopping** (`early_stopping=True`, `n_iter_no_change=15`,
`batch_size=32`, `max_iter=500`, Adam), so every architecture stops under the
same rule.

---

## Model Architecture

| Setting | Value |
|---------|-------|
| Type | MLP (fully connected feedforward) |
| Hidden activation | ReLU |
| Output activation | Softmax (5 classes) |
| Optimizer | Adam |
| Loss | Cross-entropy |
| Early stopping | patience 15, batch size 32, max 500 epochs |
| Preprocessing | StandardScaler (fit on train split only) |
| Model selection | Dev set (20% held-out) |
| Final evaluation | Test set (20% held-out, touched once) |

---

## Files

| File | Description |
|------|-------------|
| `neural_network_code.py` | Main training/evaluation script → `results.md`, `roc.png` |
| `make_report_figures.py` | Regenerates all report figures into `../outputs/neural_network/` |
| `gradient_diagnostics.py` | Numpy MLP that measures per-layer gradient health at init → `gradient_health.png` |
| `author_features_extracted_full.csv` | Full feature matrix: 107 features × 723 passages, keyed by `author`, `passage_id` |
| `results.md` | Latest dev/test evaluation (generated) |
| `roc.png` | One-vs-rest ROC curves for the selected model (generated) |
| `config.yaml` | Feature/run configuration |
| `evaluation_results_full_and_subsets (1).md`, `with_early_stopping.md` | Legacy 5-fold CV results (superseded by `results.md`) |

### Feature groups (107 total)

| Category | Examples |
|----------|---------|
| Lexical richness | `hapax_ratio`, `yule_k`, `simpson_d`, `brunet_w`, `honore_r` |
| Word-length stats | `avg_word_len`, `std_word_len`, `word_len_1_frac` … `word_len_6_frac` |
| Sentence stats | `avg_sent_len`, `std_sent_len`, `median_sent_len` |
| Function words | `fw_the`, `fw_to`, `fw_i` … (top-50 English function words) |
| Punctuation rates | `punct_comma_rate`, `punct_semicolon_rate`, `punct_paren_rate` … |
| Category rates | `cat_hedge_rate`, `cat_amplifier_rate`, `cat_conj_rate` |
| POS proportions | `pos_noun`, `pos_verb`, `pos_adj`, `pos_adv`, `pos_pron` … |
| Surface / character | `uppercase_ratio`, `contraction_rate`, `char_a_rate` … `char_u_rate` |

---

## Results

Selected by dev accuracy: **All 107 features · Depth 1 (64,)** — dev 0.9034.

| Metric | Value |
|--------|-------|
| Test accuracy | **0.9310** |
| Weighted F1 | 0.9297 |
| ROC-AUC (macro OvR) | 0.9951 |

**Depth matters the wrong way.** Accuracy is flat from depth 1 to depth 10, then
collapses to ~0.21 (the 1/5 random baseline) at depth 50 — a vanishing-gradient
optimisation failure, not over-fitting. A shallow MLP is preferable.

**Gradient diagnostic** (`gradient_diagnostics.py`). Measuring the RMS weight
gradient at initialisation (Glorot-uniform init, exactly sklearn's ReLU scheme,
averaged over 50 inits) confirms the cause: the gradient reaching the
input-facing first layer collapses with depth.

| Depth | First-layer grad RMS | Output grad RMS | Ratio (out/first) |
|------:|---------------------:|----------------:|------------------:|
| 1  | 2.0×10⁻² | 7.0×10⁻² | 3.5 |
| 3  | 7.4×10⁻³ | 3.1×10⁻² | 4.2 |
| 10 | 4.2×10⁻⁴ | 5.8×10⁻³ | 13.7 |
| 50 | 2.5×10⁻¹⁰ | 5.5×10⁻³ | 2.2×10⁷ |

At 50 layers the first layer sees a gradient ~8 orders of magnitude weaker than
in the shallow net — it cannot learn, so the network never leaves its random
init. See `outputs/neural_network/gradient_health.png`.

**Seed stability.** Retrained under 10 seeds (split + weight init), test accuracy
is **0.931 ± 0.018** (range 0.890–0.959); the `random_state=42` headline sits at
the mean, so it is a representative run.
