# Neural Network — Open-Set Stylometric Author Classification

A NN authorship classifier trained on 107 stylometric features extracted from
LessWrong blog posts. Five known authors — **Eliezer Yudkowsky, Johnswentworth,
Raemon, Scottalexander, Zvi** — plus a synthetic sixth class,
**`none_of_the_5_authors`**, so the model can answer "written by *none* of the
five" instead of being forced to pick one. **873 passages over 6 classes.**

> The full written report is
> [`../../docs/neural_network_report.tex`](../../docs/neural_network_report.tex).

---

## Layout

```
src/neural_network/
├── data/
│   ├── author_features_extracted_full.csv   # 5-author feature matrix (107 feats × 723 passages)
│   ├── cleaned_5/                            # the 5 known authors (raw cleaned articles)
│   └── cleaned_35/                           # 35-author pool the "none" class is drawn from
├── scripts/
│   ├── neural_network_code.py               # builds none class, trains/evaluates → outputs/results.md
│   ├── make_report_figures.py               # validation heatmap, depth, confusion
│   ├── gradient_diagnostics.py              # per-layer gradient health → outputs/gradient_health.png
│   └── plot_depth50_gradient.py, plot_perlayer_gradient.py   # standalone gradient-curve views
└── outputs/                                  # all generated figures + results.md
```

---

## The `none_of_the_5_authors` class

Built from **15 other LessWrong authors** (not among the five), sampling **10
articles each = 150 "none" passages**, then merged with the five-author feature
matrix. Authors are chosen deterministically with `seed=42` from
`data/cleaned_35/`. Features are extracted on the fly via
`src.features.extract_features`.

Why a dedicated class instead of thresholding a five-author model's confidence?
Indirect thresholding fails — unknown passages routinely get *high*-confidence
predictions for one of the five authors. Training the boundary directly works,
and here it works perfectly (see Results).

---

## Setup

**Dependencies** — Python ≥ 3.10 plus:

| Package | Used for |
|---------|----------|
| `numpy`, `pandas` | data handling |
| `scikit-learn` | `MLPClassifier`, splitting, scaling, metrics |
| `matplotlib` | report figures |
| `nltk` | tokenisation / POS-tagging when building the none-class from raw articles (regex fallback otherwise) |

**Install**

```bash
# from the project root; a virtual environment is recommended
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install numpy pandas scikit-learn matplotlib nltk
```

**Run** (from `src/neural_network/scripts/`)

```bash
cd src/neural_network/scripts

# Build the none class, merge, train the grid, select on validation, retrain the
# winner, evaluate on test. Writes ../outputs/results.md
python neural_network_code.py

# Regenerate every report figure into ../outputs/
# (validation heatmap, depth-effect, confusion matrix)
python make_report_figures.py

# Gradient-health diagnostic -> ../outputs/gradient_health.png
python gradient_diagnostics.py
```

---

## Evaluation Design

Data is split **60 / 20 / 20** (train / validation / test), stratified by class.
Every combination of **3 feature subsets** (top 30, top 50, all 107) × **4
network depths** (1, 3, 10, 50 hidden layers of 64 units) — **12 configs** — is
trained on the train split and ranked by validation accuracy. The winner is
retrained on train+validation, then evaluated **once** on the held-out test set.
No outlier removal: the full feature matrix is used throughout. The top-30 /
top-50 subsets are ranked by **Random Forest permutation importance**
(`n_estimators=500, max_depth=5`); see `scripts/rf_align_check.py`.

```
train (60%) ─► fit all 12 configs ─► rank by valid acc ─► pick best
valid (20%) ─► score all 12 configs ────────────────────────────┘
                                                          │
train+valid ─────────────────────────── retrain best ─────┘
test  (20%) ───────────────────────── final report (touched once)
```

Training uses **early stopping** (`early_stopping=True`, `n_iter_no_change=15`,
`batch_size=32`, `max_iter=500`, Adam), so every architecture stops under the
same rule.

---

## Model Architecture

| Setting | Value |
|---------|-------|
| Type | NN (fully connected feedforward) |
| Hidden activation | ReLU |
| Output activation | Softmax (6 classes) |
| Optimizer | Adam |
| Loss | Cross-entropy |
| Early stopping | patience 15, batch size 32, max 500 epochs |
| Preprocessing | StandardScaler (fit on train split only) |
| Model selection | Validation set (20% held-out) |
| Final evaluation | Test set (20% held-out, touched once) |

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

Selected by validation accuracy: **All 107 features · Depth 1 (64,)** — validation 0.9429.

| Metric | Value |
|--------|-------|
| Test accuracy | **0.9486** |
| Weighted F1 | 0.9485 |

The `none_of_the_5_authors` class is recovered **perfectly** — precision =
recall = F1 = **1.00**. No unknown-author passage is misattributed to one of the
five, and no known-author passage is dumped into the none bin. Among the five
known authors the few errors concentrate on Raemon (recall 0.88) and Scott
Alexander (recall 0.90).

**Depth matters the wrong way.** Accuracy is flat from depth 1 to depth 10, then
collapses to ~0.17 (the 1/6 random baseline) at depth 50 — a vanishing-gradient
optimisation failure, not over-fitting. A shallow NN is preferable.

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
init. See `outputs/gradient_health.png`.
