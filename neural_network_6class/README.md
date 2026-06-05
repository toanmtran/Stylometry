# Neural Network — Open-Set Author Classification (6-class)

The open-set variant of the [5-class MLP](../neural_network/). It adds a sixth
class, **`none_of_the_5_authors`**, so the model can answer "this passage was
written by *none* of the five known authors" instead of being forced to pick
one. Total: **873 passages over 6 classes**.

> The full written report is
> [`../docs/neural_network_report.tex`](../docs/neural_network_report.tex)
> (Section 4, "Open-Set Recognition").

---

## The `none_of_the_5_authors` class

Built from **15 other LessWrong authors** (not among the five), sampling **10
articles each = 150 "none" passages**, then merged with the five-author feature
matrix. Authors are chosen deterministically with `seed=42` from
`dataset/lesswrong_large/cleaned_35/`. Features are extracted on the fly via
`src.features.extract_features`.

Why a dedicated class instead of thresholding the 5-class model's confidence?
Indirect thresholding fails — unknown passages routinely get *high*-confidence
predictions for one of the five authors. Training the boundary directly works,
and here it works perfectly (see Results).

---

## Setup

**Dependencies** — same as the 5-class folder: `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, and `nltk` (used to extract features from the raw
none-class articles; regex fallback if NLTK data is missing). Python ≥ 3.10.

```bash
# from the project root
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install numpy pandas scikit-learn matplotlib nltk
```

**Run**

```bash
# Builds the none class, merges, trains the grid, selects on dev,
# retrains the winner, evaluates on test. Writes results.md + roc.png here.
cd neural_network_6class
python neural_network_6class.py
```

It reads the 5-author features from
`../neural_network/author_features_extracted_full.csv` and the raw none-class
articles from `../dataset/lesswrong_large/cleaned_35/`.

---

## Evaluation Design

Identical protocol to the 5-class model: **60 / 20 / 20** stratified split, a
**12-config** grid (3 feature subsets × 4 depths) ranked on dev, winner
retrained on train+dev and evaluated **once** on test. Early stopping
(`patience=15`, `batch_size=32`, `max_iter=500`, Adam). No outlier removal.

---

## Files

| File | Description |
|------|-------------|
| `neural_network_6class.py` | Builds the none class, trains/evaluates → `results.md`, `roc.png` |
| `results.md` | Latest dev/test evaluation (generated) |
| `roc.png` | One-vs-rest ROC curves for the selected model (generated) |

---

## Results

Selected by dev accuracy: **All 107 features · Depth 1 (64,)** — dev 0.9429.

| Metric | Value |
|--------|-------|
| Test accuracy | **0.9486** |
| Weighted F1 | 0.9485 |
| ROC-AUC (macro OvR) | 0.9960 |

The `none_of_the_5_authors` class is recovered **perfectly** — precision =
recall = F1 = **1.00**. No unknown-author passage is misattributed to one of the
five, and no known-author passage is dumped into the none bin. Adding the class
even *raises* overall accuracy versus the closed-set 5-class model (0.949 vs
0.931): contrasting against 15 diverse "other" authors sharpens the boundaries
between the five targets.

**Seed stability.** Across 10 seeds (split + weight init, none class fixed), test
accuracy is **0.943 ± 0.015** (range 0.909–0.960) — tighter than the 5-class
spread, anchored by the perfectly separable none class.
