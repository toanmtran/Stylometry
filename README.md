# Authorship Identification via Writing Style — A Case Study on LessWrong

A comparative study of three machine-learning approaches to authorship identification,
trained on essays crawled from [LessWrong](https://www.lesswrong.com/). The project
explores **K-Means clustering**, **SVM / Logistic Regression** pair-classifiers, and a **Multi-Layer Perceptron** classifier, sharing a common handcrafted stylometric feature space.

The full write-up is available in [docs/project_report.pdf](docs/project_report.pdf).

---

## Getting the project

The combined source code and dataset exceed the MS Teams upload limit, so the
project is hosted on GitHub instead. The full repository — code, dataset, and
report — can be downloaded in either of two ways:

**Option 1 — Clone with Git (recommended):**

```bash
git clone https://github.com/toanmtran/Stylometry.git
cd Stylometry
```

**Option 2 — Download as a ZIP archive:**

1. Open <https://github.com/toanmtran/Stylometry> in a browser.
2. Click the green **Code** button → **Download ZIP**.
3. Extract the archive and `cd` into the resulting `Stylometry-main/` folder.

Once you have the project locally, follow the [Installation](#installation)
section below.

---

## Repository structure

```
Stylometry/
├── dataset/
│   ├── raw/               # Raw JSON crawled from LessWrong (5, 10, 35 authors)
│   ├── cleaned/           # Cleaned / typography-normalised corpora
│   └── scripts/           # Crawlers + raw vs. cleaned statistics
├── src/
│   ├── kmeans/            # K-Means clustering + ARI sweep experiments
│   ├── neural_network/    # MLP classifier + gradient / feature diagnostics
│   └── svm&logisticregression/   # Pairwise SVM and Logistic Regression
├── docs/                  # LaTeX sources + compiled PDF reports
├── requirements.txt
└── README.md
```

---

## Installation

The project targets **Python 3.10+**. All three problems share the same environment.

```bash
# 1. Clone the repo (skip if you already did this above)
git clone https://github.com/toanmtran/Stylometry.git
cd Stylometry

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

NLTK resources (`punkt`, `averaged_perceptron_tagger`) are downloaded
automatically the first time the feature extractor runs.

---

## Dataset

Cleaned corpora are already shipped under [dataset/cleaned/](dataset/cleaned/)
(`cleaned_5`, `cleaned_10`, `cleaned_35`). To **re-crawl** from LessWrong:

```bash
python dataset/scripts/crawl_lesswrong5.py     # 5  authors
python dataset/scripts/crawl_lesswrong10.py    # 10 authors
python dataset/scripts/crawl_lesswrong35.py    # 35 authors

python dataset/scripts/stats_raw.py            # corpus statistics on raw/
python dataset/scripts/stats_clean.py          # corpus statistics on cleaned/
```

---

## Running the experiments

All commands are run from the **project root** (`Stylometry/`).

### 1. K-Means clustering

The k-means pipeline (feature extraction → KMeans → ARI evaluation) is split into reusable modules under
[src/kmeans/scripts/](src/kmeans/scripts/) and experiment drivers under
[experiments/](src/kmeans/scripts/experiments/).

```bash
# Effect of number of clusters K
python -m src.kmeans.scripts.experiments.effect_of_k

# Effect of articles-per-author M
python -m src.kmeans.scripts.experiments.effect_of_m

# Effect of document length N
python -m src.kmeans.scripts.experiments.effect_of_n

# Effect of publication era
python -m src.kmeans.scripts.experiments.effect_of_era

# Final 10-cluster visualisation
python -m src.kmeans.scripts.experiments.cluster_10
```

Outputs (figures, cached feature matrices) are written to
`src/kmeans/outputs/` and `src/kmeans/cache/`.

### 2. Neural Network (MLP)

A six-class classifier (5 known authors + a synthetic `none_of_the_5_authors`
class) with grid-searched depth and feature-subset.

```bash
# Main classifier — produces results.md + figures
python src/neural_network/scripts/neural_network_code.py

# Random-Forest permutation importance (feature ranking)
python src/neural_network/scripts/rf_align_check.py

# Gradient diagnostics + per-layer / depth-50 plots
python src/neural_network/scripts/gradient_diagnostics.py
python src/neural_network/scripts/plot_perlayer_gradient.py
python src/neural_network/scripts/plot_depth50_gradient.py

# Report figures and slide deck
python src/neural_network/scripts/make_report_figures.py
python src/neural_network/scripts/plot_feature_importance.py
python src/neural_network/scripts/make_slides.py
```

Outputs land in `src/neural_network/outputs/`.

### 3. SVM and Logistic Regression

Pairwise same-author / different-author classifiers. The scripts use sibling
imports, so run them from inside their own folder:

```bash
cd src/svm&logisticregression/scripts

# Build pair datasets across seeds (writes processed_data_seed_*.npz)
python data_prepare.py

# Hyper-parameter grid search (C, gamma) with K-fold CV
python grid_search.py

# Logistic Regression baseline
python logistic_regression.py

# Final SVM evaluation (test accuracy, ROC, confusion matrix)
python final_evaluation.py
```

A walk-through notebook is also available at
[src/svm&logisticregression/scripts/demo.ipynb](src/svm&logisticregression/scripts/demo.ipynb).