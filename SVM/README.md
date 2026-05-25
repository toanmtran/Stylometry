# Author Verification using Support Vector Machine

## Overview
This project implements an author verification model using:
- Pairwise feature engineering
- Support Vector Machine (SVM)
- Author-based train/test splitting
- Multi-seed evaluation

The system determines whether two documents are written by the same author.

---

## Project Structure

```text
ML_Project/
│
├── data/
│   └── LesswrongLarge.csv                # Raw Dataset
│   └── processed_data_seed_19.npz        # Processed Dataset (Seeds 19 to 23)
│   └── ...
│   └── processed_data_seed_23.npz        
│
├── src/                                  # Source code folder
│   ├── data_prepare.py                   # Module for data preparing
│   ├── grid_search.py                    # Module for performing Grid Search
│   └── final_evaluation.py               # Module for traning final model and evaluation
│   └── Demo.ipynb                        # Notebook for demostrating workflow and results
│
├── README.md                             
└── Report.pdf                            
```

## Main Components

### 1. data_prepare.py
Responsible for:
- Loading the raw dataset
- Splitting authors into training/test sets
- Generating pairs for test set
- Constructing pairwise interaction features
- Saving processed datasets

Generated pairwise features:
- Absolute Difference: |f1 - f2|
- Squared Difference: (f1 - f2)^2
- Element-wise Product: f1 * f2

---

### 2. grid_search.py
Responsible for:
- Author-based 5-Fold Cross Validation
- Generating pairs for training and validation sets
- Hyperparameter tuning for:
  - C
  - gamma
- Choosing the optimal hyperparameters for the final model

---

### 3. final_evaluation.py
Responsible for:
- Final SVM training
- Testing on unseen authors
- Reporting:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix

---

## Experimental Design

### Author-Level Splitting
Documents written by the same author never appear simultaneously in both training and testing sets.

### Multi-Seed Evaluation
The entire pipeline is repeated across 5 independent random seeds:
- Seed 19
- Seed 20
- Seed 21
- Seed 22
- Seed 23

Final performance is reported using:
- Mean
- Standard Deviation

---

## Requirements

Install required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## Workflow/ Usage

The entire workflow was visually packaged in notebook file: `Demo.ipynb`.

---

## Results

| Metric | Mean ± Std |
|---|---|
| Accuracy | 0.7326 ± 0.0189 |
| ROC-AUC | 0.8139 ± 0.0186 |

The results demonstrate stable and balanced authorship verification performance across multiple random author splits.

---
