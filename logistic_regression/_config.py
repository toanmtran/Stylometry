"""
Shared configuration and utilities for the logistic_regression experiments.

Centralises constants, hyperparameter grids, and reusable helpers so each
task script can import them instead of copy-pasting.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent

# Feature CSVs (Task 1)
FEATURES_25AUTHORS = os.path.join(_HERE, "task1_authorship_attribution", "features_25authors.csv")
FEATURES_10AUTHORS = os.path.join(_HERE, "task1_authorship_attribution", "features_10authors.csv")

# Dataset CSVs
LESSWRONG_LARGE = os.path.join(ROOT, "SVM", "LesswrongLarge.csv")

# ── Constants ──────────────────────────────────────────────────────────────────

METADATA_COLS = {"author", "passage_id", "_label"}
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3

PARAM_GRID_LR = {
    "C":        [0.01, 0.1, 1, 10, 100],
    "penalty":  ["l2"],
    "solver":   ["lbfgs"],
    "max_iter": [1000, 2000, 5000],
}

PARAM_GRID_LR_SIMPLE = {
    "C":        [0.1, 1, 10],
    "penalty":  ["l2"],
    "solver":   ["lbfgs"],
    "max_iter": [1000, 2000],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def remove_outliers(
    df: pd.DataFrame,
    contamination: float | str = 0.05,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Remove outliers per author using IsolationForest."""
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    kept = []
    for _, group in df.groupby("author"):
        clf = IsolationForest(contamination=contamination, random_state=random_state)
        mask = clf.fit_predict(group[feature_cols].values) == 1
        kept.append(group[mask])
    return pd.concat(kept).reset_index(drop=True)


def setup_project_path() -> None:
    """Add project root to sys.path so src.* imports work."""
    import sys
    project_root = str(ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def n_param_combos(param_grid: dict) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


def configure_stdout_utf8() -> None:
    """Ensure stdout uses UTF-8 encoding on Windows."""
    import io
    import sys
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def preprocess_data(
    df: pd.DataFrame,
    with_outliers: bool = True,
    contamination: float | str = 0.05,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str], StandardScaler]:
    """
    Prepare X, y for classification.

    Returns X (unscaled), y (encoded), label_encoder, feature_cols, scaler.
    The scaler is fit on all data — callers doing CV MUST re-fit per fold.
    """
    if not with_outliers:
        df = remove_outliers(df, contamination=contamination)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["author"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, feature_cols, scaler


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict | None = None,
    cv: int = INNER_FOLDS,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Train a LogisticRegression with GridSearchCV on the full dataset."""
    if param_grid is None:
        param_grid = PARAM_GRID_LR

    gs = GridSearchCV(
        LogisticRegression(random_state=random_state, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_


# ── Verification helpers ───────────────────────────────────────────────────────

def generate_pair_features(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Compute pairwise feature vector from two individual feature vectors."""
    return np.concatenate([
        np.abs(f1 - f2),
        (f1 - f2) ** 2,
        f1 * f2,
    ])
