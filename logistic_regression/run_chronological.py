"""
Chronological Stylometry — Per-Author Time-Period Classification

Goal: Determine whether an author's writing style changes measurably over time.

Approach:
  1. For each author with sufficient pre-2022 AND post-2024 passages:
  2. Extract stylometric features
  3. Train a binary classifier (Logistic Regression) to distinguish
     "early" (<=2022) vs "late" (>=2024) periods
  4. Analyze which features change most over time

Data source: outputs/corpus_stats/lesswrong_regular/cleaned/stats_min1000.csv
lists per-author pre/post counts. We use the cleaned dataset to get actual texts.

Usage:
    python run_chronological.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import extract_features
from src.preprocess import (
    strip_lesswrong_boilerplate,
    canonicalize_typography,
    word_count,
    truncate_to_words,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


# ── Config ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
MIN_PASSAGES_PER_PERIOD = 15  # Minimum passages per period to include author


def load_passages_with_dates(folder: Path) -> list[dict]:
    """Load all passages with their date info from cleaned JSON files."""
    records = []
    for fp in sorted(folder.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            raw = f.read()
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            import re
            cleaned = re.sub(r",\s*([\]}])", r"\1", raw)
            items = json.loads(cleaned)
        for item in items:
            text = item.get("text", "")
            author = item.get("author", "Unknown").strip().lower()
            date = item.get("date", "")
            records.append({
                "author": author,
                "text": text,
                "date": date,
            })
    return records


def classify_era(date_str: str) -> str | None:
    """Classify a date string into 'early' (<=2022), 'late' (>=2024), or None."""
    if not date_str or len(date_str) < 4:
        return None
    try:
        year = int(date_str[:4])
    except ValueError:
        return None
    if year <= 2022:
        return "early"
    elif year >= 2024:
        return "late"
    else:
        return None  # 2023 — ambiguous, skip


def main():
    parser = argparse.ArgumentParser(description="Chronological Stylometry Analysis")
    parser.add_argument("--data-dir",
                        default=os.path.join("..", "dataset", "lesswrong_regular", "cleaned"),
                        help="Path to cleaned dataset folder")
    parser.add_argument("--output-dir", default=".", help="Output directory for results")
    parser.add_argument("--min-words", type=int, default=500, help="Min words per passage")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading passages from {data_path}...")
    all_passages = load_passages_with_dates(data_path)
    print(f"  Total passages loaded: {len(all_passages)}")

    # Classify era and filter
    for p in all_passages:
        p["era"] = classify_era(p["date"])

    # Group by author
    author_passages = defaultdict(lambda: {"early": [], "late": []})
    for p in all_passages:
        era = p["era"]
        if era is None:
            continue
        author_passages[p["author"]][era].append(p)

    print(f"\n{'='*60}")
    print("  Per-Author Chronological Analysis")
    print(f"{'='*60}")

    all_results = []

    for author, periods in sorted(author_passages.items()):
        early_passages = periods["early"]
        late_passages = periods["late"]

        if len(early_passages) < MIN_PASSAGES_PER_PERIOD or \
           len(late_passages) < MIN_PASSAGES_PER_PERIOD:
            print(f"\n  Skipping {author}: {len(early_passages)} early, "
                  f"{len(late_passages)} late (< {MIN_PASSAGES_PER_PERIOD})")
            continue


        print(f"\n  {'-'*40}")
        print(f"  Author: {author}")
        print(f"  Early passages (<=2022): {len(early_passages)}")
        print(f"  Late passages  (>=2024): {len(late_passages)}")

        print(f"  {'-'*40}")

        # Clean texts
        early_clean = []
        for p in early_passages:
            text = canonicalize_typography(p["text"])
            text = strip_lesswrong_boilerplate(text)
            if word_count(text) >= args.min_words:
                early_clean.append(text)

        late_clean = []
        for p in late_passages:
            text = canonicalize_typography(p["text"])
            text = strip_lesswrong_boilerplate(text)
            if word_count(text) >= args.min_words:
                late_clean.append(text)

        print(f"  After cleaning (>{args.min_words}w): {len(early_clean)} early, {len(late_clean)} late")
        if len(early_clean) < 10 or len(late_clean) < 10:
            print(f"  ⚠ Too few after cleaning, skipping.")
            continue

        # Truncate to same length
        all_texts = early_clean + late_clean
        min_wc = min(word_count(t) for t in all_texts)
        all_texts = [truncate_to_words(t, min_wc) for t in all_texts]

        # Extract features
        X_list = []
        for t in all_texts:
            X_list.append(list(extract_features(t).values()))
        feature_names = list(extract_features(all_texts[0]).keys())
        X = np.array(X_list)
        y = np.array([0] * len(early_clean) + [1] * len(late_clean))

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Logistic Regression with cross-validation
        clf = LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")

        # Train on full data for feature analysis
        clf.fit(X_scaled, y)
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": clf.coef_[0],
            "abs_coef": np.abs(clf.coef_[0]),
        }).sort_values("abs_coef", ascending=False)

        result = {
            "author": author,
            "n_early": len(early_clean),
            "n_late": len(late_clean),
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "top_features_early": coef_df[coef_df["coefficient"] < 0].head(5)[
                ["feature", "coefficient"]
            ].to_string(index=False),
            "top_features_late": coef_df[coef_df["coefficient"] > 0].head(5)[
                ["feature", "coefficient"]
            ].to_string(index=False),
        }
        all_results.append(result)

        print(f"  CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
        print(f"\n  Top features associated with EARLY period (negative coef):")
        for _, row in coef_df[coef_df["coefficient"] < 0].head(5).iterrows():
            print(f"    {row['feature']}: {row['coefficient']:.4f}")
        print(f"\n  Top features associated with LATE period (positive coef):")
        for _, row in coef_df[coef_df["coefficient"] > 0].head(5).iterrows():
            print(f"    {row['feature']}: {row['coefficient']:.4f}")

    # ── Summary ─────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo authors with sufficient data found.")
        return

    print(f"\n{'='*60}")
    print("  SUMMARY: Chronological Stylometry")
    print(f"{'='*60}")
    print(f"{'Author':25s} {'Early':>6s} {'Late':>6s} {'Accuracy':>10s} {'Std':>6s}")
    print(f"{'-'*25} {'-'*6} {'-'*6} {'-'*10} {'-'*6}")
    for r in all_results:
        print(f"{r['author']:25s} {r['n_early']:6d} {r['n_late']:6d} "
              f"{r['mean_accuracy']:.4f}     {r['std_accuracy']:.4f}")

    # Save markdown report
    md_path = output_path / "chronological_results.md"
    print(f"\nSaving report to {md_path}...")

    md = "# Chronological Stylometry Results\n\n"
    md += "## Per-Author Summary\n\n"
    md += "| Author | Early (<=2022) | Late (>=2024) | CV Accuracy | Std |\n"
    md += "|--------|----------------|---------------|-------------|-----|\n"
    for r in all_results:
        md += f"| {r['author']} | {r['n_early']} | {r['n_late']} | "
        md += f"{r['mean_accuracy']:.4f} | {r['std_accuracy']:.4f} |\n"

    md += "\n## Feature Analysis Per Author\n\n"
    for r in all_results:
        md += f"### {r['author']}\n\n"
        md += f"**Accuracy:** {r['mean_accuracy']:.4f} +/- {r['std_accuracy']:.4f}\n\n"
        md += "**Features → Early period:**\n\n"
        md += f"```\n{r['top_features_early']}\n```\n\n"
        md += "**Features → Late period:**\n\n"
        md += f"```\n{r['top_features_late']}\n```\n\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
