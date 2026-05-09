"""
Extract stylometric features from cleaned datasets for N authors.

Usage:
    python scripts/extract_features_for_n_authors.py --input ../dataset/lesswrong_regular/cleaned_10 --output ../datasets/features_10authors.csv
    python scripts/extract_features_for_n_authors.py --input ../dataset/lesswrong_large/cleaned_25 --output ../datasets/features_25authors.csv
    python scripts/extract_features_for_n_authors.py --input ../dataset/lesswrong_large/cleaned_35 --output ../datasets/features_35authors.csv

This enables scaling experiments: train Logistic Regression on 10, 25, 35 authors
and compare performance with the 5-author baseline.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add src to path so we can import features & preprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import extract_features
from src.preprocess import (
    LoadedDoc,
    strip_lesswrong_boilerplate,
    canonicalize_typography,
    word_count,
)


def load_json_folder(folder: Path) -> list[dict]:
    """Load all JSON files from a folder; return list of {author, text, file}."""
    records = []
    for fp in sorted(folder.glob("*.json")):
        file_stem = fp.stem
        with open(fp, encoding="utf-8-sig") as f:
            raw = f.read()
        # Handle possible trailing commas / control chars
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            # Try cleaning control chars
            import re
            cleaned = re.sub(r",\s*([\]}])", r"\1", raw)
            items = json.loads(cleaned)
        for i, item in enumerate(items):
            text = item.get("text", "")
            author = item.get("author", "Unknown").strip().lower()
            records.append({
                "author": author,
                "text": text,
                "file": file_stem,
                "idx": i,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract stylometric features for N authors")
    parser.add_argument("--input", required=True, help="Path to cleaned_N folder")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--min-words", type=int, default=500,
                        help="Minimum word count after cleaning (default: 500)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    print(f"Loading data from {input_path}...")
    records = load_json_folder(input_path)
    print(f"  Loaded {len(records)} raw records")

    # Clean and filter
    valid = []
    dropped = 0
    for rec in records:
        text = canonicalize_typography(rec["text"])
        text = strip_lesswrong_boilerplate(text)
        wc = word_count(text)
        if wc < args.min_words:
            dropped += 1
            continue
        rec["text_clean"] = text
        rec["word_count"] = wc
        valid.append(rec)

    print(f"  After cleaning: {len(valid)} valid, {dropped} dropped (< {args.min_words} words)")

    # Truncate all to same length (min word count)
    M = min(r["word_count"] for r in valid)
    print(f"  Truncating all to M = {M} words")

    from src.preprocess import truncate_to_words
    for rec in valid:
        rec["text_clean"] = truncate_to_words(rec["text_clean"], M)

    # Extract features
    print("  Extracting features...")
    rows = []
    for rec in valid:
        feats = extract_features(rec["text_clean"])
        feats["author"] = rec["author"]
        feats["passage_id"] = f"{rec['file']}_{rec['idx']:04d}"
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"  Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")

    # Count authors
    n_authors = df["author"].nunique()
    per_author = df.groupby("author").size()
    print(f"  Authors: {n_authors}")
    print(f"  Per-author distribution:")
    for auth, cnt in per_author.items():
        print(f"    {auth}: {cnt}")

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
