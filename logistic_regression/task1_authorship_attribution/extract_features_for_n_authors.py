"""
Extract stylometric features from cleaned datasets for N authors.

Usage:
    python extract_features_for_n_authors.py --input ../../dataset/lesswrong_regular/cleaned_10 --output features_10authors.csv
    python extract_features_for_n_authors.py --input ../../dataset/lesswrong_large/cleaned_25 --output features_25authors.csv
    python extract_features_for_n_authors.py --input ../../dataset/lesswrong_large/cleaned_35 --output features_35authors.csv
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features import extract_features
from src.preprocess import (
    load_folder,
    canonicalize_typography,
    strip_lesswrong_boilerplate,
    word_count,
    truncate_to_words,
)


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
    docs = load_folder(input_path)
    print(f"  Loaded {len(docs)} raw records")

    # Clean and filter
    valid = []
    dropped = 0
    for d in docs:
        text = canonicalize_typography(d.text)
        text = strip_lesswrong_boilerplate(text)
        wc = word_count(text)
        if wc < args.min_words:
            dropped += 1
            continue
        valid.append({
            "author": d.declared_author,
            "text_clean": text,
            "word_count": wc,
            "file": d.file_stem,
            "idx": d.index,
        })

    print(f"  After cleaning: {len(valid)} valid, {dropped} dropped (< {args.min_words} words)")

    # Truncate all to same length (min word count)
    M = min(r["word_count"] for r in valid)
    print(f"  Truncating all to M = {M} words")

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
    print("  Per-author distribution:")
    for auth, cnt in per_author.items():
        print(f"    {auth}: {cnt}")

    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
