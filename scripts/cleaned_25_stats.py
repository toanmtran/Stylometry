"""
Stats table for the cleaned_25 dataset.

Columns:
    author | total | pre (<=2022) | post (>=2024) | >= 1526 words | < 1526 words

Output:
    outputs/corpus_stats/lesswrong_large/cleaned_25/stats.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import pandas as pd

from src.corpus import load_corpus

WC_THRESHOLD = 1526


def main() -> None:
    corpus = load_corpus("lesswrong_large", cleaned_dir="cleaned_25")
    df = pd.DataFrame({
        "author": corpus.authors,
        "word_count": corpus.word_counts,
        "era": corpus.eras,
    })

    print(f"Loaded {len(df)} docs across {df['author'].nunique()} authors "
          f"from cleaned_25")

    rows = []
    for author, g in df.groupby("author"):
        rows.append({
            "author": author,
            "total": len(g),
            "pre (<=2022)": int((g["era"] == "pre").sum()),
            "post (>=2024)": int((g["era"] == "post").sum()),
            f">= {WC_THRESHOLD} words": int((g["word_count"] >= WC_THRESHOLD).sum()),
            f"< {WC_THRESHOLD} words": int((g["word_count"] < WC_THRESHOLD).sum()),
        })

    table = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)

    print(table.to_string(index=False))

    out_dir = _PROJECT_ROOT / "outputs" / "corpus_stats" / "lesswrong_large" / "cleaned_25"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "stats.csv"
    table.to_csv(out, index=False)
    print(f"\n-> {out.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
