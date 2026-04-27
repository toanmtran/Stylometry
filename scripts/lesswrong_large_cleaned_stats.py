"""
Feasibility tables for a cleaned corpus.

For each minimum-length threshold in {1000, 1500, 2000, 2500, 3000} words,
emit a per-author table with:
    total, pre (<=2022), post (>=2024), other_year,
    min(pre, post), wc_min, wc_max, wc_mean.

Rows sort by min(pre, post) desc — that's the bottleneck for balanced
pre/post clustering.

Output:
    outputs/corpus_stats/<source>/<cleaned_dir>/stats_min{1000,1500,2000,2500,3000}.csv
    (plus pretty-printed tables + feasibility counts to stdout)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from scripts.lesswrong_regular_cleaned_stats import (
    corpus_to_dataframe,
    stats_table,
    summarize,
    THRESHOLDS,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="lesswrong_large")
    ap.add_argument("--cleaned-dir", default="cleaned", dest="cleaned_dir")
    args = ap.parse_args()

    out_dir = _PROJECT_ROOT / "outputs" / "corpus_stats" / args.source / args.cleaned_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = corpus_to_dataframe(args.source, args.cleaned_dir)
    print(f"Loaded {len(df)} cleaned docs across {df['author'].nunique()} "
          f"authors from source='{args.source}'")
    print(f"Global word-count range: {df['word_count'].min()}"
          f"..{df['word_count'].max()}  median={int(df['word_count'].median())}")

    for min_wc in THRESHOLDS:
        table = stats_table(df, min_wc)
        out = out_dir / f"stats_min{min_wc}.csv"
        table.to_csv(out, index=False)
        summarize(table, min_wc)
        print(f"  -> {out.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
