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

import pandas as pd

from src.corpus import load_corpus


THRESHOLDS = [1000, 1500, 2000, 2500, 3000]


def corpus_to_dataframe(source: str, cleaned_dir: str) -> pd.DataFrame:
    corpus = load_corpus(source, cleaned_dir=cleaned_dir)
    return pd.DataFrame({
        "author": corpus.authors,
        "word_count": corpus.word_counts,
        "era": corpus.eras,
        "date": corpus.dates,
    })


def stats_table(df: pd.DataFrame, min_wc: int) -> pd.DataFrame:
    sub = df[df["word_count"] >= min_wc]
    rows = []
    for author, g in sub.groupby("author"):
        pre = int((g["era"] == "pre").sum())
        post = int((g["era"] == "post").sum())
        other = int(len(g) - pre - post)
        rows.append({
            "author": author,
            "total": int(len(g)),
            "pre (<=2022)": pre,
            "post (>=2024)": post,
            "other_year": other,
            "min_pre_post": min(pre, post),
            "wc_min": int(g["word_count"].min()),
            "wc_max": int(g["word_count"].max()),
            "wc_mean": int(round(g["word_count"].mean())),
        })

    # Authors that vanished entirely at this threshold still appear as zeros.
    for author in sorted(df["author"].unique()):
        if author not in {r["author"] for r in rows}:
            rows.append({
                "author": author,
                "total": 0, "pre (<=2022)": 0, "post (>=2024)": 0,
                "other_year": 0, "min_pre_post": 0,
                "wc_min": 0, "wc_max": 0, "wc_mean": 0,
            })

    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["min_pre_post", "total"], ascending=[False, False]
    ).reset_index(drop=True)


def summarize(table: pd.DataFrame, min_wc: int) -> None:
    total = int(table["total"].sum())
    pre = int(table["pre (<=2022)"].sum())
    post = int(table["post (>=2024)"].sum())
    with_both = int((table["min_pre_post"] > 0).sum())

    feas = {N: int((table["min_pre_post"] >= N).sum())
            for N in [5, 10, 15, 20, 25, 30]}

    print(f"\n════ Min length ≥ {min_wc} words ════")
    print(table.to_string(index=False))
    print(f"\n  kept total={total}  pre={pre}  post={post}  "
          f"authors with both pre & post (>=1) = {with_both}")
    print("  feasibility — authors with min(pre, post) >= N:")
    for N, n in feas.items():
        print(f"    N={N:>2d}:  {n} authors")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="lesswrong_regular")
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
