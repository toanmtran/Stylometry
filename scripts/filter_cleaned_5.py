"""
Remove articles with word_count < 3000 from dataset/lesswrong_regular/cleaned_5/.

Edits each author JSON in-place (overwrites). Prints a summary table after
removal: author name, total articles kept, count with year >= 2024,
count with year <= 2022, and minimum word count across kept articles.

Usage:
    python scripts/filter_cleaned_5.py [--source lesswrong_regular] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

MIN_WORDS = 3000


def _recount(r: dict) -> int:
    # Use split() — the conservative count a human or text editor would see.
    # The stored word_count field uses \b\w+\b which overcounts (splits contractions,
    # hyphenated phrases), so borderline articles can appear shorter than their
    # stored value suggests.
    return len(r.get("text", "").split())


def filter_cleaned_5(source: str, dry_run: bool) -> None:
    cleaned_dir = _PROJECT_ROOT / "dataset" / source / "cleaned_5"
    if not cleaned_dir.exists():
        raise SystemExit(f"Directory not found: {cleaned_dir}")

    author_files = sorted(f for f in cleaned_dir.glob("*.json")
                          if not f.name.startswith("_"))
    if not author_files:
        raise SystemExit(f"No author JSON files found in {cleaned_dir}")

    rows: list[dict] = []
    total_removed = 0

    for fp in author_files:
        records = json.loads(fp.read_text(encoding="utf-8"))

        kept = []
        for r in records:
            live_wc = _recount(r)
            r["word_count"] = live_wc  # keep stored field in sync
            if live_wc >= MIN_WORDS:
                kept.append(r)

        removed = len(records) - len(kept)
        total_removed += removed

        if not dry_run:
            fp.write_text(
                json.dumps(kept, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        wcs   = [r["word_count"] for r in kept]
        years = [r.get("year") for r in kept]

        rows.append({
            "author":       fp.stem,
            "total":        len(kept),
            "removed":      removed,
            "ge_2024":      sum(1 for y in years if y is not None and y >= 2024),
            "le_2022":      sum(1 for y in years if y is not None and y <= 2022),
            "min_wc":       min(wcs) if wcs else 0,
        })

    # ── Print summary table ───────────────────────────────────────────────────
    tag = "  [DRY RUN — no files written]" if dry_run else ""
    print(f"\nfilter_cleaned_5  threshold={MIN_WORDS}w{tag}\n")

    header = f"{'Author':<22}  {'Total':>7}  {'Removed':>7}  {'≥2024':>6}  {'≤2022':>6}  {'Min wc':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['author']:<22}  {r['total']:>7}  {r['removed']:>7}"
            f"  {r['ge_2024']:>6}  {r['le_2022']:>6}  {r['min_wc']:>7}"
        )
    print("-" * len(header))
    kept_total = sum(r["total"] for r in rows)
    print(f"{'TOTAL':<22}  {kept_total:>7}  {total_removed:>7}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="lesswrong_regular",
                    help="Dataset source folder under dataset/")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview removals without writing any files")
    args = ap.parse_args()
    filter_cleaned_5(args.source, args.dry_run)


if __name__ == "__main__":
    main()
