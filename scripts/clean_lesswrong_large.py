"""
Clean raw per-author JSONs under dataset/lesswrong_large/raw/ into
dataset/lesswrong_large/cleaned/.

Input files (per author, in raw/):
    lesswrong_<author>_data.json — list of post dicts with {author, text,
                                   category, platform, date}.

Output files (per author, in cleaned/):
    <author>.json  — list of cleaned post dicts with enriched fields.
    _dropped.jsonl — audit log of every post that was rejected.

The lesswrong_large format is identical to lesswrong_regular (same keys,
no time_group field) so all the same cleaning rules apply.

Hard drops (recorded in _dropped.jsonl, not written to cleaned/):
  - empty raw text
  - error-stub ("too many requests")
  - too-short hard floor (<100w after cleaning)
  - below minimum length (<1500w after cleaning)
  - low sentence-density (<1 of .!? per 50 words) — transcripts
  - exact duplicate text (first occurrence kept)
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

from scripts.clean_lesswrong_regular import clean_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="lesswrong_large",
                    help="Dataset source folder name (under dataset/)")
    args = ap.parse_args()
    clean_source(args.source)


if __name__ == "__main__":
    main()
