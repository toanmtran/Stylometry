"""
Clean raw per-author JSONs under dataset/<source>/raw/ into
dataset/<source>/cleaned/.

Input files (per author, in raw/):
    <anything>.json — list of post dicts with at least {author, text, …}.

Output files (per author, in cleaned/):
    <author>.json  — list of cleaned post dicts with enriched fields.
    _dropped.jsonl — audit log of every post that was rejected.

Cleaning is currently LessWrong-specific (fenced code, LaTeX, URLs,
list markers, images, HTML tags, footnote markers). If other sources
need different cleaners they can be added via --source switching later.

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
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from src.preprocess import (
    _sanitize_json,
    word_count,
    strip_lesswrong_boilerplate,
)


MIN_WORDS_HARD = 100
MIN_WORDS = 1500
MIN_SENTENCE_DENSITY = 1 / 50

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_FOOTNOTE = re.compile(r"\[\^?\d+\]")


def clean_text_lesswrong(text: str) -> str:
    text = _MD_IMAGE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = strip_lesswrong_boilerplate(text)
    text = _FOOTNOTE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    m = re.match(r"(\d{4})", date_str.strip())
    return int(m.group(1)) if m else None


def clean_source(source: str) -> None:
    raw_dir = _PROJECT_ROOT / "dataset" / source / "raw"
    out_dir = _PROJECT_ROOT / "dataset" / source / "cleaned"
    drop_log = out_dir / "_dropped.jsonl"

    if not raw_dir.exists():
        raise SystemExit(f"Raw dir not found: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe prior cleaned outputs for this source so stale authors don't linger.
    for old in out_dir.glob("*.json"):
        old.unlink()
    if drop_log.exists():
        drop_log.unlink()

    cleaner = clean_text_lesswrong  # only source for now

    by_author: dict[str, list[dict]] = defaultdict(list)
    drops: list[dict] = []
    seen_texts: dict[str, str] = {}

    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        raise SystemExit(f"No JSON files in {raw_dir}")

    total_raw = 0
    for fp in raw_files:
        items = json.loads(_sanitize_json(fp.read_text(encoding="utf-8-sig")))
        stem = fp.stem
        for i, item in enumerate(items, start=1):
            total_raw += 1
            rec_id = f"{stem}::{i:02d}"
            raw_text = item.get("text", "") or ""
            original_wc = word_count(raw_text)

            if not raw_text.strip():
                drops.append({"id": rec_id, "reason": "empty-raw",
                              "wc": 0, "original_wc": 0})
                continue
            if "too many requests" in raw_text.lower():
                drops.append({"id": rec_id, "reason": "error-stub",
                              "wc": 0, "original_wc": original_wc})
                continue

            cleaned = cleaner(raw_text)
            wc = word_count(cleaned)

            if wc < MIN_WORDS_HARD:
                drops.append({"id": rec_id,
                              "reason": f"too-short-hard (<{MIN_WORDS_HARD}w)",
                              "wc": wc, "original_wc": original_wc})
                continue

            if wc < MIN_WORDS:
                drops.append({"id": rec_id,
                              "reason": f"too-short (<{MIN_WORDS}w)",
                              "wc": wc, "original_wc": original_wc})
                continue

            n_enders = sum(cleaned.count(p) for p in ".!?")
            density = n_enders / max(wc, 1)
            if density < MIN_SENTENCE_DENSITY:
                drops.append({
                    "id": rec_id,
                    "reason": (f"low-sentence-density "
                               f"({n_enders} enders / {wc}w = {density:.4f})"),
                    "wc": wc, "original_wc": original_wc,
                })
                continue

            if cleaned in seen_texts:
                drops.append({
                    "id": rec_id,
                    "reason": f"duplicate-of {seen_texts[cleaned]}",
                    "wc": wc, "original_wc": original_wc,
                })
                continue
            seen_texts[cleaned] = rec_id

            author = (item.get("author") or "").strip().lower()
            date = item.get("date")
            by_author[author].append({
                "author": author,
                "text": cleaned,
                "word_count": wc,
                "original_word_count": original_wc,
                "date": date,
                "year": parse_year(date),
                "time_group": item.get("time_group"),
                "category": item.get("category"),
                "platform": item.get("platform"),
                "source_file": stem,
                "source_index": i,
            })

    # Write per-author files using just <author>.json (the cleaned/ path
    # already says what these are — no _cleaned suffix).
    total_kept = 0
    for author, records in sorted(by_author.items()):
        records.sort(key=lambda r: (r.get("date") or "", r["source_index"]))
        out_path = out_dir / f"{author}.json"
        out_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_kept += len(records)
        print(f"  {author:22s}  kept={len(records):3d}  -> {out_path.name}")

    with drop_log.open("w", encoding="utf-8") as f:
        for d in drops:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    by_reason: dict[str, int] = defaultdict(int)
    for d in drops:
        key = d["reason"].split("(")[0].strip().split(" ")[0]
        by_reason[key] += 1

    print(f"\nRaw docs read: {total_raw}")
    print(f"Kept: {total_kept} docs across {len(by_author)} authors")
    print(f"Dropped: {len(drops)}")
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:30s}  {n}")
    print(f"\nDrop audit -> {drop_log.relative_to(_PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="lesswrong_regular",
                    help="Dataset source folder name (under dataset/)")
    args = ap.parse_args()
    clean_source(args.source)


if __name__ == "__main__":
    main()
