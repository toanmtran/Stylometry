"""
Remove articles with word_count strictly < 1526 from all files in
dataset/lesswrong_large/cleaned_25/.
"""

import json
from pathlib import Path

THRESHOLD = 1526
DATA_DIR = Path(__file__).parent.parent / "dataset" / "lesswrong_large" / "cleaned_25"


def process_file(path: Path) -> tuple[int, int]:
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)

    before = len(articles)
    kept = [a for a in articles if a.get("word_count", 0) >= THRESHOLD]
    after = len(kept)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    return before, after


def main():
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        print(f"No JSON files found in {DATA_DIR}")
        return

    total_before = total_after = 0
    for path in files:
        before, after = process_file(path)
        removed = before - after
        total_before += before
        total_after += after
        print(f"{path.name:<40} {before:>4} -> {after:>4}  (removed {removed})")

    print(f"\nTotal: {total_before} -> {total_after}  (removed {total_before - total_after})")


if __name__ == "__main__":
    main()
