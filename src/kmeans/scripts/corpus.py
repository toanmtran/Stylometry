"""Per-author JSON corpus loader and filtering helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

_WORD_RE = re.compile(r"\b\w+\b")


@dataclass
class Corpus:
    """Parallel-list corpus: one entry per document across all list fields."""
    texts: list[str]
    labels: list[str]
    authors: list[str]
    eras: list[str | None]
    dates: list[str | None]
    word_counts: list[int]
    source: str = ""
    cleaned_dir: str = ""
    truncated_to: int | None = None

    def __len__(self) -> int:
        return len(self.texts)

    def select(self, indices: list[int]) -> "Corpus":
        """Return a new Corpus restricted to the given row indices."""
        return Corpus(
            texts=[self.texts[i] for i in indices],
            labels=[self.labels[i] for i in indices],
            authors=[self.authors[i] for i in indices],
            eras=[self.eras[i] for i in indices],
            dates=[self.dates[i] for i in indices],
            word_counts=[self.word_counts[i] for i in indices],
            source=self.source,
            cleaned_dir=self.cleaned_dir,
            truncated_to=self.truncated_to,
        )

    def filter(
        self,
        *,
        era: str | None = None,
        min_words: int | None = None,
        authors: list[str] | None = None,
    ) -> "Corpus":
        """Return a new Corpus keeping only docs matching all given constraints."""
        keep: list[int] = []
        authors_set = set(authors) if authors is not None else None
        for i in range(len(self)):
            if era is not None and self.eras[i] != era:
                continue
            if min_words is not None and self.word_counts[i] < min_words:
                continue
            if authors_set is not None and self.authors[i] not in authors_set:
                continue
            keep.append(i)
        return self.select(keep)

    def balance_per_author_era(
        self, *, n_per_era: int, seed: int = 0,
    ) -> "Corpus":
        """Sample exactly n_per_era pre and n_per_era post docs per author."""
        import random
        rng = random.Random(seed)

        by_author_era: dict[tuple[str, str], list[int]] = {}
        for i, (a, e) in enumerate(zip(self.authors, self.eras)):
            if e not in ("pre", "post"):
                continue
            by_author_era.setdefault((a, e), []).append(i)

        keep: list[int] = []
        for (author, era), rows in sorted(by_author_era.items()):
            if len(rows) < n_per_era:
                raise ValueError(
                    f"balance_per_author_era: {author}/{era} has only "
                    f"{len(rows)} docs, need {n_per_era}."
                )
            keep.extend(rng.sample(rows, n_per_era))

        keep.sort()
        return self.select(keep)

    def truncate_to(self, n: int) -> "Corpus":
        """Truncate every text to exactly n words."""
        if n <= 0:
            raise ValueError(f"truncate_to: n must be positive, got {n}")
        if len(self) == 0:
            return self
        shortest = min(self.word_counts)
        if shortest < n:
            raise ValueError(
                f"truncate_to: n={n} but shortest doc has only {shortest} "
                "words. Filter by min_words first."
            )
        new_texts = [_truncate_to_words(t, n) for t in self.texts]
        new_wcs = [_word_count(t) for t in new_texts]
        return replace(
            self,
            texts=new_texts,
            word_counts=new_wcs,
            truncated_to=n,
        )


def load_corpus(
    source: str,
    *,
    cleaned_dir: str,
    dataset_root: Path,
) -> Corpus:
    """Load all per-author JSON files under dataset_root/source/cleaned_dir/."""
    folder = dataset_root / source / cleaned_dir
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist.")

    records: list[dict] = []
    for fp in sorted(folder.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        for item in json.loads(fp.read_text(encoding="utf-8")):
            records.append(item)

    records.sort(key=lambda r: (
        r.get("author", ""),
        r.get("date") or "",
        r.get("source_index", 0),
    ))

    return Corpus(
        texts=[r["text"] for r in records],
        labels=[_label_for(r) for r in records],
        authors=[r.get("author", "") for r in records],
        eras=[_era_for(r.get("year")) for r in records],
        dates=[r.get("date") for r in records],
        word_counts=[int(r.get("word_count", 0)) for r in records],
        source=source,
        cleaned_dir=cleaned_dir,
    )


def _label_for(record: dict) -> str:
    author = record.get("author", "unknown")
    year = record.get("year") or "xxxx"
    idx = record.get("source_index", 0)
    return f"{author[:12]}_{year}_{idx:02d}"


def _era_for(year: int | None) -> str | None:
    if year is None:
        return None
    if year <= 2022:
        return "pre"
    if year >= 2024:
        return "post"
    return None


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _truncate_to_words(text: str, n: int) -> str:
    count = 0
    for m in _WORD_RE.finditer(text):
        count += 1
        if count == n:
            return text[: m.end()]
    return text
