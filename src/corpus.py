"""
Load a cleaned corpus from `dataset/<source>/<cleaned_dir>/*.json`.

`cleaned_dir` defaults to `"cleaned"` (the common cleaned corpus written
by scripts/clean_corpus.py). Problem scripts that operate on a further-
processed subset pass their own directory name (e.g. "cleaned_10").

Standard object: `Corpus` — parallel lists of texts, labels, authors,
eras, dates, word_counts. Supports filtering (by era, min_words, author
subset) and equal-length truncation. Each problem script composes these
to build its own view of the data without reaching back into raw files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

_WORD_RE = re.compile(r"\b\w+\b")


@dataclass
class Corpus:
    """A parallel-list corpus representation.

    All list-valued fields have length N (= number of docs) and share
    ordering — element i in every list refers to the same document.
    """
    texts: list[str]
    labels: list[str]
    authors: list[str]
    eras: list[str | None]         # "pre" | "post" | None
    dates: list[str | None]
    word_counts: list[int]
    source: str = ""               # e.g. "lesswrong_regular"
    cleaned_dir: str = ""          # e.g. "cleaned_10" — the on-disk subfolder
    truncated_to: int | None = None  # if equal_length_truncate has been applied

    def __len__(self) -> int:
        return len(self.texts)

    # ── Indexing / row selection ─────────────────────────────────────────

    def select(self, indices: list[int]) -> "Corpus":
        """Return a new Corpus with only the rows at `indices`."""
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

    # ── Filtering ────────────────────────────────────────────────────────

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

    # ── Balanced sampling ────────────────────────────────────────────────

    def balance_per_author_era(
        self, *, n_per_era: int, seed: int = 0,
    ) -> "Corpus":
        """Return a new Corpus with exactly `n_per_era` pre and `n_per_era`
        post docs per author, drawn via a seeded random sample. Raises if
        any author has fewer than `n_per_era` docs in either era. Docs with
        era=None (neither pre nor post) are dropped."""
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

    # ── Length control ───────────────────────────────────────────────────

    def equal_length_truncate(self, *, floor: int | None = None) -> "Corpus":
        """Truncate every text to M = min(word_count). If `floor` is given,
        M = max(floor, min(word_count)) — use this to *require* a minimum
        length after you've pre-filtered."""
        if len(self) == 0:
            return self
        M = min(self.word_counts)
        if floor is not None and M < floor:
            raise ValueError(
                f"equal_length_truncate: floor={floor} but shortest doc has "
                f"only {M} words. Filter by min_words first."
            )
        new_texts = [_truncate_to_words(t, M) for t in self.texts]
        new_wcs = [_word_count(t) for t in new_texts]
        return replace(
            self,
            texts=new_texts,
            word_counts=new_wcs,
            truncated_to=M,
        )

    def truncate_to(self, n: int) -> "Corpus":
        """Truncate every text to exactly `n` words (first n). Raises if any
        doc is shorter than n — pre-filter with `min_words=n` to guarantee
        coverage. Used for controlled length sweeps (effect-of-N experiments)."""
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


# ── Loader ───────────────────────────────────────────────────────────────────


def load_corpus(
    source: str = "lesswrong",
    *,
    cleaned_dir: str = "cleaned",
    dataset_root: Path | None = None,
) -> Corpus:
    """Read every cleaned per-author JSON under `dataset/<source>/<cleaned_dir>/`.

    `cleaned_dir` defaults to `"cleaned"` — the common cleaned corpus.
    Problem scripts working on a further-processed subset pass their own
    directory (e.g. `"cleaned_10"` for the K=10 problem set).

    Each file is a list of dicts with keys: author, text, word_count,
    date, year, time_group, ... (see scripts/clean_corpus.py).

    The returned Corpus is sorted by (author, date, source_index) for
    reproducibility.
    """
    root = dataset_root or Path(__file__).resolve().parents[1] / "dataset"
    folder = root / source / cleaned_dir
    if not folder.exists():
        raise FileNotFoundError(
            f"{folder} does not exist. Run scripts/clean_corpus.py first."
        )

    records: list[dict] = []
    for fp in sorted(folder.glob("*.json")):
        if fp.name.startswith("_"):  # skip _dropped.jsonl and friends
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


# ── helpers ─────────────────────────────────────────────────────────────────


def _label_for(record: dict) -> str:
    """Short stable label: <author>_<YYYY>_<index> (readable in plots)."""
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
