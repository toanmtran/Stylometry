"""Per-corpus boilerplate stripping, typography normalisation, and length truncation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_TYPOGRAPHY_MAP = str.maketrans({
    "‘": "'",
    "’": "'",
    "`": "'",
    "“": '"',
    "”": '"',
})


def canonicalize_typography(text: str) -> str:
    """Map curly quotes and apostrophes to their ASCII equivalents."""
    return text.translate(_TYPOGRAPHY_MAP)


_WORD_RE = re.compile(r"\b\w+\b")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def truncate_to_words(text: str, n: int) -> str:
    """Return the prefix of text containing the first n word tokens."""
    count = 0
    for m in _WORD_RE.finditer(text):
        count += 1
        if count == n:
            return text[: m.end()]
    return text


_LW_CODE_FENCE  = re.compile(r"```.*?```", re.DOTALL)
_LW_INLINE_CODE = re.compile(r"`[^`\n]+`")
_LW_MATH_BLOCK  = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
_LW_MATH_INLINE = re.compile(r"\$[^$\n]+\$|\\\([^)]+\\\)")
_LW_URL         = re.compile(r"https?://\S+|www\.\S+")
_LW_LIST_MARKER = re.compile(r"^\s*[\-\*•]\s+|^\s*\d+\.\s+", re.MULTILINE)


def strip_lesswrong_boilerplate(text: str) -> str:
    text = _LW_CODE_FENCE.sub(" ", text)
    text = _LW_MATH_BLOCK.sub(" ", text)
    text = _LW_INLINE_CODE.sub(" ", text)
    text = _LW_MATH_INLINE.sub(" ", text)
    text = _LW_URL.sub(" ", text)
    text = _LW_LIST_MARKER.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sanitize_json(raw: str) -> str:
    """Escape stray control chars inside JSON strings and strip trailing commas."""
    cleaned, in_string, escape_next = [], False, False
    for ch in raw:
        if escape_next:
            cleaned.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            cleaned.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            cleaned.append(ch)
            continue
        if in_string and ch in ("\n", "\r", "\t"):
            cleaned.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            continue
        cleaned.append(ch)
    return re.sub(r",\s*([}\]])", r"\1", "".join(cleaned))


@dataclass
class LoadedDoc:
    file_stem: str
    index: int
    declared_author: str
    text: str


def load_folder(folder: Path) -> list[LoadedDoc]:
    out: list[LoadedDoc] = []
    for fp in sorted(folder.glob("*.json")):
        items = json.loads(_sanitize_json(fp.read_text(encoding="utf-8-sig")))
        for i, item in enumerate(items, start=1):
            out.append(
                LoadedDoc(
                    file_stem=fp.stem,
                    index=i,
                    declared_author=item["author"].strip().lower(),
                    text=item["text"],
                )
            )
    return out


@dataclass
class PreparedDataset:
    docs: list[str]
    labels: list[str]
    authors: list[str]
    min_words: int
    dropped: list[tuple[str, str, int]]


def prepare(
    folder: Path,
    *,
    cleaner: Callable[[str], str],
    min_valid_words: int = 100,
    min_sentence_density: float | None = None,
    label_prefix_len: int = 12,
    stem_transform: Callable[[str], str] | None = None,
) -> PreparedDataset:
    """Load, clean, filter, and equal-length-truncate every doc in a folder."""
    raw = load_folder(folder)
    cleaned: list[tuple[LoadedDoc, str, int]] = []
    dropped: list[tuple[str, str, int]] = []
    seen_texts: dict[str, str] = {}

    for d in raw:
        label = f"{d.file_stem[:label_prefix_len]}_{d.index:02d}"
        cleaned_text = cleaner(d.text)
        wc = word_count(cleaned_text)

        if wc < min_valid_words:
            reason = (
                "error-stub"
                if "too many requests" in d.text.lower()
                else f"too-short (<{min_valid_words}w)"
            )
            dropped.append((label, reason, wc))
            continue

        if min_sentence_density is not None:
            n_enders = sum(cleaned_text.count(p) for p in ".!?")
            density = n_enders / max(wc, 1)
            if density < min_sentence_density:
                dropped.append(
                    (label,
                     f"low-sentence-density ({n_enders} enders / {wc}w "
                     f"= {density:.4f} < {min_sentence_density:.4f})",
                     wc)
                )
                continue

        key = cleaned_text
        if key in seen_texts:
            dropped.append((label, f"duplicate-of:{seen_texts[key]}", wc))
            continue
        seen_texts[key] = label

        cleaned.append((d, cleaned_text, wc))

    if not cleaned:
        raise RuntimeError(f"No valid documents in {folder}")

    M = min(wc for _, _, wc in cleaned)

    docs, labels, authors = [], [], []
    for d, text, _ in cleaned:
        label = f"{d.file_stem[:label_prefix_len]}_{d.index:02d}"
        docs.append(truncate_to_words(text, M))
        labels.append(label)
        authors.append(d.declared_author)

    return PreparedDataset(
        docs=docs,
        labels=labels,
        authors=authors,
        min_words=M,
        dropped=dropped,
    )


def print_report(ds: PreparedDataset, name: str) -> None:
    print(f"\n── Preprocessing report: {name} ──")
    print(f"  Surviving documents: {len(ds.docs)}")
    print(f"  Unique declared authors: {len(set(ds.authors))}")
    print(f"  Truncated to first M = {ds.min_words} words")

    if ds.dropped:
        print(f"  Dropped {len(ds.dropped)} docs:")
        for label, reason, wc in ds.dropped:
            print(f"    - {label}  [{reason}, {wc}w]")
