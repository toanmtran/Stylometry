"""
Dataset preprocessing helpers for the stylometry demos.

Two concerns:

1. Boilerplate. Aeon articles embed newsletter pitches, a donation footer,
   and a `byAuthor+BIO … Edited by…` header that repeats across every
   article from the same author — massive same-author text overlap that
   inflates stylometric similarity for trivial reasons. We strip those
   patterns. Guardian articles are substantially cleaner; we only
   normalise whitespace.

2. Length. Features like yule_k, honoré_r, hapax_ratio and absolute
   vocabulary counts vary with document length. Truncating every doc to
   the same first-M-words prefix removes that confound.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ── Typography canonicalisation ──────────────────────────────────────────────
#
# Different publishing pipelines emit different quote glyphs. Straight vs curly
# apostrophes (' vs ’) and straight vs curly double quotes (" vs “ ”) are a
# typography choice made by the CMS, not by the author — but downstream they
# produce completely different tokens and char n-grams, so "don't" and "don’t"
# look like different words. Without this normalisation, features like
# `contraction_rate` and char n-grams such as "n't"/"n’t" end up fingerprinting
# the publishing pipeline instead of measuring style.

_TYPOGRAPHY_MAP = str.maketrans({
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK (curly apostrophe)
    "`": "'",  # GRAVE ACCENT — rarely used as apostrophe
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
})


def canonicalize_typography(text: str) -> str:
    """Map typographic quotes/apostrophes to their ASCII equivalents."""
    return text.translate(_TYPOGRAPHY_MAP)


# ── Word counting ────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b\w+\b")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def truncate_to_words(text: str, n: int) -> str:
    """Return the prefix of `text` containing the first `n` word tokens."""
    count = 0
    for m in _WORD_RE.finditer(text):
        count += 1
        if count == n:
            return text[: m.end()]
    return text


# ── Aeon cleanup ─────────────────────────────────────────────────────────────

# The footer always starts with this sentence (often duplicated). Anything
# after it is donation/related-articles/copyright boilerplate.
_AEON_FOOTER_START = re.compile(
    r"We publish hard-won knowledge from real people", re.IGNORECASE
)

# Inline newsletter pitch. Appears verbatim, often duplicated back-to-back.
_AEON_NEWSLETTER = re.compile(
    r"Join more than [\d,]+\s*newsletter subscribers.*?unsubscribe anytime\.\s*",
    re.IGNORECASE | re.DOTALL,
)

# The `byAuthor Name+BIO` marker that introduces the bio paragraph.
# Allow commas/ampersands for multi-author articles.
_AEON_BYLINE_MARKER = re.compile(r"\bby[A-Z][\w'’\-.,& ]*?\+BIO\b\s*")

# "Edited byEditor Name"
_AEON_EDITED_BY = re.compile(
    r"\bEdited by[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b\s*"
)


def strip_aeon_boilerplate(text: str) -> str:
    text = _AEON_FOOTER_START.split(text, maxsplit=1)[0]
    text = _AEON_NEWSLETTER.sub(" ", text)
    text = _AEON_BYLINE_MARKER.sub(" ", text)
    text = _AEON_EDITED_BY.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_aeon_byline(text: str) -> str | None:
    """Return the name(s) in the `byName+BIO` marker, or None."""
    m = re.match(r"\s*by([A-Z][\w'’\-., & ]*?)\+BIO", text)
    return m.group(1).strip() if m else None


# ── Guardian cleanup ─────────────────────────────────────────────────────────


def strip_guardian_boilerplate(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── LessWrong cleanup ────────────────────────────────────────────────────────
#
# LessWrong posts frequently embed fenced code, inline code, LaTeX math, URLs,
# and bullet/numbered list markers. These break NLTK's sentence tokenizer
# (which in turn makes avg_sent_len explode to 100+ words on ~3% of docs) and
# inject topic-specific character sequences that leak into any char-n-gram
# features. We strip all of them and collapse whitespace. Callers should use
# `min_valid_words >= 1000` so posts that were mostly code/math get dropped
# rather than pulling the truncation length down for everyone else.

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


# ── Loading ──────────────────────────────────────────────────────────────────


def _sanitize_json(raw: str) -> str:
    """Escape stray control chars inside JSON strings; strip trailing commas."""
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
    declared_author: str  # from JSON `author` field
    text: str             # raw, before cleaning


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


# ── Pipeline ─────────────────────────────────────────────────────────────────


@dataclass
class PreparedDataset:
    docs: list[str]            # cleaned + truncated text
    labels: list[str]
    authors: list[str]         # declared author (lowercase)
    min_words: int             # M used for truncation
    dropped: list[tuple[str, str, int]]  # (label, reason, word_count)
    byline_mismatches: list[tuple[str, str, str]]  # (label, declared, actual)


def prepare(
    folder: Path,
    *,
    cleaner: Callable[[str], str],
    byline_extractor: Callable[[str], str | None] | None = None,
    min_valid_words: int = 100,
    min_sentence_density: float | None = None,
    label_prefix_len: int = 12,
    stem_transform: Callable[[str], str] | None = None,
) -> PreparedDataset:
    """
    Load, clean, filter, and truncate a dataset folder.

    Steps:
      1. Load every per-author JSON file.
      2. Clean each text with `cleaner` (removes dataset boilerplate).
      3. Drop docs shorter than `min_valid_words` after cleaning (error
         pages, scraping stubs).
      4. If `min_sentence_density` is set, drop docs whose ratio of
         sentence-ending punctuation (. ! ?) to word count falls below
         it. Catches pasted transcripts or run-on content that isn't
         really prose. Value is sentences-per-word; 1/50 = 0.02 means
         "at least one end-punctuation per 50 words" (normal prose runs
         ~1 per 15-25 words).
      5. Drop exact-duplicate texts that appeared more than once in the
         dataset (same scraped article re-indexed).
      6. M := min word count across surviving docs. Truncate all to M.
      7. Collect byline-vs-declared-author mismatches for reporting.
    """
    raw = load_folder(folder)
    cleaned: list[tuple[LoadedDoc, str, int]] = []
    dropped: list[tuple[str, str, int]] = []
    seen_texts: dict[str, str] = {}  # normalized text -> first label to keep

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

        # Drop exact duplicates (keep first occurrence).
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
    byline_mismatches: list[tuple[str, str, str]] = []
    for d, text, _ in cleaned:
        label = f"{d.file_stem[:label_prefix_len]}_{d.index:02d}"
        docs.append(truncate_to_words(text, M))
        labels.append(label)
        authors.append(d.declared_author)

        if byline_extractor is not None:
            actual = byline_extractor(d.text)
            if actual and actual.lower() != d.declared_author.lower():
                byline_mismatches.append((label, d.declared_author, actual))

    return PreparedDataset(
        docs=docs,
        labels=labels,
        authors=authors,
        min_words=M,
        dropped=dropped,
        byline_mismatches=byline_mismatches,
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

    if ds.byline_mismatches:
        print(
            f"  ⚠ {len(ds.byline_mismatches)} byline/filename mismatches "
            "(label left as declared; investigate source data):"
        )
        for label, declared, actual in ds.byline_mismatches:
            print(f"    - {label}: file says '{declared}', byline says '{actual}'")
