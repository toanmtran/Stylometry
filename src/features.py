"""
Stylometric feature extraction.

Produces 100+ handcrafted features per document covering:
  - word/sentence length statistics
  - vocabulary richness (Yule K, Simpson D, Brunét W, Honoré R, hapax ratios)
  - function-word frequencies (top-50 English)
  - stylistic word categories (hedges, amplifiers, discourse markers, conjunctions)
  - punctuation rates
  - POS-tag distribution (coarse groups)
  - character-level ratios (uppercase, vowels)

Tokenization uses NLTK when available, regex fallback otherwise.
"""

from __future__ import annotations

import math
import re
import string
import warnings
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


# ── Tokenization (NLTK with regex fallback) ──────────────────────────────────

_HAS_NLTK = False
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk import pos_tag
    _HAS_NLTK = True
except Exception:
    def sent_tokenize(text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    def word_tokenize(text):
        return re.findall(r"\b\w+(?:'\w+)?\b", text)
    def pos_tag(tokens):
        return [(t, "NN") for t in tokens]


# ── Lexical resources ────────────────────────────────────────────────────────

FUNCTION_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
]

HEDGE_WORDS = [
    "maybe", "perhaps", "probably", "possibly", "might", "could",
    "somewhat", "rather", "quite", "fairly", "seemingly",
]
AMPLIFIERS = [
    "very", "really", "extremely", "absolutely", "totally",
    "completely", "utterly", "highly", "incredibly", "remarkably",
]
DISCOURSE_MARKERS = [
    "however", "therefore", "moreover", "furthermore",
    "nevertheless", "meanwhile", "nonetheless", "consequently",
    "thus", "hence", "accordingly", "besides", "otherwise",
]
CONJUNCTIONS = [
    "and", "but", "or", "nor", "for", "yet", "so",
    "because", "although", "though", "while", "whereas", "since",
]


# ── Tokenization helpers ─────────────────────────────────────────────────────

def _tokenize(text: str):
    sentences = sent_tokenize(text)
    tokens_raw = word_tokenize(text)
    tokens_alpha = [t.lower() for t in tokens_raw if t.isalpha()]
    return sentences, tokens_raw, tokens_alpha


# ── Vocabulary-richness helpers ──────────────────────────────────────────────

def _yule_k(tokens):
    freq = Counter(tokens)
    freq_spectrum = Counter(freq.values())
    N = len(tokens)
    if N <= 1:
        return 0.0
    M2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    return 10000 * (M2 - N) / (N * N) if N > 0 else 0


def _simpson_d(tokens):
    freq = Counter(tokens)
    N = len(tokens)
    if N <= 1:
        return 0.0
    return sum(n * (n - 1) for n in freq.values()) / (N * (N - 1))


def _hapax_ratio(tokens):
    freq = Counter(tokens)
    return sum(1 for v in freq.values() if v == 1) / len(tokens) if tokens else 0


def _hapax_dis_ratio(tokens):
    freq = Counter(tokens)
    return sum(1 for v in freq.values() if v == 2) / len(tokens) if tokens else 0


def _brunet_w(tokens):
    N, V = len(tokens), len(set(tokens))
    if N <= 1 or V <= 1:
        return 0
    return N ** (V ** -0.172)


def _honore_r(tokens):
    freq = Counter(tokens)
    N, V = len(tokens), len(freq)
    V1 = sum(1 for v in freq.values() if v == 1)
    if V1 == V or N == 0 or V == 0:
        return 0
    return 100 * math.log(N) / (1 - V1 / V)


def _pos_tag_distribution(tokens_alpha):
    if not tokens_alpha:
        return {}
    tagged = pos_tag(tokens_alpha)
    total = len(tagged)
    tag_counts = Counter(tag for _, tag in tagged)
    return {tag: count / total for tag, count in tag_counts.items()}


# ── Main entry points ────────────────────────────────────────────────────────

def extract_features(text: str) -> Dict[str, float]:
    """Extract 100+ stylometric features from a single document."""
    sentences, tokens_raw, tokens = _tokenize(text)
    n_sent = len(sentences) or 1
    n_tok = len(tokens) or 1
    vocab = set(tokens)
    n_vocab = len(vocab) or 1
    freq = Counter(tokens)

    f: Dict[str, float] = {}

    # Basic stats
    f["n_tokens"] = n_tok
    f["n_sentences"] = n_sent
    f["n_vocab"] = n_vocab
    f["type_token_ratio"] = n_vocab / n_tok

    # Word length
    wlens = [len(w) for w in tokens]
    f["avg_word_len"] = np.mean(wlens)
    f["std_word_len"] = np.std(wlens)
    f["median_word_len"] = np.median(wlens)
    f["max_word_len"] = max(wlens) if wlens else 0
    for k_len in [1, 2, 3, 4, 5, 6]:
        f[f"word_len_{k_len}_frac"] = sum(1 for l in wlens if l == k_len) / n_tok

    # Sentence length
    slens = [len(word_tokenize(s)) for s in sentences]
    f["avg_sent_len"] = np.mean(slens)
    f["std_sent_len"] = np.std(slens)
    f["median_sent_len"] = np.median(slens)
    f["max_sent_len"] = max(slens) if slens else 0
    f["min_sent_len"] = min(slens) if slens else 0

    # Vocabulary richness
    f["hapax_ratio"] = _hapax_ratio(tokens)
    f["hapax_dis_ratio"] = _hapax_dis_ratio(tokens)
    f["yule_k"] = _yule_k(tokens)
    f["simpson_d"] = _simpson_d(tokens)
    f["brunet_w"] = _brunet_w(tokens)
    f["honore_r"] = _honore_r(tokens)

    # Function words
    for fw in FUNCTION_WORDS:
        f[f"fw_{fw}"] = freq.get(fw, 0) / n_tok

    # Stylistic word categories
    for name, wordlist in [
        ("hedge", HEDGE_WORDS),
        ("amplifier", AMPLIFIERS),
        ("discourse", DISCOURSE_MARKERS),
        ("conj", CONJUNCTIONS),
    ]:
        f[f"cat_{name}_rate"] = sum(freq.get(w, 0) for w in wordlist) / n_tok

    # Punctuation
    punct_counts = Counter(c for c in text if c in string.punctuation)
    for p, label in [
        (",", "comma"), (".", "period"), (";", "semicolon"),
        (":", "colon"), ("!", "excl"), ("?", "question"),
        ("-", "dash"), ("(", "paren"), ("'", "apost"),
        ('"', "dquote"),
    ]:
        f[f"punct_{label}_rate"] = punct_counts.get(p, 0) / n_tok

    # Contractions — match straight ('), curly (’), and backtick (`) apostrophes.
    # The curly form is what most publishing pipelines emit, and without it the
    # rate collapses to 0 for authors whose text has been through a smart-quotes
    # editor — turning this into a typography signal rather than a linguistic one.
    contraction_pat = re.compile(
        r"\b\w+n['’`]t\b|\b\w+['’`](s|re|ve|ll|d|m)\b", re.I,
    )
    f["contraction_rate"] = len(contraction_pat.findall(text)) / n_tok

    # Sentence starters
    starters = [word_tokenize(s)[0].lower() for s in sentences if word_tokenize(s)]
    starter_freq = Counter(starters)
    f["pronoun_start_rate"] = sum(
        starter_freq.get(p, 0) for p in ["i", "you", "he", "she", "it", "we", "they"]
    ) / n_sent

    # Paragraph length
    n_para = max(1, len([p for p in text.split("\n\n") if p.strip()]))
    f["avg_paragraph_len"] = n_tok / n_para

    # POS tags
    pos_dist = _pos_tag_distribution(tokens)
    pos_groups = {
        "pos_noun": ["NN", "NNS", "NNP", "NNPS"],
        "pos_verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
        "pos_adj": ["JJ", "JJR", "JJS"],
        "pos_adv": ["RB", "RBR", "RBS"],
        "pos_pron": ["PRP", "PRP$", "WP", "WP$"],
        "pos_det": ["DT", "PDT", "WDT"],
        "pos_prep": ["IN"],
        "pos_conj": ["CC"],
        "pos_modal": ["MD"],
    }
    for group_name, tags in pos_groups.items():
        f[group_name] = sum(pos_dist.get(t, 0) for t in tags)

    # Character level
    alpha_chars = [c.lower() for c in text if c.isalpha()]
    n_alpha = len(alpha_chars) or 1
    f["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / (len(text) or 1)
    for vowel in "aeiou":
        f[f"char_{vowel}_rate"] = sum(1 for c in alpha_chars if c == vowel) / n_alpha

    return f


def extract_features_batch(docs, labels=None) -> pd.DataFrame:
    """Extract features for a list of documents; return a DataFrame indexed by label."""
    if labels is None:
        labels = [f"Doc_{i + 1}" for i in range(len(docs))]
    records = []
    for doc, label in zip(docs, labels):
        feats = extract_features(doc)
        feats["_label"] = label
        records.append(feats)
    return pd.DataFrame(records).set_index("_label").fillna(0)
