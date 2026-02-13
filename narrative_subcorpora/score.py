"""Scoring functions for measuring text relevance to seed terms."""

from __future__ import annotations

import re


def term_frequency_score(text: str, terms: list[str]) -> float:
    """Fraction of unique seed terms found in *text*.

    Returns a value in [0, 1] — the proportion of seed terms that appear
    at least once.
    """
    if not terms:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for t in terms if t.lower() in text_lower)
    return hits / len(terms)


def term_density_score(text: str, terms: list[str]) -> float:
    """Total seed-term hits divided by the number of words in *text*.

    Returns a value in [0, ∞) — higher means more concentrated.
    """
    words = text.split()
    if not words or not terms:
        return 0.0
    text_lower = text.lower()
    hits = sum(
        len(re.findall(re.escape(t.lower()), text_lower))
        for t in terms
    )
    return hits / len(words)


def combined_score(
    text: str,
    terms: list[str],
    *,
    freq_weight: float = 1.0,
    density_weight: float = 0.0,
) -> float:
    """Weighted average of term-frequency and term-density scores."""
    total_weight = freq_weight + density_weight
    if total_weight == 0:
        return 0.0
    freq = term_frequency_score(text, terms)
    density = term_density_score(text, terms)
    return (freq * freq_weight + density * density_weight) / total_weight
