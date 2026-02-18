"""Scoring functions for measuring text relevance to seed terms."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable


# ── Basic scores ──────────────────────────────────────────────────────

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


# ── Advanced keyword-based scores ─────────────────────────────────────

def tfidf_score(
    text: str,
    terms: list[str],
    doc_frequencies: dict[str, int],
    n_docs: int,
) -> float:
    """TF-IDF relevance score using seed terms as the vocabulary.

    For each seed term, computes TF (count in text / total words) times
    IDF (log of total docs / docs containing the term). Returns the sum
    across all seed terms.

    Parameters
    ----------
    text : str
        The text to score.
    terms : list[str]
        Seed terms.
    doc_frequencies : dict[str, int]
        Mapping from each term to the number of documents it appears in.
        Build this once from your corpus with ``build_doc_frequencies``.
    n_docs : int
        Total number of documents in the corpus.
    """
    words = text.lower().split()
    if not words or not terms:
        return 0.0
    word_counts = Counter(words)
    total = len(words)
    score = 0.0
    for t in terms:
        t_lower = t.lower()
        tf = word_counts.get(t_lower, 0) / total
        df = doc_frequencies.get(t_lower, 0)
        idf = math.log((1 + n_docs) / (1 + df)) + 1  # smoothed IDF
        score += tf * idf
    return score


def build_doc_frequencies(texts: list[str], terms: list[str]) -> dict[str, int]:
    """Count how many documents each seed term appears in.

    Use the output as the ``doc_frequencies`` argument to ``tfidf_score``.
    """
    terms_lower = [t.lower() for t in terms]
    df: dict[str, int] = {t: 0 for t in terms_lower}
    for text in texts:
        text_lower = text.lower()
        for t in terms_lower:
            if t in text_lower:
                df[t] += 1
    return df


def bm25_score(
    text: str,
    terms: list[str],
    doc_frequencies: dict[str, int],
    n_docs: int,
    avgdl: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25 relevance score using seed terms as the query.

    A standard information-retrieval ranking function. Higher is more
    relevant.

    Parameters
    ----------
    text : str
        The text to score.
    terms : list[str]
        Seed terms (treated as the query).
    doc_frequencies : dict[str, int]
        Per-term document frequencies (from ``build_doc_frequencies``).
    n_docs : int
        Total number of documents.
    avgdl : float
        Average document length (in words) across the corpus.
    k1 : float
        Term-frequency saturation parameter (default 1.5).
    b : float
        Length-normalisation parameter (default 0.75).
    """
    words = text.lower().split()
    if not words or not terms:
        return 0.0
    word_counts = Counter(words)
    dl = len(words)
    score = 0.0
    for t in terms:
        t_lower = t.lower()
        tf = word_counts.get(t_lower, 0)
        df = doc_frequencies.get(t_lower, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avgdl)
        score += idf * numerator / denominator
    return score


def term_cluster_score(text: str, terms: list[str], *, window: int = 50) -> float:
    """Proximity-based score: how tightly seed terms cluster together.

    Slides a *window*-word window over the text and returns the maximum
    number of distinct seed terms found in any single window, divided by
    the total number of seed terms. A score of 1.0 means every seed term
    appeared within one window.

    Good for detecting passages where multiple event-related terms
    co-occur, even if overall density is low.
    """
    if not terms:
        return 0.0
    words = text.lower().split()
    if not words:
        return 0.0
    terms_lower = set(t.lower() for t in terms)
    max_hits = 0
    for i in range(max(1, len(words) - window + 1)):
        chunk = set(words[i : i + window])
        hits = len(terms_lower & chunk)
        if hits > max_hits:
            max_hits = hits
    return max_hits / len(terms)


def weighted_term_score(text: str, term_weights: dict[str, float]) -> float:
    """Score using per-term importance weights.

    Parameters
    ----------
    text : str
        The text to score.
    term_weights : dict[str, float]
        Mapping from term to its weight/importance.  For example, give
        rare or highly specific terms a higher weight than common ones.

    Returns the sum of weights for each unique term found, divided by the
    sum of all weights.
    """
    if not term_weights:
        return 0.0
    text_lower = text.lower()
    total_weight = sum(term_weights.values())
    if total_weight == 0:
        return 0.0
    hit_weight = sum(
        w for t, w in term_weights.items() if t.lower() in text_lower
    )
    return hit_weight / total_weight


# ── Combined score ────────────────────────────────────────────────────

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


# ── Grouped scoring ───────────────────────────────────────────────────

def group_term_scores(
    text: str,
    term_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Compute term-coverage score for each named group of seed terms.

    Parameters
    ----------
    text : str
        The text to score.
    term_groups : dict[str, list[str]]
        Mapping from group name to list of seed terms.  Typical groups
        are ``"location"``, ``"event_type"``, and ``"impact"``.

    Returns
    -------
    dict[str, float]
        Per-group term-coverage scores in [0, 1].
    """
    return {
        group: term_frequency_score(text, terms)
        for group, terms in term_groups.items()
    }


def combine_group_scores(
    scores: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
    combine: str | Callable[[dict[str, float]], float] = "geometric",
) -> float:
    """Combine per-group scores into a single value.

    Parameters
    ----------
    scores : dict[str, float]
        Per-group scores, typically from ``group_term_scores``.
    weights : dict[str, float], optional
        Per-group weights, used only by the ``"weighted_sum"`` strategy.
        Groups not listed default to weight 1.0.
    combine : str or callable
        How to combine per-group scores:

        - ``"geometric"`` *(default)* — geometric mean.  Requires
          non-zero evidence in *every* group; a single zero collapses
          the combined score to zero.
        - ``"weighted_sum"`` — weighted average.  A strong group can
          compensate for a weak one.
        - ``"min"`` — minimum across groups.  The combined score equals
          the weakest group; strict AND logic.
        - ``"product"`` — product of all scores.  Like geometric but
          without the normalising exponent; degrades faster with many
          groups.

        You can also pass any callable ``f(scores: dict[str, float]) ->
        float`` for custom combination logic.

    Returns
    -------
    float
        Combined score.
    """
    if callable(combine):
        return float(combine(scores))

    if not scores:
        return 0.0

    values = list(scores.values())

    if combine == "geometric":
        if any(v == 0.0 for v in values):
            return 0.0
        return math.exp(sum(math.log(v) for v in values) / len(values))

    elif combine == "weighted_sum":
        w = weights or {}
        total_w = sum(w.get(k, 1.0) for k in scores)
        if total_w == 0:
            return 0.0
        return sum(scores[k] * w.get(k, 1.0) for k in scores) / total_w

    elif combine == "min":
        return min(values)

    elif combine == "product":
        result = 1.0
        for v in values:
            result *= v
        return result

    else:
        raise ValueError(
            f"Unknown combine strategy: {combine!r}. "
            "Use 'geometric', 'weighted_sum', 'min', or 'product', "
            "or pass a callable."
        )
