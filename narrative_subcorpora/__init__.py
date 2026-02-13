"""narrative-subcorpora — construct subcorpora of historical text data around events."""

from .corpus import Corpus, Subcorpus
from .event import Event
from .ingest import ingest
from .diagnostics import selection_report, score_distribution

__all__ = [
    "Corpus",
    "Subcorpus",
    "Event",
    "ingest",
    "selection_report",
    "score_distribution",
]
