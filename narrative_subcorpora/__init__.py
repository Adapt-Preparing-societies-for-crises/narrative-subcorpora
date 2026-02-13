"""narrative-subcorpora — construct subcorpora of historical text data around events."""

from .corpus import Corpus, Subcorpus
from .event import Event
from .ingest import ingest

__all__ = ["Corpus", "Subcorpus", "Event", "ingest"]
