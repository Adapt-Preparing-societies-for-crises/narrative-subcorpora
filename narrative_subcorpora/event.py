"""Event definitions loaded from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dateutil.parser import parse as parse_date


@dataclass(frozen=True)
class Event:
    """A historical event with a start date and seed terms."""

    label: str
    full_name: str
    start_date: date
    terms: list[str]
    term_groups: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path, label: str) -> Event:
        """Load a single event by *label* from a JSON file."""
        events = cls.load_all(path)
        for ev in events:
            if ev.label == label:
                return ev
        available = [e.label for e in events]
        raise KeyError(f"Event '{label}' not found. Available: {available}")

    @classmethod
    def load_all(cls, path: str | Path) -> list[Event]:
        """Load every event from a JSON file."""
        path = Path(path)
        with path.open() as f:
            raw = json.load(f)
        return [cls._from_dict(d) for d in raw]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _from_dict(cls, d: dict) -> Event:
        start = parse_date(d["start_date"], dayfirst=True).date()
        raw_groups = d.get("term_groups", {})
        return cls(
            label=d["label"],
            full_name=d.get("full name", d["label"]),
            start_date=start,
            terms=list(d.get("terms", [])),
            term_groups={k: list(v) for k, v in raw_groups.items()},
        )

    def __str__(self) -> str:
        groups_info = f", {len(self.term_groups)} groups" if self.term_groups else ""
        return f"{self.full_name} ({self.start_date}, {len(self.terms)} terms{groups_info})"
