"""CLI entry point — the ``nsc`` command."""

from __future__ import annotations

from pathlib import Path

import click

from .corpus import Corpus
from .event import Event
from .ingest import ingest


@click.group()
def cli():
    """nsc — narrative-subcorpora toolkit."""


# ── ingest ────────────────────────────────────────────────────────────

@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, help="Output parquet path.")
@click.option("--text-col", default="text", help="Name of the text column.")
@click.option("--date-col", default="date", help="Name of the date column.")
@click.option("--clean", is_flag=True, help="Apply basic text cleaning.")
@click.option("--language", default="nl", help="Language for cleaning.")
def ingest_cmd(source, output, text_col, date_col, clean, language):
    """Ingest tabular data (CSV/TSV/Excel) into a parquet file."""
    out = ingest(
        source,
        output,
        text_col=text_col,
        date_col=date_col,
        do_clean=clean,
        language=language,
    )
    click.echo(f"Written to {out}")


# ── events ────────────────────────────────────────────────────────────

@cli.command()
@click.argument("path", type=click.Path(exists=True))
def events(path):
    """List events defined in a JSON file."""
    evts = Event.load_all(path)
    for ev in evts:
        click.echo(f"  {ev.label:20s} {ev.full_name} ({ev.start_date})")


# ── describe ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("path", type=click.Path(exists=True))
def describe(path):
    """Describe a parquet file (schema, row count)."""
    info = Corpus(path).describe()
    click.echo(f"Path:    {info['path']}")
    click.echo(f"Rows:    {info['num_rows']}")
    click.echo(f"Columns: {info['num_columns']}")
    for col in info["columns"]:
        click.echo(f"  {col['name']:30s} {col['type']}")


# ── extract ───────────────────────────────────────────────────────────

def _parse_window(window: str) -> int:
    """Parse a window string like '6m' into months."""
    window = window.strip().lower()
    if window.endswith("m"):
        return int(window[:-1])
    return int(window)


@cli.command()
@click.option("--corpus", required=True, type=click.Path(exists=True), help="Parquet corpus file.")
@click.option("--events", "events_path", required=True, type=click.Path(exists=True), help="Events JSON file.")
@click.option("--event", "event_label", required=True, help="Event label to extract.")
@click.option("--window", default="6m", help="Time window after event (e.g. 6m).")
@click.option("--min-score", default=0.0, type=float, help="Minimum score threshold.")
@click.option("--text-col", default="text", help="Name of the text column.")
@click.option("--date-col", default="date", help="Name of the date column.")
@click.option("-o", "--output", required=True, help="Output path (.parquet or .csv).")
def extract(corpus, events_path, event_label, window, min_score, text_col, date_col, output):
    """Extract a subcorpus around an event."""
    c = Corpus(corpus, text_col=text_col, date_col=date_col)
    ev = Event.from_json(events_path, event_label)
    months = _parse_window(window)

    sub = c.after(ev, months=months).score(terms=ev.terms)

    if min_score > 0:
        sub = sub.above(min_score)

    out = Path(output)
    if out.suffix == ".csv":
        sub.to_csv(out)
    else:
        sub.to_parquet(out)

    click.echo(f"Extracted {len(sub)} texts → {out}")
