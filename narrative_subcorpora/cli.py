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
        groups_info = f"  [{', '.join(ev.term_groups)}]" if ev.term_groups else ""
        click.echo(f"  {ev.label:20s} {ev.full_name} ({ev.start_date}){groups_info}")


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
@click.option(
    "--grouped", is_flag=True, default=False,
    help="Use grouped scoring (requires term_groups in the event definition).",
)
@click.option(
    "--combine",
    default="geometric",
    type=click.Choice(["geometric", "weighted_sum", "min", "product"]),
    help="Combination strategy for grouped scoring (default: geometric).",
)
@click.option("-o", "--output", required=True, help="Output path (.parquet or .csv).")
def extract(corpus, events_path, event_label, window, min_score, text_col, date_col,
            grouped, combine, output):
    """Extract a subcorpus around an event.

    By default uses flat term-coverage scoring.  Pass --grouped to use
    per-group scoring with the groups defined in the event's JSON entry.
    """
    c = Corpus(corpus, text_col=text_col, date_col=date_col)
    ev = Event.from_json(events_path, event_label)
    months = _parse_window(window)

    if grouped:
        if not ev.term_groups:
            raise click.UsageError(
                f"Event '{ev.label}' has no term_groups defined in the JSON file. "
                "Add a 'term_groups' entry or use flat scoring (omit --grouped)."
            )
        sub = c.after(ev, months=months).score_grouped(
            ev.term_groups, combine=combine
        )
        score_col = "score_grouped"
    else:
        sub = c.after(ev, months=months).score(terms=ev.terms)
        score_col = "score"

    if min_score > 0:
        sub = sub.above(min_score, col=score_col)

    out = Path(output)
    if out.suffix == ".csv":
        sub.to_csv(out)
    else:
        sub.to_parquet(out)

    click.echo(f"Extracted {len(sub)} texts → {out}")


# ── diagnose ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--corpus", required=True, type=click.Path(exists=True), help="Parquet corpus file.")
@click.option("--events", "events_path", required=True, type=click.Path(exists=True), help="Events JSON file.")
@click.option("--event", "event_label", required=True, help="Event label.")
@click.option("--window", default="6m", help="Time window after event (e.g. 6m).")
@click.option("--min-score", default=0.0, type=float, help="Score threshold for selection.")
@click.option("--text-col", default="text", help="Name of the text column.")
@click.option("--date-col", default="date", help="Name of the date column.")
@click.option("-o", "--output", default=None, help="Save figure to file (png, pdf, svg).")
def diagnose(corpus, events_path, event_label, window, min_score, text_col, date_col, output):
    """Visualise selection diagnostics for an event subcorpus."""
    import matplotlib
    if output:
        matplotlib.use("Agg")
    else:
        # Try common interactive backends, fall back to Agg + file save
        for backend in ("TkAgg", "QtAgg", "Agg"):
            try:
                matplotlib.use(backend)
                break
            except ImportError:
                continue

    from .diagnostics import selection_report

    c = Corpus(corpus, text_col=text_col, date_col=date_col)
    ev = Event.from_json(events_path, event_label)
    months = _parse_window(window)

    sub = c.after(ev, months=months).score(terms=ev.terms)
    filtered = sub.above(min_score) if min_score > 0 else sub

    if output:
        selection_report(c, ev, filtered, months=months, output=output)
        click.echo(f"Saved figure to {output}")
    else:
        import matplotlib.pyplot as plt
        selection_report(c, ev, filtered, months=months)
        if matplotlib.get_backend().lower() == "agg":
            # No interactive backend available, save to file instead
            fallback = "diagnose_report.png"
            plt.savefig(fallback, dpi=150, bbox_inches="tight")
            click.echo(f"No interactive display available. Saved to {fallback}")
        else:
            plt.show()
