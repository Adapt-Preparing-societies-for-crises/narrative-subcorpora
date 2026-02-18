"""Diagnostics and visualisation for subcorpus selection."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from .corpus import Corpus, Subcorpus
from .event import Event


def selection_report(
    corpus: Corpus,
    event: Event,
    subcorpus: Subcorpus,
    *,
    months: int = 6,
    output: str | Path | None = None,
) -> plt.Figure:
    """Visualise how many articles were selected vs the full window.

    Produces two panels:
    1. Stacked daily counts — selected vs excluded texts since the event.
    2. Cumulative curves — total corpus in window vs selected over time.

    Parameters
    ----------
    corpus : Corpus
        The full corpus (parquet-backed).
    event : Event
        The event used for windowing.
    subcorpus : Subcorpus
        The filtered subcorpus (after scoring / thresholding).
    months : int
        How many months after the event the window covers.
    output : str or Path, optional
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # --- data prep -------------------------------------------------------
    window_df = corpus.after(event, months=months).to_dataframe()
    selected_df = subcorpus.to_dataframe()

    date_col = corpus.date_col
    window_df[date_col] = pd.to_datetime(window_df[date_col])
    selected_df[date_col] = pd.to_datetime(selected_df[date_col])

    all_daily = (
        window_df.groupby(window_df[date_col].dt.date).size().rename("all")
    )
    sel_daily = (
        selected_df.groupby(selected_df[date_col].dt.date).size().rename("selected")
    )

    daily = pd.DataFrame({"all": all_daily, "selected": sel_daily}).fillna(0).astype(int)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    daily["excluded"] = daily["all"] - daily["selected"]

    # --- figure ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    event_date = pd.Timestamp(event.start_date)

    # Panel 1: daily bar chart
    ax1.bar(daily.index, daily["selected"], width=1, label="Selected", color="#2563eb")
    ax1.bar(
        daily.index,
        daily["excluded"],
        width=1,
        bottom=daily["selected"],
        label="Excluded",
        color="#d1d5db",
    )
    ax1.axvline(event_date, color="#dc2626", linestyle="--", linewidth=1, label="Event start")
    ax1.set_ylabel("Articles per day")
    ax1.set_title(f"{event.full_name} — daily article counts")
    ax1.legend(loc="upper right")

    # Panel 2: cumulative
    ax2.plot(daily.index, daily["all"].cumsum(), label="All in window", color="#6b7280", linewidth=1.5)
    ax2.plot(daily.index, daily["selected"].cumsum(), label="Selected", color="#2563eb", linewidth=2)
    ax2.axvline(event_date, color="#dc2626", linestyle="--", linewidth=1)
    ax2.set_ylabel("Cumulative articles")
    ax2.set_xlabel("Date")
    ax2.set_title("Cumulative selection over time")
    ax2.legend(loc="upper left")

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)

    # Summary annotation
    n_all = len(window_df)
    n_sel = len(selected_df)
    pct = (n_sel / n_all * 100) if n_all else 0
    fig.suptitle(
        f"Selected {n_sel:,} of {n_all:,} articles ({pct:.1f}%)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()

    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")

    return fig


def score_distribution(
    subcorpus: Subcorpus,
    *,
    score_col: str = "score",
    bins: int = 40,
    threshold: float | None = None,
    output: str | Path | None = None,
) -> plt.Figure:
    """Histogram of score values with optional threshold line.

    Parameters
    ----------
    subcorpus : Subcorpus
        A subcorpus that already has a score column.
    score_col : str
        Name of the score column.
    bins : int
        Number of histogram bins.
    threshold : float, optional
        If given, draw a vertical line at this value.
    output : str or Path, optional
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = subcorpus.to_dataframe()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(df[score_col], bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
    if threshold is not None:
        ax.axvline(threshold, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")
        ax.legend()
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of articles")
    ax.set_title("Score distribution")

    fig.tight_layout()
    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
    return fig


def group_score_distribution(
    subcorpus: Subcorpus,
    *,
    group_cols: list[str] | None = None,
    combined_col: str = "score_grouped",
    bins: int = 30,
    threshold: float | None = None,
    output: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side histograms of per-group and combined scores.

    Shows one histogram panel per group score column plus a final panel
    for the combined score.  Call ``.score_grouped()`` first to ensure
    the columns are present.

    Parameters
    ----------
    subcorpus : Subcorpus
        A subcorpus that has been scored with ``.score_grouped()``.
    group_cols : list[str], optional
        Names of the per-group score columns to display (e.g.
        ``["score_location", "score_event_type"]``).  If *None*, all
        columns whose name starts with ``"score_"`` and ends with one of
        the group names inferred from the DataFrame are shown.  In
        practice it is clearer to pass them explicitly.
    combined_col : str
        Name of the combined score column (default ``"score_grouped"``).
    bins : int
        Number of histogram bins per panel.
    threshold : float, optional
        If given, draw a vertical threshold line on the combined-score
        panel.
    output : str or Path, optional
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    ::

        sub = corpus.after(event, months=6).score_grouped(event.term_groups)
        group_score_distribution(
            sub,
            group_cols=["score_location", "score_event_type", "score_impact"],
            threshold=0.05,
        )
    """
    df = subcorpus.to_dataframe()

    # Determine which columns to show
    if group_cols is None:
        # Auto-detect: any score_ column that isn't a known "global" score
        known_global = {"score", "score_tfidf", "score_bm25", "score_cluster",
                        "score_weighted", "score_grouped", "score_outlier",
                        "score_similarity"}
        group_cols = [
            c for c in df.columns
            if c.startswith("score_") and c not in known_global
        ]

    panels = group_cols + ([combined_col] if combined_col in df.columns else [])

    if not panels:
        raise ValueError(
            "No score columns found to plot. Pass group_cols explicitly or "
            "call .score_grouped() first."
        )

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    palette = ["#2563eb"] * (n - 1) + ["#16a34a"]  # blue for groups, green for combined

    for ax, col, color in zip(axes, panels, palette):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.hist(df[col], bins=bins, color=color, edgecolor="white", linewidth=0.5)
        if threshold is not None and col == combined_col:
            ax.axvline(
                threshold, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"threshold={threshold}",
            )
            ax.legend(fontsize=8)
        label = col.replace("score_", "").replace("_", " ").title()
        if col == combined_col:
            label = f"{label} (combined)"
        ax.set_title(label)
        ax.set_xlabel("Score")
        ax.set_ylabel("Articles")

    fig.suptitle("Score distributions by group", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")

    return fig
