"""Ingest tabular data (CSV / TSV / Excel) into parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_tabular(path: str | Path) -> pd.DataFrame:
    """Read a CSV, TSV, or Excel file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(path)
    # Default: CSV (handles .csv and anything else)
    return pd.read_csv(path)


def clean_text(text: str, *, language: str = "nl") -> str:
    """Basic text cleaning: collapse whitespace, strip."""
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())  # collapse whitespace
    return text.strip()


def ingest(
    source: str | Path,
    output: str | Path,
    *,
    text_col: str = "text",
    date_col: str = "date",
    do_clean: bool = False,
    language: str = "nl",
) -> Path:
    """Read tabular data, optionally clean text, and write to parquet.

    Returns the output path.
    """
    df = read_tabular(source)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

    if do_clean and text_col in df.columns:
        df[text_col] = df[text_col].apply(
            lambda t: clean_text(t, language=language)
        )

    output = Path(output)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output)
    return output
