"""Corpus and Subcorpus classes — fluent API for querying parquet data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta

from .event import Event
from .score import (
    combined_score,
    tfidf_score,
    bm25_score,
    build_doc_frequencies,
    term_cluster_score,
    weighted_term_score,
)


class Corpus:
    """A parquet-backed text corpus with a fluent query API.

    Parameters
    ----------
    path : str or Path
        Path to a parquet file.
    text_col : str
        Name of the column containing text.
    date_col : str
        Name of the column containing dates.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        text_col: str = "text",
        date_col: str = "date",
    ):
        self.path = Path(path)
        schema = pq.read_schema(self.path)
        columns = schema.names
        for name, col in [("text_col", text_col), ("date_col", date_col)]:
            if col not in columns:
                raise KeyError(
                    f"{name}='{col}' not found. "
                    f"Available columns: {columns}"
                )
        self.text_col = text_col
        self.date_col = date_col

    # ------------------------------------------------------------------
    # Time-window methods — return Subcorpus
    # ------------------------------------------------------------------

    def after(self, event: Event, *, months: int = 6) -> Subcorpus:
        """Texts from *event.start_date* up to *months* months later."""
        start = event.start_date
        end = start + relativedelta(months=months)
        return self._query(start, end)

    def before(self, event: Event, *, months: int = 1) -> Subcorpus:
        """Texts from *months* months before *event.start_date* up to the event."""
        end = event.start_date
        start = end - relativedelta(months=months)
        return self._query(start, end)

    def around(
        self,
        event: Event,
        *,
        months_before: int = 1,
        months_after: int = 6,
    ) -> Subcorpus:
        """Texts in a window around *event.start_date*."""
        start = event.start_date - relativedelta(months=months_before)
        end = event.start_date + relativedelta(months=months_after)
        return self._query(start, end)

    def between(self, start: str | date, end: str | date) -> Subcorpus:
        """Texts between two dates (inclusive)."""
        if isinstance(start, str):
            start = pd.Timestamp(start).date()
        if isinstance(end, str):
            end = pd.Timestamp(end).date()
        return self._query(start, end)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """Return schema and row-count information."""
        pf = pq.read_metadata(self.path)
        schema = pq.read_schema(self.path)
        return {
            "path": str(self.path),
            "num_rows": pf.num_rows,
            "num_columns": schema.names.__len__(),
            "columns": [
                {"name": name, "type": str(schema.field(name).type)}
                for name in schema.names
            ],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _query(self, start: date, end: date) -> Subcorpus:
        con = duckdb.connect()
        df = con.execute(
            f"""
            SELECT *
            FROM read_parquet('{self.path}')
            WHERE CAST("{self.date_col}" AS DATE)
                  BETWEEN '{start}' AND '{end}'
            """
        ).fetchdf()
        con.close()
        return Subcorpus(df, text_col=self.text_col, date_col=self.date_col)


class Subcorpus:
    """A filtered slice of a corpus, supporting scoring and export."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        text_col: str = "text",
        date_col: str = "date",
    ):
        self._df = df
        self.text_col = text_col
        self.date_col = date_col

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        terms: list[str],
        *,
        freq_weight: float = 1.0,
        density_weight: float = 0.0,
        col: str = "score",
    ) -> Subcorpus:
        """Add a relevance score column based on seed *terms*."""
        self._df[col] = self._df[self.text_col].apply(
            lambda t: combined_score(
                str(t),
                terms,
                freq_weight=freq_weight,
                density_weight=density_weight,
            )
        )
        return self

    def score_tfidf(
        self,
        terms: list[str],
        *,
        col: str = "score_tfidf",
    ) -> Subcorpus:
        """Score texts using TF-IDF over seed *terms*.

        Rewards terms that are frequent in a text but rare across the
        subcorpus.
        """
        texts = self._df[self.text_col].astype(str).tolist()
        doc_freq = build_doc_frequencies(texts, terms)
        n_docs = len(texts)
        self._df[col] = [
            tfidf_score(str(t), terms, doc_freq, n_docs) for t in texts
        ]
        return self

    def score_bm25(
        self,
        terms: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        col: str = "score_bm25",
    ) -> Subcorpus:
        """Score texts using BM25 ranking over seed *terms*.

        Standard search-engine relevance formula with length
        normalisation.
        """
        texts = self._df[self.text_col].astype(str).tolist()
        doc_freq = build_doc_frequencies(texts, terms)
        n_docs = len(texts)
        avgdl = sum(len(t.split()) for t in texts) / max(n_docs, 1)
        self._df[col] = [
            bm25_score(str(t), terms, doc_freq, n_docs, avgdl, k1=k1, b=b)
            for t in texts
        ]
        return self

    def score_cluster(
        self,
        terms: list[str],
        *,
        window: int = 50,
        col: str = "score_cluster",
    ) -> Subcorpus:
        """Score texts by how tightly seed *terms* cluster together.

        Slides a word-window over the text and returns the best
        concentration of distinct seed terms found in any single window.
        """
        self._df[col] = self._df[self.text_col].apply(
            lambda t: term_cluster_score(str(t), terms, window=window)
        )
        return self

    def score_weighted(
        self,
        term_weights: dict[str, float],
        *,
        col: str = "score_weighted",
    ) -> Subcorpus:
        """Score texts using per-term importance weights.

        Parameters
        ----------
        term_weights : dict[str, float]
            Mapping from term to its weight. Give rare/specific terms a
            higher weight than common ones.
        """
        self._df[col] = self._df[self.text_col].apply(
            lambda t: weighted_term_score(str(t), term_weights)
        )
        return self

    def embed(
        self,
        model: str = "all-MiniLM-L6-v2",
        *,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> Subcorpus:
        """Compute embeddings for all texts.

        Stores the embedding matrix on the subcorpus for use by
        ``.score_outlier()`` and ``.score_similarity()``.  Requires
        ``sentence-transformers``::

            pip install narrative-subcorpora[embeddings]
        """
        from .embed import embed_texts

        texts = self._df[self.text_col].astype(str).tolist()
        self._embeddings = embed_texts(
            texts, model, batch_size=batch_size, show_progress=show_progress,
        )
        self._embed_model = model
        return self

    def score_outlier(
        self,
        *,
        method: str = "centroid",
        k: int = 5,
        col: str = "score_outlier",
    ) -> Subcorpus:
        """Add an outlier score column based on embeddings.

        Call ``.embed()`` first.  Higher scores = more outlier-like.

        Parameters
        ----------
        method : str
            ``"centroid"`` — cosine distance from subcorpus centroid.
            ``"knn"`` — average cosine distance to *k* nearest neighbours.
        k : int
            Number of neighbours (only used when method is ``"knn"``).
        col : str
            Name of the output column.
        """
        from .embed import centroid_distance_scores, knn_distance_scores

        if not hasattr(self, "_embeddings"):
            raise RuntimeError("Call .embed() before .score_outlier()")
        if method == "knn":
            scores = knn_distance_scores(self._embeddings, k=k)
        else:
            scores = centroid_distance_scores(self._embeddings)
        self._df[col] = scores
        return self

    def score_similarity(
        self,
        terms: list[str],
        model: str | None = None,
        *,
        batch_size: int = 64,
        col: str = "score_similarity",
    ) -> Subcorpus:
        """Score texts by embedding similarity to seed *terms*.

        Embeds the seed terms, averages them, and computes cosine
        similarity to each text embedding.  Call ``.embed()`` first.
        """
        from .embed import embed_texts, seed_similarity_scores, _load_model

        if not hasattr(self, "_embeddings"):
            raise RuntimeError("Call .embed() before .score_similarity()")
        m = model or getattr(self, "_embed_model", "all-MiniLM-L6-v2")
        seed_emb = embed_texts(terms, m, batch_size=batch_size, show_progress=False)
        self._df[col] = seed_similarity_scores(self._embeddings, seed_emb)
        return self

    def above(self, threshold: float, *, col: str = "score") -> Subcorpus:
        """Keep only rows where *col* >= *threshold*."""
        self._df = self._df[self._df[col] >= threshold].reset_index(drop=True)
        return self

    def below(self, threshold: float, *, col: str = "score_outlier") -> Subcorpus:
        """Keep only rows where *col* <= *threshold*.

        Useful for removing outliers: ``sub.below(0.5, col="score_outlier")``.
        """
        self._df = self._df[self._df[col] <= threshold].reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self._df

    def to_parquet(self, path: str | Path) -> Path:
        """Write to a parquet file. Returns the output path."""
        path = Path(path)
        table = pa.Table.from_pandas(self._df)
        pq.write_table(table, path)
        return path

    def to_csv(self, path: str | Path) -> Path:
        """Write to a CSV file. Returns the output path."""
        path = Path(path)
        self._df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"Subcorpus({len(self._df)} texts)"
