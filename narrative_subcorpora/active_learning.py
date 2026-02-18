"""Active learning for subcorpus refinement.

Wraps a :class:`~narrative_subcorpora.corpus.Subcorpus` and iteratively
improves text selection by asking the researcher to label a small batch of
documents, then retraining a classifier on those labels to rank remaining
documents.

Requires the ``active-learning`` optional dependencies::

    pip install narrative-subcorpora[active-learning]

Quick start::

    from narrative_subcorpora import Corpus, Event, ActiveLearner

    corpus = Corpus("news.parquet", text_col="ocr", date_col="date")
    event  = Event.from_json("events.json", "watersnood")
    sub    = corpus.after(event, months=6).score(event.terms).score_grouped(event.term_groups)

    al = ActiveLearner(sub, features="scores")
    al.annotate(n=10)         # interactive labelling (ipywidgets or text fallback)
    al.status()               # show progress
    result = al.to_subcorpus()  # Subcorpus with score_al column
"""

from __future__ import annotations

import random
import textwrap
from typing import Callable

import numpy as np
import pandas as pd

from .corpus import Subcorpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCORE_COLS = frozenset(
    {
        "score",
        "score_tfidf",
        "score_bm25",
        "score_cluster",
        "score_weighted",
        "score_grouped",
        "score_similarity",
        "score_outlier",
    }
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation without scipy."""
    n = len(a)
    if n < 2:
        return float("nan")
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    d2 = np.sum((rank_a - rank_b) ** 2)
    return float(1.0 - 6.0 * d2 / (n * (n**2 - 1)))


# ---------------------------------------------------------------------------
# ActiveLearner
# ---------------------------------------------------------------------------


class ActiveLearner:
    """Iterative relevance learner for a :class:`Subcorpus`.

    The learner maintains a binary classifier (logistic regression) that is
    retrained each time new labels arrive.  Unlabelled documents are ranked
    by *uncertainty* — the ones the model is most unsure about are shown
    next, which is the fastest way to improve a small labelled set.

    Parameters
    ----------
    subcorpus : Subcorpus
        The candidate pool of texts to label.
    features : str
        How to build the feature matrix:

        * ``"scores"`` *(default)* — use existing score columns already on
          the subcorpus (e.g. ``score``, ``score_grouped``, …).  Fast and
          requires no additional computation.
        * ``"tfidf"`` — build a TF-IDF matrix over the raw text.  Slower
          but does not depend on pre-computed scores.
        * ``"embeddings"`` — use the embedding matrix stored by
          :meth:`~narrative_subcorpora.corpus.Subcorpus.embed`.  Highest
          quality; call ``.embed()`` on the subcorpus first.

    seed_terms : list[str], optional
        When ``features="tfidf"``, restrict the TF-IDF vocabulary to these
        terms (plus a small number of high-frequency corpus terms).  If
        *None*, the full vocabulary is used.
    cold_start_col : str, optional
        Column to use for cold-start ranking (before any labels exist).
        Defaults to the first ``score_*`` column found, then ``"score"``.
    score_col : str
        Name of the output column written by :meth:`to_subcorpus`.
        Default: ``"score_al"``.
    random_state : int
        Random seed for reproducibility.

    Examples
    --------
    Simulated labelling (e.g. in a script or test)::

        al = ActiveLearner(sub, features="scores")
        al.label_batch({0: True, 1: False, 3: True})
        al.retrain()
        result = al.to_subcorpus()

    Interactive labelling in a Jupyter notebook::

        al = ActiveLearner(sub, features="scores")
        al.annotate(n=10)           # shows ipywidgets cards or text prompts
        al.status()
        result = al.to_subcorpus()
    """

    def __init__(
        self,
        subcorpus: Subcorpus,
        *,
        features: str = "scores",
        seed_terms: list[str] | None = None,
        cold_start_col: str | None = None,
        score_col: str = "score_al",
        random_state: int = 42,
    ):
        if features not in ("scores", "tfidf", "embeddings"):
            raise ValueError(
                f"features must be 'scores', 'tfidf', or 'embeddings', got {features!r}"
            )
        self._sub = subcorpus
        self._df = subcorpus._df.reset_index(drop=True)
        self._features = features
        self._seed_terms = seed_terms
        self._score_col = score_col
        self._rng = random.Random(random_state)
        self._np_rng = np.random.default_rng(random_state)

        # Labels: index -> True (relevant) / False (irrelevant)
        self._labels: dict[int, bool] = {}

        # Classifier and current probability estimates
        self._clf = None
        self._al_scores: np.ndarray = self._cold_start_scores(cold_start_col)
        self._prev_scores: np.ndarray | None = None  # for stability tracking

        # Build feature matrix (lazy for embeddings)
        self._X: np.ndarray | None = None
        if features != "embeddings":
            self._X = self._build_features()

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    def _build_features(self) -> np.ndarray:
        if self._features == "scores":
            return self._scores_features()
        if self._features == "tfidf":
            return self._tfidf_features()
        # embeddings — deferred until first retrain
        return self._embedding_features()

    def _scores_features(self) -> np.ndarray:
        """Use all numeric score columns as features."""
        score_cols = [
            c for c in self._df.columns
            if (c in _SCORE_COLS or (c.startswith("score_") and c not in _SCORE_COLS))
            and pd.api.types.is_numeric_dtype(self._df[c])
        ]
        if not score_cols:
            raise RuntimeError(
                "No score columns found.  Call .score() / .score_grouped() on "
                "the subcorpus before using features='scores', or switch to "
                "features='tfidf'."
            )
        X = self._df[score_cols].fillna(0.0).to_numpy(dtype=float)
        return X

    def _tfidf_features(self) -> np.ndarray:
        """TF-IDF over the text column, optionally restricted to seed_terms."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = self._df[self._sub.text_col].astype(str).tolist()
        kwargs: dict = dict(
            sublinear_tf=True,
            min_df=2,
            max_features=20_000,
        )
        if self._seed_terms:
            # Add seed terms explicitly so they are never excluded
            kwargs["vocabulary"] = {t: i for i, t in enumerate(self._seed_terms)}
        vec = TfidfVectorizer(**kwargs)
        X = vec.fit_transform(texts)
        self._tfidf_vectorizer = vec
        return X.toarray()

    def _embedding_features(self) -> np.ndarray:
        """Return the embedding matrix stored by .embed()."""
        if not hasattr(self._sub, "_embeddings"):
            raise RuntimeError(
                "Call .embed() on the subcorpus before using features='embeddings'."
            )
        return np.asarray(self._sub._embeddings, dtype=float)

    # ------------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------------

    def _cold_start_scores(self, col: str | None) -> np.ndarray:
        """Return an initial ranking (before any labels exist)."""
        df = self._df
        if col and col in df.columns:
            return df[col].fillna(0.0).to_numpy(dtype=float)

        # Auto-detect: prefer score_grouped, then score, then any score_ col
        for candidate in ("score_grouped", "score", "score_tfidf", "score_bm25"):
            if candidate in df.columns:
                return df[candidate].fillna(0.0).to_numpy(dtype=float)

        score_cols = [c for c in df.columns if c.startswith("score_")]
        if score_cols:
            return df[score_cols[0]].fillna(0.0).to_numpy(dtype=float)

        # No score columns at all — uniform
        return np.ones(len(df), dtype=float)

    # ------------------------------------------------------------------
    # Labelling
    # ------------------------------------------------------------------

    def label(self, idx: int, relevant: bool) -> ActiveLearner:
        """Label a single document.

        Parameters
        ----------
        idx : int
            Row index in the subcorpus DataFrame.
        relevant : bool
            ``True`` if the document is relevant to the event, ``False``
            otherwise.
        """
        self._labels[int(idx)] = relevant
        return self

    def label_batch(self, labels: dict[int, bool]) -> ActiveLearner:
        """Label multiple documents at once.

        Parameters
        ----------
        labels : dict[int, bool]
            Mapping from row index to label.
        """
        for idx, rel in labels.items():
            self._labels[int(idx)] = rel
        return self

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def retrain(self) -> ActiveLearner:
        """Fit the classifier on all current labels and update scores.

        Must have at least one positive *and* one negative label.

        Returns
        -------
        ActiveLearner
            Returns *self* for chaining.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        if not self._labels:
            raise RuntimeError("No labels yet. Call .label() or .annotate() first.")

        labels_arr = list(self._labels.items())
        idxs = [i for i, _ in labels_arr]
        ys = np.array([1 if r else 0 for _, r in labels_arr])

        if len(set(ys)) < 2:
            print(
                "[ActiveLearner] Need both positive and negative labels to train. "
                f"Currently have only {'positive' if ys[0] == 1 else 'negative'} labels."
            )
            return self

        # Build features if not yet done (embeddings deferred)
        if self._X is None:
            self._X = self._embedding_features()

        X_train = self._X[idxs]
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        )
        clf.fit(X_train, ys)
        self._clf = clf

        # Score all documents
        self._prev_scores = self._al_scores.copy()
        proba = clf.predict_proba(self._X)
        pos_class = list(clf.classes_).index(1) if hasattr(clf, "classes_") else 1
        # predict_proba is on the pipeline's final estimator
        lr = clf.named_steps["logisticregression"]
        pos_idx = list(lr.classes_).index(1)
        self._al_scores = proba[:, pos_idx]
        return self

    # ------------------------------------------------------------------
    # Query strategy
    # ------------------------------------------------------------------

    def next_batch(self, n: int = 10) -> list[int]:
        """Return indices of the next *n* documents to label.

        Before any labels exist (cold start), returns a mix of top-scored
        and random documents.  After training, uses uncertainty sampling
        (documents where the model is least confident).

        Parameters
        ----------
        n : int
            Number of documents to return.

        Returns
        -------
        list[int]
            Row indices into the subcorpus DataFrame.
        """
        unlabelled = [i for i in range(len(self._df)) if i not in self._labels]
        if not unlabelled:
            return []
        n = min(n, len(unlabelled))

        if self._clf is None:
            # Cold start: 2/3 top-scored, 1/3 random
            n_top = max(1, int(n * 2 / 3))
            n_rand = n - n_top
            ranked = sorted(unlabelled, key=lambda i: -self._al_scores[i])
            top = ranked[:n_top]
            rest = ranked[n_top:]
            rand = self._rng.sample(rest, min(n_rand, len(rest)))
            batch = top + rand
        else:
            # Uncertainty sampling: |prob - 0.5| is smallest = most uncertain
            uncertainty = np.abs(self._al_scores[unlabelled] - 0.5)
            order = np.argsort(uncertainty)[:n]
            batch = [unlabelled[i] for i in order]

        return batch

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> None:
        """Print a summary of labelling progress and model quality."""
        n_pos = sum(1 for v in self._labels.values() if v)
        n_neg = sum(1 for v in self._labels.values() if not v)
        n_total = len(self._df)
        print(f"Labelled: {len(self._labels)} / {n_total}  "
              f"(+{n_pos} relevant, -{n_neg} irrelevant)")

        if self._clf is not None and self._prev_scores is not None:
            stability = _spearman(self._prev_scores, self._al_scores)
            print(f"Rank stability (Spearman vs previous fit): {stability:.3f}")

        if self._clf is not None and len(self._labels) >= 4:
            from sklearn.model_selection import cross_val_score

            labels_arr = list(self._labels.items())
            idxs = [i for i, _ in labels_arr]
            ys = np.array([1 if r else 0 for _, r in labels_arr])
            min_class = min(np.bincount(ys))
            if len(set(ys)) >= 2 and min_class >= 2:
                cv = min(3, int(min_class))
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scores = cross_val_score(
                            self._clf, self._X[idxs], ys,
                            cv=cv, scoring="f1", error_score=0.0,
                        )
                    print(f"CV F1 (k={cv}): {scores.mean():.3f} ± {scores.std():.3f}")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Interactive annotation
    # ------------------------------------------------------------------

    def annotate(
        self,
        n: int = 10,
        *,
        auto_retrain: bool = True,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """Interactively label the next *n* documents.

        In a Jupyter notebook with ipywidgets available, shows a card
        interface with **Relevant**, **Not relevant**, and **Skip** buttons.
        Falls back to a plain ``input()``-based loop otherwise.

        Parameters
        ----------
        n : int
            Number of documents to present.
        auto_retrain : bool
            Whether to retrain the classifier automatically after the
            batch is labelled.
        on_done : callable, optional
            Called with no arguments when the batch is finished.
        """
        batch = self.next_batch(n)
        if not batch:
            print("[ActiveLearner] No unlabelled documents remaining.")
            return

        try:
            import ipywidgets as _w
            from IPython.display import display as _display
            _display  # silence unused-import warning
            in_jupyter = True
        except ImportError:
            in_jupyter = False

        def _finish(collected: dict[int, bool]) -> None:
            self.label_batch(collected)
            if auto_retrain:
                pos = sum(v for v in self._labels.values())
                neg = sum(not v for v in self._labels.values())
                if pos > 0 and neg > 0:
                    self.retrain()
                    print(f"[ActiveLearner] Retrained on {len(self._labels)} labels.")
                else:
                    print(
                        "[ActiveLearner] Need both positive and negative labels "
                        "before retraining."
                    )
            if on_done:
                on_done()

        if in_jupyter:
            self._annotate_widgets(batch, _finish)
        else:
            self._annotate_text(batch, _finish)

    def _annotate_widgets(
        self,
        batch: list[int],
        on_done: Callable[[dict[int, bool]], None],
    ) -> None:
        """ipywidgets card interface."""
        import ipywidgets as w
        from IPython.display import display, clear_output

        df = self._df
        text_col = self._sub.text_col
        collected: dict[int, bool] = {}
        remaining = list(batch)

        output = w.Output()
        progress = w.IntProgress(
            value=0, min=0, max=len(batch),
            description="Progress:", style={"description_width": "initial"},
        )

        def show_card() -> None:
            if not remaining:
                with output:
                    clear_output(wait=True)
                    print("[ActiveLearner] All done — batch complete.")
                on_done(collected)
                return

            idx = remaining[0]
            text = str(df.at[idx, text_col])
            preview = textwrap.fill(text[:600], width=80)
            if len(text) > 600:
                preview += " …"

            score_info = ""
            for sc in ("score_grouped", "score", "score_al"):
                if sc in df.columns:
                    score_info = f"  [{sc}: {df.at[idx, sc]:.3f}]"
                    break

            btn_rel = w.Button(
                description="Relevant",
                button_style="success",
                layout=w.Layout(width="140px"),
            )
            btn_irr = w.Button(
                description="Not relevant",
                button_style="danger",
                layout=w.Layout(width="140px"),
            )
            btn_skip = w.Button(
                description="Skip",
                button_style="warning",
                layout=w.Layout(width="100px"),
            )
            btn_stop = w.Button(
                description="Stop",
                button_style="",
                layout=w.Layout(width="100px"),
            )

            header = w.HTML(
                f"<b>Document {len(collected)+1} / {len(batch)}"
                f"  (row {idx}){score_info}</b>"
            )
            body = w.Textarea(
                value=preview,
                layout=w.Layout(width="100%", height="160px"),
                disabled=True,
            )
            buttons = w.HBox([btn_rel, btn_irr, btn_skip, btn_stop])
            card = w.VBox([header, body, buttons])

            def on_relevant(_: w.Button) -> None:
                collected[idx] = True
                remaining.pop(0)
                progress.value += 1
                show_card()

            def on_irrelevant(_: w.Button) -> None:
                collected[idx] = False
                remaining.pop(0)
                progress.value += 1
                show_card()

            def on_skip(_: w.Button) -> None:
                remaining.pop(0)
                progress.value += 1
                show_card()

            def on_stop(_: w.Button) -> None:
                remaining.clear()
                with output:
                    clear_output(wait=True)
                    print("[ActiveLearner] Stopped early.")
                on_done(collected)

            btn_rel.on_click(on_relevant)
            btn_irr.on_click(on_irrelevant)
            btn_skip.on_click(on_skip)
            btn_stop.on_click(on_stop)

            with output:
                clear_output(wait=True)
                display(card)

        display(progress)
        display(output)
        show_card()

    def _annotate_text(
        self,
        batch: list[int],
        on_done: Callable[[dict[int, bool]], None],
    ) -> None:
        """Plain input()-based annotation loop (non-Jupyter fallback)."""
        df = self._df
        text_col = self._sub.text_col
        collected: dict[int, bool] = {}

        print(f"[ActiveLearner] Labelling {len(batch)} documents.")
        print("Enter: y = relevant, n = not relevant, s = skip, q = quit\n")

        for pos, idx in enumerate(batch, 1):
            text = str(df.at[idx, text_col])
            preview = text[:400] + (" …" if len(text) > 400 else "")
            print(f"── Document {pos}/{len(batch)}  (row {idx}) ──")
            print(textwrap.fill(preview, width=78))
            print()
            while True:
                ans = input("Label [y/n/s/q]: ").strip().lower()
                if ans == "y":
                    collected[idx] = True
                    break
                elif ans == "n":
                    collected[idx] = False
                    break
                elif ans == "s":
                    break
                elif ans == "q":
                    print("[ActiveLearner] Quit.")
                    on_done(collected)
                    return
                else:
                    print("  Please enter y, n, s, or q.")
            print()

        on_done(collected)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_subcorpus(self) -> Subcorpus:
        """Return a :class:`Subcorpus` with the active-learning scores.

        The returned subcorpus has an extra ``score_al`` column (or
        whatever name was set via *score_col*) containing the classifier's
        estimated probability of relevance for every document.

        If the classifier has not been trained yet, the column contains the
        cold-start ranking values instead.

        Returns
        -------
        Subcorpus
        """
        df = self._df.copy()
        df[self._score_col] = self._al_scores
        return Subcorpus(df, text_col=self._sub.text_col, date_col=self._sub.date_col)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        trained = "trained" if self._clf is not None else "untrained"
        n_pos = sum(v for v in self._labels.values())
        n_neg = sum(not v for v in self._labels.values())
        return (
            f"ActiveLearner("
            f"{len(self._df)} docs, "
            f"features={self._features!r}, "
            f"{trained}, "
            f"labels={len(self._labels)} (+{n_pos}/-{n_neg})"
            f")"
        )
