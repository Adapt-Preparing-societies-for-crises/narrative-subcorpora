"""Embedding-based scoring and outlier detection.

Requires the ``sentence-transformers`` package::

    pip install narrative-subcorpora[embeddings]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _load_model(model: str | SentenceTransformer) -> SentenceTransformer:
    """Accept a model name or an already-loaded model."""
    if isinstance(model, str):
        try:
            from sentence_transformers import SentenceTransformer as ST
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding features. "
                "Install it with: pip install narrative-subcorpora[embeddings]"
            )
        return ST(model)
    return model


def embed_texts(
    texts: list[str],
    model: str | SentenceTransformer,
    *,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts into embeddings.

    Parameters
    ----------
    texts : list[str]
        Texts to embed.
    model : str or SentenceTransformer
        Model name (e.g. ``"all-MiniLM-L6-v2"``) or a pre-loaded model.
    batch_size : int
        Batch size for encoding.
    show_progress : bool
        Show a progress bar during encoding.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(len(texts), embedding_dim)``.
    """
    m = _load_model(model)
    return m.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


# ── Outlier scoring ──────────────────────────────────────────────────


def centroid_distance_scores(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance from each embedding to the centroid.

    Texts far from the centroid are outliers — they are unlike the
    "average" text in the subcorpus. Returns values in [0, 2] where
    0 = identical to centroid, 2 = opposite direction.
    """
    centroid = embeddings.mean(axis=0, keepdims=True)
    # Normalise
    norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_c = np.linalg.norm(centroid, axis=1, keepdims=True)
    norms_e = np.where(norms_e == 0, 1, norms_e)
    norms_c = np.where(norms_c == 0, 1, norms_c)
    cos_sim = (embeddings / norms_e) @ (centroid / norms_c).T
    return (1 - cos_sim.squeeze()).astype(float)


def knn_distance_scores(embeddings: np.ndarray, *, k: int = 5) -> np.ndarray:
    """Average cosine distance to the *k* nearest neighbours.

    Texts in sparse regions (far from their neighbours) are outliers.
    Returns values in [0, 2].
    """
    # Cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    n = len(embeddings)
    k = min(k, n - 1)
    if k <= 0:
        return np.zeros(n)

    # Distance = 1 - similarity, set self-similarity to inf distance
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, np.inf)

    # For each row, take the k smallest distances
    partitioned = np.partition(dist_matrix, k, axis=1)[:, :k]
    return partitioned.mean(axis=1).astype(float)


def seed_similarity_scores(
    embeddings: np.ndarray,
    seed_embedding: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between each text and a seed-term embedding.

    The seed embedding can be produced by embedding the seed terms
    (or a description of the event) and averaging them. Returns values
    in [-1, 1] where 1 = most similar to the seed.
    """
    if seed_embedding.ndim == 1:
        seed_embedding = seed_embedding.reshape(1, -1)
    # Average if multiple seed embeddings
    if seed_embedding.shape[0] > 1:
        seed_embedding = seed_embedding.mean(axis=0, keepdims=True)

    norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_s = np.linalg.norm(seed_embedding, axis=1, keepdims=True)
    norms_e = np.where(norms_e == 0, 1, norms_e)
    norms_s = np.where(norms_s == 0, 1, norms_s)
    return ((embeddings / norms_e) @ (seed_embedding / norms_s).T).squeeze().astype(float)


# ── High-level convenience ────────────────────────────────────────────


def outlier_scores(
    texts: list[str],
    model: str | SentenceTransformer,
    *,
    method: str = "centroid",
    k: int = 5,
    batch_size: int = 64,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed texts and compute outlier scores in one call.

    Parameters
    ----------
    texts : list[str]
        Texts to analyse.
    model : str or SentenceTransformer
        Model name or pre-loaded model.
    method : str
        ``"centroid"`` (distance from centroid) or ``"knn"`` (average
        distance to *k* nearest neighbours).
    k : int
        Number of neighbours for the knn method.
    batch_size : int
        Batch size for encoding.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    embeddings : np.ndarray
        The computed embeddings (shape ``(n, dim)``).
    scores : np.ndarray
        Outlier scores (shape ``(n,)``). Higher = more outlier-like.
    """
    embeddings = embed_texts(texts, model, batch_size=batch_size, show_progress=show_progress)
    if method == "knn":
        scores = knn_distance_scores(embeddings, k=k)
    else:
        scores = centroid_distance_scores(embeddings)
    return embeddings, scores
