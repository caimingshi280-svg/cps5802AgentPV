"""Embedding backends for RAG.

MVP backend: :class:`TfidfEmbedder` — pure scikit-learn, deterministic,
no network or model download. Sufficient for the assignment's 5 / 30
document scale and for unit testing.

Polish-phase backend (not implemented in MVP, see ``rag/README.md``):
``SentenceTransformerEmbedder`` wrapping ``BAAI/bge-small-en-v1.5``.
Both backends conform to the :class:`Embedder` protocol so the retriever
does not care which one is wired in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(ABC):
    """Embedder protocol — fit on a corpus, then embed any text."""

    name: str
    dim: int

    @abstractmethod
    def fit(self, texts: Iterable[str]) -> Embedder:
        """Fit the embedder on a corpus. Return self for chaining."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n, dim)`` matrix of float32 embeddings."""


class TfidfEmbedder(Embedder):
    """TF-IDF embedder backed by :class:`sklearn.feature_extraction.text.TfidfVectorizer`.

    The vector is L2-normalized so cosine similarity reduces to a dot product.
    Vocabulary is fitted from the corpus passed to :meth:`fit`; text seen at
    query time but not at fit time still gets a valid sparse-to-dense vector
    (zero on unknown tokens).
    """

    name = "tfidf"

    def __init__(
        self,
        *,
        max_features: int = 8192,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            stop_words="english",
            norm="l2",
        )
        self.dim = -1
        self._fitted = False

    def fit(self, texts: Iterable[str]) -> TfidfEmbedder:
        corpus = [t for t in texts if t.strip()]
        if not corpus:
            raise ValueError("TfidfEmbedder.fit received empty corpus")
        self._vectorizer.fit(corpus)
        self.dim = len(self._vectorizer.vocabulary_)
        self._fitted = True
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.embed called before fit()")
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        sparse = self._vectorizer.transform(list(texts))
        return sparse.toarray().astype(np.float32, copy=False)


class SentenceTransformerEmbedder(Embedder):
    """Dense embeddings via ``sentence-transformers`` (lazy model load).

    :meth:`fit` is a no-op beyond ensuring the model is loaded; vectors are
    L2-normalized at encode time so cosine similarity equals the dot product.
    """

    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self.dim = -1

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, texts: Iterable[str]) -> SentenceTransformerEmbedder:
        del texts  # corpus not required for ST inference
        self._load()
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self._model.encode(  # type: ignore[union-attr]
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)
