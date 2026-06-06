"""Rerank retrieved chunks before prompting (MVP placeholder).

The MVP backend is the :class:`IdentityReranker`: it returns the chunks
in the order produced by the retriever. This is intentional — TF-IDF
cosine ordering is already reasonable for the small corpus, and adding a
rerank model in MVP would bloat the dependency tree.

Polish phase: implement :class:`CrossEncoderReranker` using
``BAAI/bge-reranker-base`` and swap it in via ``configs/settings.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from rag.retrieval import RetrievedChunk


class Reranker(ABC):
    """Reranker protocol — input list goes in, ranked list comes out."""

    name: str

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return chunks reordered by relevance to ``query``."""


class IdentityReranker(Reranker):
    """No-op reranker: returns the input order unchanged.

    This exists so ReAct / RAG callers can always call ``reranker.rerank(...)``
    without a feature toggle. Tests verify it preserves order and respects
    ``top_k``.
    """

    name = "identity"

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        del query  # MVP doesn't use the query for reranking
        if top_k is None:
            return list(chunks)
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        return list(chunks)[:top_k]
