"""In-memory dense retriever for AgentPV's RAG layer.

MVP scope (rule §27):

* Loads chunks via :func:`rag.chunking.chunk_directory`.
* Embeds them with the supplied :class:`rag.embedding.Embedder`.
* On query, returns the top-k chunks by cosine similarity.

The polish-phase upgrade swaps the in-memory matrix for ChromaDB without
changing the :class:`Retriever` interface or the :class:`RetrievedChunk`
shape consumed by tools and prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk, chunk_directory
from rag.embedding import Embedder, TfidfEmbedder
from utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk plus its retrieval score, returned to tools / prompts."""

    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "source": self.chunk.source,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "text": self.chunk.text,
        }


class Retriever:
    """Dense retriever over a fixed collection of chunks."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        if not chunks:
            raise ValueError("Retriever requires at least one chunk")
        self.chunks = chunks
        self.embedder: Embedder = embedder or TfidfEmbedder()
        # 用 chunk 文本拟合词表 + 计算 chunk 嵌入矩阵。
        self.embedder.fit([c.text for c in chunks])
        matrix = self.embedder.embed([c.text for c in chunks])
        self._matrix = self._l2_normalize(matrix)
        log.info(
            "retriever_built",
            extra={
                "n_chunks": len(chunks),
                "dim": self._matrix.shape[1],
                "embedder": self.embedder.name,
            },
        )

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize rows so that cosine sim = dot product."""

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (matrix / norms).astype(np.float32, copy=False)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top-k chunks ranked by cosine similarity to ``query``."""

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not query.strip():
            return []
        q_vec = self._l2_normalize(self.embedder.embed([query]))[0]
        scores = self._matrix @ q_vec
        # 部分排序快于全排序——chunks 数小（5–100）时差异忽略，但保持习惯。
        idx = np.argsort(-scores)[: min(top_k, len(self.chunks))]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in idx]


def build_retriever_from_dir(
    documents_dir: Path,
    *,
    max_chars: int = 1200,
    embedder: Embedder | None = None,
) -> Retriever:
    """Convenience: chunk every ``*.md`` in a dir and build a Retriever."""

    chunks = chunk_directory(documents_dir, max_chars=max_chars)
    if not chunks:
        raise ValueError(f"No chunks produced from {documents_dir}")
    return Retriever(chunks=chunks, embedder=embedder)
