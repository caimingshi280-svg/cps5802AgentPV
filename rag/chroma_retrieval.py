"""ChromaDB-backed dense retrieval with externally supplied embeddings.

Indexes built by :mod:`rag.ingest` store normalized vectors; queries use the
same :class:`rag.embedding.SentenceTransformerEmbedder` model recorded in
collection metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.chunking import Chunk
from rag.embedding import SentenceTransformerEmbedder
from rag.retrieval import RetrievedChunk
from utils.logging_config import get_logger

log = get_logger(__name__)

_META_MODEL_KEY = "embedding_model"


def chroma_collection_has_vectors(chroma_dir: Path, collection_name: str) -> bool:
    """Return True when a persisted Chroma collection exists and is non-empty."""

    if not chroma_dir.is_dir():
        return False
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        coll = client.get_collection(name=collection_name)
        return int(coll.count()) > 0
    except Exception:  # noqa: BLE001 — missing collection or corrupt db
        return False


class ChromaRetriever:
    """Query a persisted Chroma index; API mirrors :meth:`rag.retrieval.Retriever.search`."""

    def __init__(
        self,
        *,
        chroma_dir: Path,
        collection_name: str,
        embedding_model: str | None = None,
    ) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection: Any = self._client.get_collection(name=collection_name)
        meta = self._collection.metadata or {}
        model_name = embedding_model or meta.get(_META_MODEL_KEY)
        if not model_name or not isinstance(model_name, str):
            raise ValueError(
                "Chroma collection missing embedding_model metadata; re-run rag ingest."
            )
        self._embedder = SentenceTransformerEmbedder(model_name)
        self._embedder.fit([])
        n = int(self._collection.count())
        log.info(
            "chroma_retriever_ready",
            extra={"collection": collection_name, "n_vectors": n, "model": model_name},
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not query.strip():
            return []
        n_total = int(self._collection.count())
        if n_total == 0:
            return []
        k = min(top_k, n_total)
        q = self._embedder.embed([query])[0].tolist()
        raw = self._collection.query(
            query_embeddings=[q],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        out: list[RetrievedChunk] = []
        for text, meta, dist in zip(docs, metas, dists, strict=True):
            m = meta or {}
            section_raw = m.get("section") or ""
            section = str(section_raw).strip() or None
            chunk = Chunk(
                text=str(text or ""),
                source=str(m.get("source") or "unknown"),
                title=str(m.get("title") or "untitled"),
                section=section,
            )
            # Cosine space: distance = 1 - cosine_similarity for normalized embeddings.
            score = float(1.0 - float(dist)) if dist is not None else 0.0
            out.append(RetrievedChunk(chunk=chunk, score=score))
        return out
