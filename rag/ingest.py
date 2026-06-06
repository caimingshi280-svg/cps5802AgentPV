"""Build or refresh the persisted Chroma index from ``rag/knowledge_base/documents``.

Run::

    python -m rag.ingest

Or with options::

    python -m rag.ingest --reset

Environment / settings follow :mod:`configs.settings` (``AGENTPV_*``).
"""
from __future__ import annotations

import argparse

from configs.settings import Settings, get_settings
from rag.chunking import chunk_directory
from rag.embedding import SentenceTransformerEmbedder

_META_MODEL_KEY = "embedding_model"


def ingest_knowledge_base(
    settings: Settings | None = None,
    *,
    reset: bool = False,
) -> int:
    """Chunk markdown, embed with sentence-transformers, upsert into Chroma.

    Returns the number of vectors written.
    """

    import chromadb

    s = settings or get_settings()
    documents_dir = s.knowledge_base_dir
    chroma_dir = s.chroma_dir
    collection_name = s.chroma_collection_name
    model_name = s.embedding_model

    chunks = chunk_directory(documents_dir)
    if not chunks:
        raise RuntimeError(f"No chunks from {documents_dir}")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:  # noqa: BLE001
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            _META_MODEL_KEY: model_name,
            "hnsw:space": "cosine",
        },
    )

    embedder = SentenceTransformerEmbedder(model_name)
    embedder.fit([])
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    ids = [f"{c.source}__{i}" for i, c in enumerate(chunks)]
    metadatas = [
        {
            "source": c.source,
            "title": c.title,
            "section": c.section or "",
        }
        for c in chunks
    ]
    emb_list = [row.tolist() for row in embeddings]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=emb_list,
        metadatas=metadatas,
    )
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest markdown KB into Chroma.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before re-ingesting.",
    )
    args = parser.parse_args()
    n = ingest_knowledge_base(reset=args.reset)
    print(f"Ingested {n} chunk vectors into Chroma.")


if __name__ == "__main__":
    main()
