"""Tool: retrieve_knowledge — query the RAG knowledge base.

Wraps :class:`rag.retrieval.Retriever` in the project's :class:`Tool`
contract: typed I/O, timeout, structured ToolError on failure (rule §11).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from configs.settings import Settings

from rag.chunking import chunk_directory
from rag.chroma_retrieval import ChromaRetriever, chroma_collection_has_vectors
from rag.embedding import TfidfEmbedder
from rag.reranking import IdentityReranker, Reranker
from rag.retrieval import RetrievedChunk, Retriever
from tools.base import Tool


class RetrieveKnowledgeInput(BaseModel):
    """Inputs for the retrieve_knowledge tool."""

    query: str = Field(min_length=1, description="Natural-language search query.")
    top_k: int = Field(default=3, ge=1, le=20)


class RetrievedDocSummary(BaseModel):
    """One ranked retrieval result, flattened for prompt consumption."""

    title: str
    source: str
    section: str | None
    score: float
    text: str


class RetrieveKnowledgeOutput(BaseModel):
    """Output: ranked retrieval results."""

    query: str
    docs: list[RetrievedDocSummary]
    source_titles: list[str] = Field(
        description="Distinct titles of cited documents (for Recommendation.knowledge_sources)."
    )


def _summary_from_retrieved(chunk: RetrievedChunk) -> RetrievedDocSummary:
    return RetrievedDocSummary(
        title=chunk.chunk.title,
        source=chunk.chunk.source,
        section=chunk.chunk.section,
        score=round(chunk.score, 4),
        text=chunk.chunk.text,
    )


class RetrieveKnowledgeTool(Tool[RetrieveKnowledgeInput, RetrieveKnowledgeOutput]):
    """Tool implementation backed by a pre-built dense retriever."""

    name = "retrieve_knowledge"
    description = "Retrieve top-k relevant documentation chunks for a fault query."
    input_model = RetrieveKnowledgeInput
    output_model = RetrieveKnowledgeOutput
    timeout_s = 5.0

    def __init__(
        self,
        retriever: Retriever | ChromaRetriever,
        reranker: Reranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker or IdentityReranker()

    async def _run(self, inp: RetrieveKnowledgeInput) -> RetrieveKnowledgeOutput:
        retrieved = self.retriever.search(inp.query, top_k=max(inp.top_k * 2, inp.top_k))
        # Rerank then truncate (currently identity; polish 升级到 cross-encoder)
        ranked = self.reranker.rerank(inp.query, retrieved, top_k=inp.top_k)
        docs = [_summary_from_retrieved(rc) for rc in ranked]
        # 去重保留首次出现的文档标题，作为 Recommendation.knowledge_sources。
        seen: set[str] = set()
        source_titles: list[str] = []
        for d in docs:
            if d.title not in seen:
                seen.add(d.title)
                source_titles.append(d.title)
        return RetrieveKnowledgeOutput(
            query=inp.query, docs=docs, source_titles=source_titles
        )


def build_default_tool(
    documents_dir: Path,
    *,
    settings: Settings | None = None,
) -> RetrieveKnowledgeTool:
    """Construct a RetrieveKnowledgeTool from TF-IDF or persisted Chroma."""

    from configs.settings import get_settings

    s = settings or get_settings()
    mode = (s.rag_retrieval or "auto").strip().lower()
    want_chroma = mode == "chroma" or (
        mode == "auto"
        and chroma_collection_has_vectors(s.chroma_dir, s.chroma_collection_name)
    )
    if want_chroma and mode != "tfidf":
        try:
            cr = ChromaRetriever(
                chroma_dir=s.chroma_dir,
                collection_name=s.chroma_collection_name,
            )
            return RetrieveKnowledgeTool(retriever=cr)
        except Exception as exc:  # noqa: BLE001
            from utils.logging_config import get_logger

            get_logger(__name__).warning(
                "retrieve_knowledge_chroma_unavailable",
                extra={"error": str(exc)},
            )

    chunks = chunk_directory(documents_dir)
    if not chunks:
        raise RuntimeError(f"Knowledge base at {documents_dir} produced no chunks")
    retriever = Retriever(chunks=chunks, embedder=TfidfEmbedder())
    return RetrieveKnowledgeTool(retriever=retriever)


def __dir__() -> list[str]:  # pragma: no cover - introspection helper
    return [
        "RetrieveKnowledgeTool",
        "RetrieveKnowledgeInput",
        "RetrieveKnowledgeOutput",
        "RetrievedDocSummary",
        "build_default_tool",
    ]


# Backwards-compat alias for orchestration code that wants the type name.
RetrieveKnowledge = RetrieveKnowledgeTool
_: Any = None  # silence unused-import warnings if any
