"""Unit tests for :mod:`rag.embedding`, :mod:`rag.retrieval`, :mod:`rag.reranking`, :mod:`rag.prompting`."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from rag.chunking import Chunk
from rag.embedding import TfidfEmbedder
from rag.prompting import PromptBuilder
from rag.reranking import IdentityReranker
from rag.retrieval import RetrievedChunk, Retriever, build_retriever_from_dir

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_tfidf_fit_and_embed_shapes() -> None:
    emb = TfidfEmbedder()
    emb.fit(["alpha beta", "beta gamma", "delta"])
    out = emb.embed(["alpha", "beta gamma"])
    assert out.shape[0] == 2
    assert out.shape[1] == emb.dim
    assert out.dtype == np.float32


def test_tfidf_embed_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        TfidfEmbedder().embed(["alpha"])


def test_tfidf_fit_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        TfidfEmbedder().fit([])


def test_tfidf_l2_norm_implicit() -> None:
    """sklearn's TfidfVectorizer with norm='l2' returns unit vectors per row."""
    emb = TfidfEmbedder()
    emb.fit(["alpha beta gamma delta", "epsilon zeta eta theta"])
    out = emb.embed(["alpha beta", "delta"])
    norms = np.linalg.norm(out, axis=1)
    assert all(0.0 < n <= 1.0 + 1e-6 for n in norms)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(
            text="Inverter fault diagnosis requires P_ac collapse to zero while P_dc holds.",
            source="inverter.md",
            title="Inverter Fault",
            section="Diagnosis",
        ),
        Chunk(
            text="Partial shading produces a multi-step IV curve under normal irradiance.",
            source="shading.md",
            title="Partial Shading",
            section="Symptoms",
        ),
        Chunk(
            text="Battery thermal runaway is preceded by elevated cell temperature.",
            source="thermal.md",
            title="Thermal Anomaly",
            section="Indicators",
        ),
    ]


def test_retriever_returns_topk_in_score_order() -> None:
    r = Retriever(_make_chunks())
    out = r.search("inverter P_ac collapse", top_k=2)
    assert len(out) == 2
    assert out[0].chunk.title == "Inverter Fault"
    # Scores must be monotonically non-increasing.
    assert out[0].score >= out[1].score


def test_retriever_score_is_finite_and_bounded() -> None:
    r = Retriever(_make_chunks())
    out = r.search("partial shading", top_k=3)
    for rc in out:
        assert isinstance(rc, RetrievedChunk)
        assert -1.0 - 1e-6 <= rc.score <= 1.0 + 1e-6


def test_retriever_empty_query_returns_empty() -> None:
    r = Retriever(_make_chunks())
    assert r.search("", top_k=5) == []
    assert r.search("    ", top_k=5) == []


def test_retriever_topk_zero_raises() -> None:
    r = Retriever(_make_chunks())
    with pytest.raises(ValueError, match="positive"):
        r.search("anything", top_k=0)


def test_retriever_topk_larger_than_corpus_returns_all() -> None:
    r = Retriever(_make_chunks())
    out = r.search("battery thermal", top_k=99)
    assert len(out) == 3


def test_retriever_requires_at_least_one_chunk() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        Retriever([])


def test_build_retriever_from_dir(tmp_path) -> None:
    (tmp_path / "doc.md").write_text(
        "# Title\n\n## A\nalpha beta gamma", encoding="utf-8"
    )
    r = build_retriever_from_dir(tmp_path)
    out = r.search("alpha", top_k=1)
    assert len(out) == 1
    assert "alpha" in out[0].chunk.text


def test_retrieved_chunk_to_dict_is_jsonable() -> None:
    rc = RetrievedChunk(
        chunk=Chunk(text="x", source="s.md", title="T", section=None),
        score=0.123456,
    )
    d = rc.to_dict()
    assert d["score"] == 0.1235
    assert d["title"] == "T"


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def test_identity_reranker_preserves_order() -> None:
    r = Retriever(_make_chunks())
    retrieved = r.search("inverter", top_k=3)
    ranked = IdentityReranker().rerank("inverter", retrieved, top_k=2)
    assert ranked == retrieved[:2]


def test_identity_reranker_no_topk_returns_all() -> None:
    r = Retriever(_make_chunks())
    retrieved = r.search("inverter", top_k=3)
    ranked = IdentityReranker().rerank("inverter", retrieved)
    assert ranked == retrieved


def test_identity_reranker_negative_topk_raises() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        IdentityReranker().rerank("q", [], top_k=-1)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def test_prompt_builder_renders_recommendation_prompt() -> None:
    from api.schemas import Alert, Severity, SystemType

    chunks = _make_chunks()
    retrieved = [RetrievedChunk(chunk=chunks[0], score=0.91)]
    alert = Alert(
        timestamp=datetime(2026, 5, 9, tzinfo=UTC),
        system_id="PV_001",
        system_type=SystemType.PV,
        fault_class="Inverter_fault",
        severity=Severity.CRITICAL,
        confidence=0.95,
        sensor_snapshot={"P_dc": 250.0, "P_ac": 0.0},
    )
    prompt = PromptBuilder().render_recommendation_prompt(alert, retrieved)
    assert "PV_001" in prompt
    assert "Inverter_fault" in prompt
    assert "P_dc: 250.0" in prompt
    assert "Inverter Fault — Diagnosis" in prompt
    assert "0.91" in prompt
