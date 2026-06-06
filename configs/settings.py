"""Global configuration loader for AgentPV.

Resolution order (highest priority first):

1. Environment variables prefixed with ``AGENTPV_``.
2. ``.env`` file at the project root.
3. ``configs/<APP_ENV>.yaml`` overlay (APP_ENV defaults to ``dev``).
4. ``configs/base.yaml`` defaults.
5. Pydantic field defaults declared on :class:`Settings`.

中文说明
--------
本模块为全项目的**单一配置入口**。业务代码应通过 ``get_settings()`` 读取路径、
LLM 后端、知识库目录等，避免在多处重复解析 yaml / 环境变量。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """Typed application settings shared by every service."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_prefix="AGENTPV_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    artifacts_dir: Path = _PROJECT_ROOT / "quantization" / "artifacts"
    knowledge_base_dir: Path = _PROJECT_ROOT / "rag" / "knowledge_base" / "documents"
    chroma_dir: Path = _PROJECT_ROOT / "rag" / "knowledge_base" / "chroma_db"

    # Reproducibility
    seed: int = 42

    # ML / training
    batch_size: int = 256

    # RAG
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    vector_backend: str = Field(default="chromadb", description="chromadb | faiss")
    rag_retrieval: str = Field(
        default="auto",
        description="auto | tfidf | chroma — auto uses Chroma when the persisted index is non-empty.",
    )
    chroma_collection_name: str = "agentpv_kb"

    # Inference
    onnx_threads: int = 4
    inference_latency_budget_ms: int = 100

    # Agent
    llm_backend: str = Field(
        default="mock",
        description="mock | ollama (local HTTP only; no cloud LLM vendor in-tree).",
    )
    llm_timeout_s: int = 30
    react_max_iterations: int = 5

    # Service URLs (override per-environment)
    edge_url: str = "http://edge-service:8000"
    agent_url: str = "http://agent-service:8001"
    vector_db_url: str = "http://vector-db:8000"

    # Local LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Logging
    log_level: str = "INFO"
    log_format: str = Field(default="json", description="json | text")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk, returning an empty dict when absent."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _agentpv_env_overrides() -> dict[str, Any]:
    """Map ``AGENTPV_*`` process environment keys to :class:`Settings` field names.

    YAML overlays are merged before this step; values here must win so that
    ``AGENTPV_LLM_BACKEND=mock`` overrides ``llm_backend`` in ``dev.yaml``.
    """
    prefix = "AGENTPV_"
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        field = key[len(prefix) :].lower()
        if not field:
            continue
        out[field] = value
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings, applying base + environment overlays."""
    env = os.getenv("APP_ENV", "dev").strip().lower() or "dev"
    base_overrides = _load_yaml(_CONFIGS_DIR / "base.yaml")
    env_overrides = _load_yaml(_CONFIGS_DIR / f"{env}.yaml")
    merged: dict[str, Any] = {
        **base_overrides,
        **env_overrides,
        **_agentpv_env_overrides(),
    }
    return Settings(**merged)


settings: Settings = get_settings()
