"""Project-wide :mod:`pathlib` constants.

Use these instead of hardcoded path strings anywhere in the codebase
(see project rule §4 — no hardcoded paths).

中文说明
--------
集中定义仓库根目录及 data、artifacts、reports 等子路径；新增功能时请在此扩展
常量，不要在各模块手写 ``Path(".../omar/data")``。
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"
ORCHESTRATOR_DIR: Path = DATA_DIR / "orchestrator"

# Model artifacts
ARTIFACTS_DIR: Path = PROJECT_ROOT / "quantization" / "artifacts"

# Knowledge base
KB_DOCS_DIR: Path = PROJECT_ROOT / "rag" / "knowledge_base" / "documents"
CHROMA_DIR: Path = PROJECT_ROOT / "rag" / "knowledge_base" / "chroma_db"

# Reports
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Docs
DOCS_DIR: Path = PROJECT_ROOT / "docs"
ALERT_SCHEMA_PATH: Path = DOCS_DIR / "alert_schema.json"

# Configs
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"


def ensure_dir(path: Path) -> Path:
    """Create the directory if it does not exist; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
