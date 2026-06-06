"""Shared pytest fixtures and path bootstrapping."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure the project root is on sys.path so tests can import top-level
# packages (api, configs, utils, ...) without an installed package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Pin tests to deterministic config defaults.
os.environ.setdefault("APP_ENV", "test")

import configs.settings as _settings_mod  # noqa: E402

_settings_mod.get_settings.cache_clear()
_settings_mod.settings = _settings_mod.get_settings()
