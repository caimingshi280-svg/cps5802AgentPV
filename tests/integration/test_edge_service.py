"""Integration tests for the FastAPI edge service.

These tests exercise the HTTP layer with FastAPI's :class:`TestClient` —
no actual ``uvicorn`` process is spawned. They depend on a self-contained
ONNX model which we build inside ``tmp_path`` and point the service at
via the ``AGENTPV_ARTIFACTS_DIR`` environment variable.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.schemas import (  # noqa: E402
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    SystemType,
)
from configs.settings import get_settings  # noqa: E402
from models.cnn1d import CNN1D  # noqa: E402
from quantization.onnx_export import export_checkpoint  # noqa: E402
from training.data import FeatureStats  # noqa: E402

PV_FEATURE_NAMES = [
    "V_dc",
    "I_dc",
    "P",
    "T_module",
    "T_amb",
    "G",
    "P_ac",
    "eta",
]


def _build_onnx(tmp_path, system_type: SystemType):
    classes = list(PV_FAULT_CLASSES) if system_type is SystemType.PV else list(BESS_FAULT_CLASSES)
    n_classes = len(classes)
    torch.manual_seed(0)
    model = CNN1D(in_channels=8, n_classes=n_classes, dropout=0.0)
    stats = FeatureStats(
        mean=np.zeros(8, dtype=np.float32),
        std=np.ones(8, dtype=np.float32),
    )
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "val_macro_f1": 0.95,
        "val_accuracy": 0.95,
        "system_type": system_type.value,
        "n_classes": n_classes,
        "label_classes": classes,
        "feature_stats": stats.to_dict(),
        "model_arch": "CNN1D",
        "in_channels": 8,
        "dropout": 0.0,
    }
    ckpt = tmp_path / f"cnn1d_{system_type.value.lower()}_best.pt"
    torch.save(payload, ckpt)
    onnx = tmp_path / f"cnn1d_{system_type.value.lower()}.onnx"
    export_checkpoint(checkpoint_path=ckpt, output_path=onnx)
    return onnx


@pytest.fixture
def app_with_models(tmp_path, monkeypatch):
    """Spin up the FastAPI app pointed at a tmp artifacts dir with both models."""

    _build_onnx(tmp_path, SystemType.PV)
    _build_onnx(tmp_path, SystemType.BESS)

    monkeypatch.setenv("AGENTPV_ARTIFACTS_DIR", str(tmp_path))
    get_settings.cache_clear()  # type: ignore[attr-defined]

    import importlib

    from api import edge_service

    edge_service = importlib.reload(edge_service)
    return edge_service.app


@pytest.fixture
def client(app_with_models):
    with TestClient(app_with_models) as c:
        yield c


def _sample_window(system_type: SystemType) -> dict:
    rng = np.random.default_rng(seed=0)
    return {
        "timestamp_start": datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC).isoformat(),
        "system_id": f"{system_type.value}_test_001",
        "system_type": system_type.value,
        "sample_rate_hz": 1.0,
        "window_size": 60,
        "feature_names": PV_FEATURE_NAMES,
        "values": rng.standard_normal((60, 8)).tolist(),
        "operating_condition": "high_irradiance",
    }


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


def test_healthz_reports_ok_when_both_models_loaded(client) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "service": "edge", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_pv_returns_alert(client) -> None:
    resp = client.post("/predict", json=_sample_window(SystemType.PV))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["system_type"] == "PV"
    assert body["fault_class"] in PV_FAULT_CLASSES
    assert body["severity"] in ("monitor", "warning", "critical")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_bess_returns_alert(client) -> None:
    resp = client.post("/predict", json=_sample_window(SystemType.BESS))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["system_type"] == "BESS"
    assert body["fault_class"] in BESS_FAULT_CLASSES


def test_predict_rejects_invalid_window(client) -> None:
    """Pydantic validation must reject malformed requests (rule §11)."""

    bad = _sample_window(SystemType.PV)
    bad["window_size"] = 99  # Inconsistent with values' row count
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_extra_fields(client) -> None:
    """StrictBaseModel.extra='forbid' must surface as 422."""

    bad = _sample_window(SystemType.PV)
    bad["unknown_extra_field"] = "should-fail"
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_per_system_latency(client) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"PV", "BESS"}
    for stats in body.values():
        assert "p50_ms" in stats and stats["p50_ms"] > 0
        assert stats["p99_ms"] < 100.0  # Edge budget


# ---------------------------------------------------------------------------
# Degraded mode (only one model present)
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_only_pv(tmp_path, monkeypatch):
    _build_onnx(tmp_path, SystemType.PV)
    monkeypatch.setenv("AGENTPV_ARTIFACTS_DIR", str(tmp_path))
    get_settings.cache_clear()  # type: ignore[attr-defined]
    import importlib

    from api import edge_service

    edge_service = importlib.reload(edge_service)
    return edge_service.app


def test_healthz_reports_degraded_when_bess_missing(app_with_only_pv) -> None:
    with TestClient(app_with_only_pv) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


def test_predict_returns_503_for_missing_model(app_with_only_pv) -> None:
    with TestClient(app_with_only_pv) as c:
        resp = c.post("/predict", json=_sample_window(SystemType.BESS))
        assert resp.status_code == 503
        assert "BESS" in resp.json()["detail"]
