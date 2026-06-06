"""Fault injection over clean simulator output.

Each fault class has a dedicated ``_inject_<fault>`` pure function so any
single fault can be regenerated, inspected, and unit-tested in isolation
(project rule §2). The mappings between labels and injectors are exposed
through :data:`PV_INJECTORS` and :data:`BESS_INJECTORS`.

The fault parameters (magnitude, decay time, etc.) are deliberately picked
to be *statistically distinguishable* yet realistic — see the per-function
docstring for the physics rationale.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from api.schemas import (
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
)
from simulation.battery_simulator import BESS_FEATURE_NAMES
from simulation.pv_simulator import PV_FEATURE_NAMES

# 索引常量——避免在公式里出现魔法 column index（rule §4 no magic numbers）。
_PV_IDX = {name: i for i, name in enumerate(PV_FEATURE_NAMES)}
_BESS_IDX = {name: i for i, name in enumerate(BESS_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# PV faults (7 classes including Normal)
# ---------------------------------------------------------------------------


def _inject_pv_normal(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """No-op for Normal class (returns a copy for purity)."""

    return arr.copy()


def _inject_partial_shading(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Partial shading: I_dc drops 30–70 % for a contiguous fraction of the window.

    Physics: a leaf or bird droppings shadow part of the array → that string's
    current collapses while voltage barely moves; module temp slightly rises
    from the shaded mismatch."""

    out = arr.copy()
    drop = float(rng.uniform(0.30, 0.70))
    start = int(rng.integers(0, max(1, arr.shape[0] // 3)))
    out[start:, _PV_IDX["I_dc"]] *= 1.0 - drop
    out[start:, _PV_IDX["P"]] = (
        out[start:, _PV_IDX["V_dc"]] * out[start:, _PV_IDX["I_dc"]]
    )
    out[start:, _PV_IDX["P_ac"]] *= 1.0 - drop
    out[start:, _PV_IDX["T_module"]] += rng.uniform(1.0, 3.0)
    return out


def _inject_soiling(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Soiling/dust: irradiance reaching the cells dampened by 5–25 %, slow.

    Physics: dust accumulates over time → all electrical output scales down
    proportionally; temperature unchanged."""

    out = arr.copy()
    factor = float(rng.uniform(0.75, 0.95))
    for col in ("I_dc", "P", "P_ac", "G", "eta"):
        out[:, _PV_IDX[col]] *= factor
    return out


def _inject_bypass_diode_fault(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bypass diode short: one substring's contribution to V drops near zero.

    Physics: a shorted bypass diode bypasses 1/3 of the module → V drops by
    roughly 1/3, while I is largely preserved. Local hot-spot raises T."""

    out = arr.copy()
    out[:, _PV_IDX["V_dc"]] *= float(rng.uniform(0.55, 0.75))
    out[:, _PV_IDX["P"]] = out[:, _PV_IDX["V_dc"]] * out[:, _PV_IDX["I_dc"]]
    out[:, _PV_IDX["P_ac"]] *= float(rng.uniform(0.55, 0.75))
    out[:, _PV_IDX["T_module"]] += rng.uniform(3.0, 8.0)
    return out


def _inject_string_disconnection(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """String disconnect: stepwise current → 0 at random instant in window.

    Physics: a connector fails → entire string opens → current collapses
    instantly. V_oc may slightly rise as load drops."""

    out = arr.copy()
    t = int(rng.integers(arr.shape[0] // 4, arr.shape[0]))
    out[t:, _PV_IDX["I_dc"]] = rng.normal(0.0, 0.02, size=arr.shape[0] - t).clip(min=0.0)
    out[t:, _PV_IDX["P"]] = 0.0
    out[t:, _PV_IDX["P_ac"]] = 0.0
    out[t:, _PV_IDX["V_dc"]] *= 1.05  # mild open-circuit rise
    return out


def _inject_inverter_fault(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inverter fault: P_ac oscillates ±20 % around DC power.

    Physics: control loop instability inside the inverter creates AC output
    oscillations that don't reflect DC input."""

    out = arr.copy()
    osc_freq_hz = float(rng.uniform(0.05, 0.3))
    t = np.arange(arr.shape[0], dtype=np.float32)
    osc = 1.0 + 0.20 * np.sin(2 * np.pi * osc_freq_hz * t + rng.uniform(0, 2 * np.pi))
    out[:, _PV_IDX["P_ac"]] *= osc
    eta_denom = np.clip(arr[:, _PV_IDX["G"]] * 1.6, 1e-3, None)
    out[:, _PV_IDX["eta"]] = np.clip(out[:, _PV_IDX["P_ac"]] / eta_denom, 0.0, 1.0)
    return out


def _inject_degradation(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Long-term degradation: power output 10–25 % below new condition.

    Physics: cumulative UV/thermal stress slowly reduces P_max over years.
    No transient signature — this is the "slow" fault class."""

    out = arr.copy()
    factor = float(rng.uniform(0.75, 0.90))
    for col in ("I_dc", "P", "P_ac", "eta"):
        out[:, _PV_IDX[col]] *= factor
    out[:, _PV_IDX["V_dc"]] *= float(rng.uniform(0.95, 0.99))
    return out


PV_INJECTORS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "PV_Normal": _inject_pv_normal,
    "Partial_shading": _inject_partial_shading,
    "Soiling": _inject_soiling,
    "Bypass_diode_fault": _inject_bypass_diode_fault,
    "String_disconnection": _inject_string_disconnection,
    "Inverter_fault": _inject_inverter_fault,
    "Degradation": _inject_degradation,
}
assert set(PV_INJECTORS.keys()) == set(PV_FAULT_CLASSES), (
    "PV_INJECTORS keys must match api.schemas.PV_FAULT_CLASSES"
)


# ---------------------------------------------------------------------------
# BESS faults (5 classes including Normal)
# ---------------------------------------------------------------------------


def _inject_bess_normal(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """No-op for Normal class."""

    return arr.copy()


def _inject_capacity_fade(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gradual capacity fade: SoH dropped 15–30 %, SOC swings widen.

    Physics: aging reduces usable capacity → for the same Ah throughput SOC
    moves more per second, so SOC trajectory steepens. SoH state low."""

    out = arr.copy()
    fade = float(rng.uniform(0.15, 0.30))
    out[:, _BESS_IDX["SoH"]] = np.clip(out[:, _BESS_IDX["SoH"]] - fade, 0.4, 1.0)
    out[:, _BESS_IDX["SOC"]] = np.clip(
        0.5 + (out[:, _BESS_IDX["SOC"]] - 0.5) * (1.0 + 1.5 * fade),
        0.0,
        1.0,
    )
    return out


def _inject_internal_resistance(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Internal resistance ↑ 30–100 %: V_term dips more under load.

    Physics: SEI growth or contact resistance → larger I*R drop under same
    current; R_est measurement increases proportionally."""

    out = arr.copy()
    factor = float(rng.uniform(1.30, 2.00))
    extra_drop = (factor - 1.0) * 0.005 * np.abs(out[:, _BESS_IDX["I"]])
    out[:, _BESS_IDX["V_term"]] -= extra_drop
    out[:, _BESS_IDX["R_est"]] *= factor
    out[:, _BESS_IDX["T"]] += rng.uniform(1.0, 3.0)
    return out


def _inject_thermal_anomaly(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Thermal anomaly: T rises >2 °C/min, exceeds safe envelope.

    Physics: a cooling fan failure or internal short causes runaway heating.
    This is the SAFETY-CRITICAL class — must be reliably detected."""

    out = arr.copy()
    rate_per_step = float(rng.uniform(0.04, 0.10))  # °C / s
    ramp = rate_per_step * np.arange(arr.shape[0], dtype=np.float32)
    out[:, _BESS_IDX["T"]] += ramp
    return out


def _inject_cell_imbalance(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cell imbalance: σ_V grows several × normal.

    Physics: mismatched cells diverge in SOC → larger inter-cell voltage
    spread; pack-level V_term is largely unaffected."""

    out = arr.copy()
    factor = float(rng.uniform(3.0, 8.0))
    out[:, _BESS_IDX["sigma_V"]] *= factor
    return out


BESS_INJECTORS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "BESS_Normal": _inject_bess_normal,
    "Capacity_fade": _inject_capacity_fade,
    "Internal_resistance_increase": _inject_internal_resistance,
    "Thermal_anomaly": _inject_thermal_anomaly,
    "Cell_imbalance": _inject_cell_imbalance,
}
assert set(BESS_INJECTORS.keys()) == set(BESS_FAULT_CLASSES), (
    "BESS_INJECTORS keys must match api.schemas.BESS_FAULT_CLASSES"
)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def inject_fault(
    clean: np.ndarray,
    label: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to the right injector based on label.

    Parameters
    ----------
    clean:
        Clean simulator output, shape ``(window_size, 8)``.
    label:
        One of :data:`api.schemas.ALL_FAULT_CLASSES`.
    rng:
        Generator from which all stochastic decisions are drawn — pass the
        same generator from the caller to keep determinism per-window.

    Returns
    -------
    np.ndarray
        Faulty array, same shape and dtype as ``clean``.
    """

    if label in PV_INJECTORS:
        return PV_INJECTORS[label](clean, rng).astype(clean.dtype, copy=False)
    if label in BESS_INJECTORS:
        return BESS_INJECTORS[label](clean, rng).astype(clean.dtype, copy=False)
    raise ValueError(f"Unknown fault label: {label!r}")
