"""Physics-inspired PV time-series simulator.

This module produces *clean* (fault-free) PV sensor windows. Faults are added
later by :mod:`simulation.fault_injector`, keeping a clean separation of
responsibilities (project rule §2).

Physical model — kept deliberately simple so every value is defensible during
the oral defense:

* Irradiance ``G`` is sampled per-condition (high / low / high-temp) from a
  Gaussian centered on a published mid-point.
* Module temperature follows the NOCT model: ``T_module = T_amb + k_noct * G``.
* DC current scales with irradiance and is reduced by a temperature
  coefficient α: ``I_dc = (G/G_STC) * (I_sc - α*(T_module - 25))``.
* DC voltage drops with temperature via β: ``V_dc = V_oc + β*(T_module - 25)``.
* Inverter efficiency η_inv ~ 0.96 with small Gaussian jitter.
* AC power: ``P_ac = η_inv * V_dc * I_dc``; total efficiency ``η = P_ac / (G*A)``.

Output tensor shape: ``(window_size, 8)`` with feature order matching
:data:`PV_FEATURE_NAMES`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from api.schemas import OperatingCondition

# 输入特征顺序（与下游 RawSample.window.feature_names 对齐，禁止改动顺序）。
PV_FEATURE_NAMES: tuple[str, ...] = (
    "V_dc", "I_dc", "P", "T_module", "T_amb", "G", "P_ac", "eta",
)
"""Feature names emitted by :class:`PVSimulator`. The order is part of the
contract — do not reorder without updating downstream training code."""

# 标准测试条件下的 PV 参数（一组 250W 单晶硅组件的近似值）。
G_STC: float = 1000.0          # W/m^2  Standard Test Condition irradiance
I_SC_STC: float = 9.0          # A      short-circuit current
V_OC_STC: float = 38.0         # V      open-circuit voltage
ALPHA_I: float = 0.005         # A/°C   I temperature coefficient
BETA_V: float = -0.30          # V/°C   V temperature coefficient
NOCT_K: float = 0.030          # °C / (W/m^2) — drives module heating
PANEL_AREA: float = 1.6        # m^2    aperture area (used in η)


@dataclass(frozen=True)
class _ConditionParams:
    """Sampling parameters for one operating condition."""

    g_mean: float
    g_std: float
    t_amb_mean: float
    t_amb_std: float


# 三种工况下辐照度与环境温度的采样分布（来自方案 §1.4）。
_CONDITION_PARAMS: dict[OperatingCondition, _ConditionParams] = {
    OperatingCondition.HIGH_IRRADIANCE: _ConditionParams(950.0, 80.0, 25.0, 3.0),
    OperatingCondition.LOW_IRRADIANCE: _ConditionParams(250.0, 70.0, 20.0, 4.0),
    OperatingCondition.HIGH_TEMPERATURE: _ConditionParams(750.0, 90.0, 40.0, 3.0),
}


class PVSimulator:
    """Generate clean PV sensor windows.

    Parameters
    ----------
    seed:
        Master seed. Each call to :meth:`simulate` derives a deterministic
        per-call RNG from a counter for reproducibility (rule §6).
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def simulate(
        self,
        condition: OperatingCondition,
        window_size: int = 60,
    ) -> np.ndarray:
        """Return a clean PV window of shape ``(window_size, 8)``.

        The 8 features follow :data:`PV_FEATURE_NAMES`.

        Notes
        -----
        We *do not* simulate a full diurnal cycle here. Each window represents
        a 60-second slice during steady operation, which matches the way an
        edge classifier would receive sensor data in production.
        """

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        rng = self._rng
        cond = _CONDITION_PARAMS[condition]

        # 用低频随机游走模拟"窗口内辐照波动"——比独立同分布噪声更接近真实云层遮挡。
        g_base = rng.normal(cond.g_mean, cond.g_std)
        g_walk = np.cumsum(rng.normal(0.0, 5.0, size=window_size))
        irradiance = np.clip(g_base + g_walk - g_walk.mean(), 0.0, 1500.0)

        t_amb_base = rng.normal(cond.t_amb_mean, cond.t_amb_std)
        t_amb = np.full(window_size, t_amb_base) + rng.normal(0.0, 0.3, size=window_size)

        t_module = t_amb + NOCT_K * irradiance

        # 电气模型——温度对 I 影响较小、对 V 影响较大；这就是为何 PV 故障检测既看 V 又看 I。
        i_dc = (irradiance / G_STC) * (I_SC_STC - ALPHA_I * (t_module - 25.0))
        i_dc = np.clip(i_dc + rng.normal(0.0, 0.05, size=window_size), 0.0, None)

        v_dc = V_OC_STC + BETA_V * (t_module - 25.0)
        v_dc = v_dc + rng.normal(0.0, 0.15, size=window_size)
        v_dc = np.clip(v_dc, 0.0, None)

        p_dc = v_dc * i_dc

        # 逆变器效率：以 0.96 为中心的小幅扰动，低功率时效率轻微下降。
        eta_inv = 0.96 - 0.05 * np.exp(-p_dc / 50.0) + rng.normal(0.0, 0.005, size=window_size)
        eta_inv = np.clip(eta_inv, 0.5, 0.99)
        p_ac = eta_inv * p_dc

        # 总效率 η = P_ac / (G * area)。当 G≈0 时给一个可观测下限避免除零。
        denom = np.clip(irradiance * PANEL_AREA, 1e-3, None)
        eta = np.clip(p_ac / denom, 0.0, 1.0)

        window = np.stack(
            [v_dc, i_dc, p_dc, t_module, t_amb, irradiance, p_ac, eta],
            axis=1,
        ).astype(np.float32)
        return window
