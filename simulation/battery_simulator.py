"""Physics-inspired battery (BESS) time-series simulator.

This module produces *clean* (fault-free) battery sensor windows. Faults
arrive later via :mod:`simulation.fault_injector`.

Battery model — first-order RC equivalent circuit with Coulomb counting:

* OCV-SOC curve linearised: ``OCV = 3.0 + 1.2 * SOC`` (typical Li-ion).
* Terminal voltage: ``V_term = OCV - I*R0 - V_RC`` with one RC pair
  (``R1, C1``) integrated by Forward Euler.
* SOC update: ``SOC[t+1] = SOC[t] - I[t] * Δt / Q_nom``.
* Thermal: ``T = T_amb + α_T * |I|`` (lumped node).
* Online R0 estimate: ``R_est = (OCV - V_term) / (I + ε)``.
* Cell imbalance proxy ``σ_V``: stddev of N_CELLS individual cells whose
  SOC is offset by tiny Gaussian noise.

Output tensor shape: ``(window_size, 8)`` matching :data:`BESS_FEATURE_NAMES`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from api.schemas import OperatingCondition

# 输入特征顺序（禁止改动）。
BESS_FEATURE_NAMES: tuple[str, ...] = (
    "V_term", "I", "SOC", "T", "R_est", "sigma_V", "N_cycle", "SoH",
)
"""Feature names emitted by :class:`BatterySimulator`."""

# 锂离子单体的近似参数（容量 100 Ah 级模组）。
Q_NOM_AH: float = 100.0          # nominal capacity, Ah
DT_S: float = 1.0                # sample interval (1 Hz)
R0_NOM: float = 0.005            # ohm — instantaneous internal resistance
R1_NOM: float = 0.010            # ohm
C1_NOM: float = 2000.0           # F — drives RC time constant ~20 s
N_CELLS: int = 96                # 96s 模组


@dataclass(frozen=True)
class _ConditionParams:
    """Per-condition charge/discharge sampling profile."""

    t_amb_mean: float
    t_amb_std: float
    current_mean: float       # signed — positive = charge in our convention
    current_std: float


_CONDITION_PARAMS: dict[OperatingCondition, _ConditionParams] = {
    # 高辐照工况通常对应白天充电
    OperatingCondition.HIGH_IRRADIANCE: _ConditionParams(28.0, 3.0, +25.0, 8.0),
    # 低辐照工况通常对应夜间放电
    OperatingCondition.LOW_IRRADIANCE: _ConditionParams(22.0, 3.0, -20.0, 6.0),
    # 高温工况：电流方向中等，温升明显
    OperatingCondition.HIGH_TEMPERATURE: _ConditionParams(38.0, 3.0, +18.0, 7.0),
}


class BatterySimulator:
    """Generate clean BESS sensor windows.

    Parameters
    ----------
    seed:
        Master seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def simulate(
        self,
        condition: OperatingCondition,
        window_size: int = 60,
    ) -> np.ndarray:
        """Return a clean BESS window of shape ``(window_size, 8)``."""

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        rng = self._rng
        cond = _CONDITION_PARAMS[condition]

        # 初始 SOC 在 [0.2, 0.9] 内均匀采样，避开极端 SOC 区
        soc = float(rng.uniform(0.2, 0.9))

        # 电流——围绕条件均值小幅波动；"+" 充电、"-" 放电（与 SOC 计算公式对齐）。
        current = rng.normal(cond.current_mean, cond.current_std, size=window_size)

        # 环境温度
        t_amb = float(rng.normal(cond.t_amb_mean, cond.t_amb_std))

        # 模组的"健康度"状态（缓变，本窗口视为常数）
        n_cycle = float(rng.uniform(50.0, 1500.0))
        soh = float(np.clip(1.0 - n_cycle / 5000.0 + rng.normal(0.0, 0.01), 0.6, 1.0))

        # RC 一阶系统：dV1/dt = -V1/(R1*C1) + I/C1（前向 Euler 离散化）
        v_rc = 0.0
        tau = R1_NOM * C1_NOM
        decay = np.exp(-DT_S / tau)

        v_term = np.zeros(window_size, dtype=np.float32)
        soc_arr = np.zeros(window_size, dtype=np.float32)
        t_arr = np.zeros(window_size, dtype=np.float32)
        r_est = np.zeros(window_size, dtype=np.float32)

        for t in range(window_size):
            i_t = current[t]
            ocv = 3.0 + 1.2 * soc

            # RC 状态前进一步
            v_rc = v_rc * decay + i_t * R1_NOM * (1 - decay)

            v_t = ocv - i_t * R0_NOM - v_rc
            # 加少量量测噪声
            v_t += rng.normal(0.0, 0.002)
            v_term[t] = v_t

            # 库仑积分：充电(I>0) → SOC 上升；放电(I<0) → SOC 下降
            soc = float(np.clip(soc + i_t * DT_S / (Q_NOM_AH * 3600.0), 0.0, 1.0))
            soc_arr[t] = soc

            # 简化热模型：与 |I| 成正比的温升
            t_arr[t] = t_amb + 0.05 * abs(i_t) + rng.normal(0.0, 0.05)

            # 在线 R 估计：避免 I≈0 时除零
            r_est[t] = (ocv - v_t) / (np.abs(i_t) + 1e-3)

        # 单体不平衡 σ_V：N_CELLS 个虚拟单体 SOC 加 ±0.5% 噪声后按 OCV 公式算电压方差
        cell_soc = soc_arr[:, None] + rng.normal(0.0, 0.005, size=(window_size, N_CELLS))
        cell_v = 3.0 + 1.2 * cell_soc
        sigma_v = cell_v.std(axis=1).astype(np.float32)

        # 周期数与 SoH 在窗口内是准静态的（同一组模组）。
        n_cycle_arr = np.full(window_size, n_cycle, dtype=np.float32)
        soh_arr = np.full(window_size, soh, dtype=np.float32)

        window = np.stack(
            [v_term, current.astype(np.float32), soc_arr, t_arr, r_est,
             sigma_v, n_cycle_arr, soh_arr],
            axis=1,
        ).astype(np.float32)
        return window
