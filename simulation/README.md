# `simulation/` — 物理仿真层（Component 1）

本模块按规则 §24 提供完整的工程文档。它是 AgentPV 的**数据源头**：
所有训练 / 验证 / 测试样本都从这里产生。

---

## 1. 模块目的（Purpose）

把"光伏 + 储能系统在不同工况下的传感器时序信号"用**物理近似公式**生成出来，
并按已知规则注入故障特征，得到带标签的训练数据。

---

## 2. 为什么需要这个模块（Why it exists）

作业明文规定 **不能用 Kaggle / HuggingFace 现成的 PV 故障数据集**（PDF §8.5）。
真实工业数据封闭、昂贵、共享受限。学术上的标准做法是**物理仿真**：

- 故障可控注入 → 标签精确
- 数据可复现 → 固定种子即可
- 类别可调 → 训练分布可控

---

## 3. 架构概览（Architecture）

```
        ┌─────────────────┐    ┌────────────────────┐
        │  pv_simulator   │    │ battery_simulator  │
        │  (clean PV ts)  │    │  (clean BESS ts)   │
        └────────┬────────┘    └─────────┬──────────┘
                 │                       │
                 ▼                       ▼
            ┌─────────────────────────────────┐
            │       fault_injector            │
            │  inject_<fault>(arr, label, …)  │
            └────────────────┬────────────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  generate_dataset.py   │
                │   CLI orchestration     │
                │   → data/processed/*   │
                │   → data/version.txt   │
                └────────────────────────┘
```

---

## 4. 输入 / 输出格式（I/O）

### 输入
- `OperatingCondition`：`high_irradiance | low_irradiance | high_temperature`
- `seed: int`：随机种子（reproducibility）
- `window_size: int`：默认 60（即 60 秒窗口，1 Hz 采样）

### 输出
**单条样本**严格符合 `api.schemas.RawSample`：

```python
RawSample(
    window=SensorWindow(
        timestamp_start=...,
        system_id="PV_001",
        system_type=SystemType.PV,
        sample_rate_hz=1.0,
        window_size=60,
        feature_names=["V_dc", "I_dc", "P", "T_module", "T_amb", "G", "P_ac", "eta"],
        values=[[...], ...],            # shape (60, 8)
        operating_condition=OperatingCondition.HIGH_IRRADIANCE,
    ),
    label="Partial_shading",
)
```

**整体数据集** 持久化为 npz + 一个 `DatasetMetadata` JSON：

```
data/processed/
├── X_pv.npz              # arr_0 = (n_pv, 60, 8) float32
├── X_bess.npz            # arr_0 = (n_bess, 60, 8) float32
├── y_pv.npz              # arr_0 = (n_pv,) <U32
├── y_bess.npz            # arr_0 = (n_bess,) <U32
├── meta_pv.csv           # 每条样本的 system_id / condition / timestamp
└── meta_bess.csv

data/splits/
├── train.csv             # sample_id, system, label, condition
├── val.csv
└── test.csv

data/version.txt          # DatasetMetadata.model_dump_json()
```

> **故意把 PV 和 BESS 拆开存**：维度都是 `(N, 60, 8)` 但语义不同；分开后下游训练
> 时可独立 sample，避免混淆。

---

## 5. 关键类与函数

### 5.1 `pv_simulator.PVSimulator`
```python
sim = PVSimulator(seed=42)
window: np.ndarray = sim.simulate(condition, window_size=60)   # (60, 8)
```

物理近似（解释清楚便于答辩）：

- `G` 辐照度：根据工况采样基线（高辐照 800-1100 W/m²；低辐照 100-400；高温 600-900）
- `T_module = T_amb + G * NOCT_coeff`（NOCT 温度模型）
- `I_dc = G/G_STC * (I_sc - α*(T_module - 25))`（电流随辐照线性 + 温度系数）
- `V_dc = V_oc + β*(T_module - 25)`（电压随温度负温度系数）
- `P = V_dc * I_dc`
- `P_ac = η_inv * P`，效率 `η_inv` ≈ 0.96 ± noise
- `η = P_ac / (G * area)` 总效率

### 5.2 `battery_simulator.BatterySimulator`
```python
sim = BatterySimulator(seed=42)
window: np.ndarray = sim.simulate(condition, window_size=60)   # (60, 8)
```

物理近似（RC 等效电路）：

- `V_term = OCV(SOC) - I*R0 - V1`，其中 `V1` 由 RC 一阶 ODE 离散化
- `OCV(SOC) ≈ 3.0 + 1.2 * SOC`（线性近似锂电池开路电压曲线）
- `T` 温度：`T_amb + α_T * |I|` 简化热模型
- `SOC` 由库仑积分更新：`SOC[t+1] = SOC[t] - I[t]*Δt / Q_nom`
- `R_est`：在线辨识简化为 `(V_oc - V_term) / I`（有保护除零）
- `σ_V`：单体电压标准差（多个虚拟单体 SOC 叠 ±0.5% 噪声计算）

### 5.3 `fault_injector.inject_fault`
```python
faulty: np.ndarray = inject_fault(clean=window, label="Partial_shading", rng=rng)
```

每类故障对应独立的纯函数 `_inject_<fault>(arr, rng) -> arr`。
这些函数的物理含义全部记录在函数 docstring 里（答辩用）。

### 5.4 `generate_dataset.generate`
```python
generate(
    out_dir=PROCESSED_DIR,
    n_pv=21000,
    n_bess=17000,
    n_pv_normal=8000,
    n_bess_normal=5000,
    seed=42,
)
```

执行流程：

1. 按目标分布构造 `(system_type, label, condition)` 任务列表，shuffle
2. 每条任务调对应仿真器 + 注入器
3. 通过 `RawSample.model_validate(...)` 强制 schema 校验（任何越界值 → 抛错）
4. 累计写到 npz；同时维护 `meta_*.csv` 索引
5. 最后按 70/15/15 切 train/val/test，写入 `data/splits/*.csv`
6. 写 `data/version.txt`（DatasetMetadata）

---

## 6. 数据流（Data flow）

```
[CLI: python -m simulation.generate_dataset --seed 42 --n-pv 21000 ...]
                       │
                       ▼
        ┌──────────────────────────────┐
        │ build_task_list(distribution)│  ← 决定每条样本什么 type/label/condition
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴───────────────┐
        │  for task in tasks:           │
        │    arr = sim.simulate(...)    │
        │    arr = inject_fault(...)    │
        │    sample = RawSample(...)    │  ← 强校验
        │    accumulate_to_npz_buffer   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  flush npz + csv + version    │
        │  split_train_val_test         │
        └──────────────────────────────┘
```

---

## 7. 设计决策（Design decisions）

| 决策 | 候选 | 选择 | 为什么 |
|---|---|---|---|
| 物理引擎 | pvlib / 自写 | **自写** | 依赖少、代码<300 行、可解释、答辩友好（§27） |
| 存储格式 | parquet / jsonl / **npz+csv** | **npz+csv** | npz 装载 1ms 级，csv 索引人类可读，pandas/numpy 标配 |
| PV 与 BESS 是否合并 | 合并到一个 (N,T,F) | **拆开** | 语义不同；F 维度都是 8 但物理意义两套；下游可独立采样 |
| 样本对象类型 | dict / dataclass / **Pydantic** | **Pydantic (RawSample)** | §3 单一来源 + 自动校验越界 |
| 标签编码 | int / **string** | **string** | 答辩可读；下游训练时再 `LabelEncoder` |
| 类别不平衡处理 | 在生成阶段对齐 / 在训练阶段加权 | **生成阶段对齐 + 训练加权两手准备** | C1 输出文档化分布；C2 用 weighted CE 对齐 |
| 三种工况覆盖 | 任选 / 按比例 | **按比例 (5:3:2)** | 高辐照样本多以贴近真实 PV 数据分布 |

---

## 8. 替代方案（Alternatives considered）

- **pvlib + ECM 库**：学术权威但学习曲线陡，3 周时间不允许。**不选**。
- **直接生成傅里叶 / GAN 合成数据**：能快速产生 50k 样本但缺乏物理可解释性，
  老师肯定追问"这个特征对应什么物理量？"，答辩易翻车。**不选**。
- **真实数据增强**：作业禁止使用现成数据集（§8.5）。**不可选**。

---

## 9. 性能 / 复杂度（Performance）

- 单条样本生成：纯 numpy，单核 < 1 ms
- 51,000 样本：总时长预估 30-60 s（CPU 单线程）
- 内存峰值：估 < 200 MB（npz 流式写入即可，无需常驻全部样本）
- 可选：用 `multiprocessing.Pool` 并行（默认关闭，避免不可重现性）

---

## 10. 测试策略（Testing）

`tests/unit/test_simulation.py` 覆盖：

| 测试 | 目的 |
|---|---|
| `test_pv_simulator_window_shape` | 形状 (T, 8) |
| `test_pv_simulator_no_nan_or_inf` | 无 NaN/Inf |
| `test_pv_simulator_irradiance_range` | G ∈ [0, 1500] |
| `test_pv_simulator_seed_reproducible` | 同 seed 输出相同 |
| `test_battery_simulator_window_shape` | 形状 (T, 8) |
| `test_battery_simulator_soc_in_range` | SOC ∈ [0, 1] |
| `test_battery_simulator_seed_reproducible` | 同 seed 输出相同 |
| `test_fault_injector_changes_signal` | 注入后 ≠ 原始 |
| `test_fault_injector_round_trip_via_schema` | 注入结果可通过 RawSample 校验 |
| `test_generate_small_dataset_smoke` | 小规模 (n=120) 端到端 |

---

## 11. 常见调试问题（Debugging）

| 现象 | 可能原因 | 解决 |
|---|---|---|
| pytest "RawSample requires window.operating_condition" | generate 时漏传 condition | 检查 fault_injector 是否丢字段 |
| Macro F1 训练时极低 | 类间特征差距过小 | 调大 fault_injector 中故障幅度参数 |
| seed 相同但结果不同 | 多线程并行 / 全局 numpy state 污染 | 用 `np.random.default_rng(seed)` 局部 RNG |
| npz 加载慢 | 文件太大（>500MB） | 改用 `mmap_mode="r"` |
| 生成数据集 t-SNE 看不出区分 | 故障幅度过小 / 特征噪声太大 | 答辩前必跑 `notebooks/eda.ipynb` 验证 |

---

## 12. 答辩可能问题（Likely professor questions）

1. **"为什么不用 pvlib？"**
   答：作业要求统计可区分的故障签名而非物理研究级精度。自写仿真依赖少、代码可解释、
   每个公式都能在答辩时白板推导，符合规则 §27 minimal working system。

2. **"数据是仿真的，怎么证明对真实场景有意义？"**
   答：(1) 物理公式来自标准 PV/电池教材（NOCT 温度模型、OCV-SOC 线性近似、库仑计数）；
   (2) 故障注入幅度参考公开论文中的真实故障量级；(3) 模型是迁移学习友好的架构，
   下游可在真实数据集上微调（讨论章节）。

3. **"50k 样本够吗？"**
   答：1D CNN 参数量 ~50k，按经验法则样本数 / 参数 ≥ 1 即足够；本数据集 50k 已经
   远超训练所需。验证集 + 测试集独立采样，可信度足够。

4. **"类别不平衡怎么办？"**
   答：(1) 生成阶段控制各类至少 3500 样本；(2) 训练阶段用 weighted CE / focal loss；
   (3) 评估时用 macro F1 而非 accuracy。

5. **"如何保证训练 / 测试集没有泄漏？"**
   答：每条样本带 `system_id`，划分按 `system_id` 分层抽样，保证同一 system_id
   的样本不会同时出现在不同 split。

---

## 13. 与其他模块的接口（Integration points）

- **下游 `models/` + `training/`**：从 `data/processed/X_*.npz` + `y_*.npz` + `data/splits/*.csv` 读
- **下游 `evaluation/`**：从 `data/version.txt` 读 `class_distribution` 写报告
- **下游 `dashboard/`**：可选地从 `meta_*.csv` 读取做"replay 模式"

---

## 14. 复现命令

```bash
# 全量数据集
python -m simulation.generate_dataset --seed 42

# 小规模冒烟（单元测试用）
python -m simulation.generate_dataset --seed 0 --n-pv 80 --n-bess 40 --out-dir data/_smoke
```
