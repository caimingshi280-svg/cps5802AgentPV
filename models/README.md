# `models/` — 模型架构（Component 2 part 1）

本模块按规则 §24 提供完整工程文档。它定义 AgentPV 的故障分类器**架构**，
不包含训练循环（见 `training/`）也不包含推理服务（见 `api/edge_service.py`、
`inference/`）。

---

## 1. 模块目的（Purpose）

提供轻量、可量化、可导出 ONNX 的 PyTorch 模型，把传感器时序窗口
`(B, T=60, F=8)` 映射到故障类别 logits `(B, n_classes)`。

---

## 2. 为什么需要这个模块（Why it exists）

作业 §3 要求 **edge inference 在 CPU 上 < 100 ms**，且模型经过压缩
（剪枝 + INT8 量化）。这要求架构在设计阶段就为压缩留出空间：

- 没有 RNN / Transformer 状态（量化与 ONNX 友好）
- 卷积通道数留有冗余（剪枝 30% 后仍能保留性能）
- 全模型 < 200k 参数（部署到边缘节点目标）

---

## 3. 架构概览（Architecture）

```text
BaseClassifier (abstract)
    └── CNN1D
            ├── feature_extractor (Conv1d × 3 + BatchNorm + AdaptiveAvgPool)
            └── classifier         (Linear + Dropout + Linear)
```

输入按 `(B, T, F)` 接受（与 `api.schemas.SensorWindow.values` 同形状），
内部 transpose 到 PyTorch Conv1d 期望的 `(B, F, T)`。

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `base.py` | `BaseClassifier` 抽象类，约束所有未来模型必须实现 `forward(B,T,F) -> (B,n)` |
| `cnn1d.py` | 1D CNN 默认架构，~46k 参数，CPU 上单样本推理 <5 ms |

后续阶段会增加：

- `lstm.py`（备选：消融实验对比）
- `tcn.py`（备选：长依赖场景）

---

## 5. 输入输出契约（Inputs & Outputs）

| 项 | 张量形状 | 单位 / 取值 |
|---|---|---|
| 输入 `x` | `(B, 60, 8)` | 标准化后的 float32（mean=0, std=1，per-channel） |
| 输出 logits | `(B, n_classes)` | 未 softmax 的实数；下游 `inference.postprocess` 做 softmax |

`n_classes` 在构造时给定：PV=7，BESS=5。

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 选择理由 |
|---|---|---|
| 1D CNN | LSTM / Transformer | 局部时间模式（突变/振荡/缓变）就足够；CPU 上更快、更易量化 |
| AdaptiveAvgPool1d(1) | Flatten 全连接 | 时间维度池化 → 参数与序列长度解耦，便于变窗口实验 |
| BatchNorm1d | LayerNorm / GroupNorm | INT8 量化对 BN 支持最成熟（折叠到前面的 Conv） |
| 双模型（PV + BESS） | 单模型 12 类 | 物理特征不同；分开训练边界更清晰，可独立部署 |

---

## 7. 反例（What NOT to put here）

- ❌ 数据加载、标准化（属于 `training/data.py`）
- ❌ 损失函数、优化器（属于 `training/losses.py`、`training/trainer.py`）
- ❌ ONNX 导出脚本（属于 `quantization/`）
- ❌ 任何 print / matplotlib（rule §10：必须用 `get_logger`）

---

## 8. 教学注释（Tensor shape walk-through）

`cnn1d.py` 顶部 docstring 详细列出每一层的张量形状变化（用作答辩材料）。
Conv1d 的 `padding=k//2` 保持时间维不变，`MaxPool1d(2)` 把 T 从 60 降到 30，
`AdaptiveAvgPool1d(1)` 把任意 T 收成长度 1，最终 `Flatten` 得到 `(B, 128)`。

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `training/data.py` 提供标准化后的张量 | `training/trainer.py` 调 `model(x)` 训练 |
| | `inference/onnx_runner.py` 加载 .pt → 导出 .onnx → ONNX Runtime 执行（S07 实现） |

模型不感知 PV / BESS 业务语义——这一切由 `n_classes` 参数和 `feature_stats`
决定，存在 checkpoint 里。

---

## 10. 测试覆盖（Tests）

`tests/unit/test_models.py`：

- forward 形状（PV / BESS 两种 n_classes）
- logits 全是有限数
- 拒绝 2D / 错误 channel 数输入
- 参数量在合理区间（10k–200k）
- 梯度回传非空

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 当前测量（MVP，CPU） |
|---|---|---|
| 单样本推理（FP32 PyTorch） | < 100 ms | < 2 ms（轻松满足） |
| 模型大小（FP32 .pt） | < 1 MB | ~190 KB |
| 训练时长（30 epoch on n=2400） | < 1 min | ~17 s（PV）/ ~15 s（BESS） |

INT8 量化与 ONNX 导出在 S08 完成；目标体积 < 100 KB、单样本 < 5 ms。

---

## 12. 未来扩展（Future work）

- `models/lstm.py` 用作消融
- `models/tcn.py` 长依赖（如果窗口扩到 600s）
- 加 `models/registry.py` 提供 `build_model(name, **kwargs)`

---

## 13. 运行示例（Usage example）

```python
from models.cnn1d import CNN1D
import torch

model = CNN1D(in_channels=8, n_classes=7)
x = torch.randn(2, 60, 8)
logits = model(x)
print(logits.shape)
print("params:", model.num_parameters())
```

---

## 14. 已知限制（Known limitations）

- 接受固定 `in_channels=8`；如果 simulator 输出维度变化，需重新训练
- AdaptiveAvgPool1d 隐含假设：故障在窗口内的**任意位置**都触发，
  不要求时序精确对齐。如果未来要求"故障发生时刻定位"，需改为 sequence-output 模型
- BatchNorm 在 batch_size=1 推理时退化到 running mean/std；
  我们已 `model.eval()` 切到 inference 模式，不影响正确性
