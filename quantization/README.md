# `quantization/` — 模型压缩与 ONNX 导出（Component 2+ / 8）

按规则 §24 的 14 节工程文档。本模块负责把 PyTorch checkpoint 转换成
**部署用 artifact**：自包含 ONNX 文件（已含输入标准化），并可进一步
做 **INT8 静态量化**满足作业 §4.2 压缩要求。结构化剪枝 / KD 在
polish 阶段加入。

---

## 1. 模块目的（Purpose）

把训练好的 .pt → 边缘部署可用的 .onnx，且让 .onnx **完全自包含**：
- 包含 per-channel 输入标准化（曾用 BESS 修复 bug 的 feature_stats 嵌入图里）
- 元数据里带 `agentpv.label_classes` / `agentpv.system_type`，下游一查就知道用法
- ONNX checker 校验通过

---

## 2. 为什么需要这个模块（Why it exists）

- 边缘节点不一定能装 PyTorch（资源、依赖、license 都是问题）；ONNX Runtime 体积小、CPU 友好
- 部署不能 ship "模型 + 标准化参数 + 标签列表" 三个文件——必须**单一 .onnx 文件即可推理**，避免对齐错误
- 作业 §3.2 明确要求 ONNX 导出 + ONNX 文件 ≤ 50 MB

---

## 3. 架构概览（Architecture）

```text
.pt checkpoint ─┐
                │
                ▼
   _build_model_from_checkpoint()
   ┌───────────────────────────┐
   │ CNN1D (load_state_dict)   │
   │     +                      │
   │ FeatureStats (mean,std)    │ ─── StandardizingClassifier ───┐
   └───────────────────────────┘                                  │
                                                                  ▼
                                                        torch.onnx.export
                                                                  │
                                                                  ▼
                                                          fake.onnx (graph)
                                                                  │
                                                                  ▼
                                                  _attach_metadata()
                                                  ── system_type, labels, stats
                                                  ── opset_version, training_f1
                                                                  │
                                                                  ▼
                                                          .onnx (validated)
```

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `onnx_export.py` | `StandardizingClassifier` 包装 + `export_checkpoint` CLI（FP32 ONNX） |
| `int8_static.py` | INT8 静态后训练量化（PTQ）+ CLI（产 `*_int8.onnx`） |
| `artifacts/` | 输出目录：`cnn1d_{pv,bess}.onnx`、`cnn1d_{pv,bess}_int8.onnx`、`cnn1d_*_best.pt` |

未来：

| 文件 | 作用 | 时机 |
|---|---|---|
| `prune.py` | 结构化剪枝 30% 通道 | polish 阶段 |
| `kd.py` | 知识蒸馏（teacher → student） | polish 阶段 |

---

## 5. 输入输出契约（Inputs & Outputs）

### `export_checkpoint(checkpoint_path, output_path, opset=17)`
- 输入：含完整 payload 的 .pt（必须有 `feature_stats` / `label_classes` / `model_arch`）
- 输出：合规 .onnx 文件 + `agentpv.*` 元数据
- 失败：缺 `feature_stats` 立即 KeyError；ONNX checker 失败 → 抛异常

### `quantize_to_int8_static(*, fp32_onnx_path, int8_onnx_path, system_type, samples_per_class=30, ...)` — S12 新增
- 输入：FP32 自含 ONNX；同 system 的训练集（自动从 `data/processed/` 加载）
- 输出：INT8 ONNX；自动复制 `agentpv.*` 元数据并加 `agentpv.precision=int8` 标记
- 内部步骤：`quant_pre_process` → 选 `samples_per_class × n_classes` 校准样本 →
  `quantize_static`（QDQ format / QInt8 weights+activations / MinMax 校准）→ 元数据回填 →
  `onnx.checker.check_model`
- 失败：FP32 不存在 → FileNotFoundError；ONNX checker 失败 → 抛异常

---

## 6. 关键设计决策（Design decisions）

### FP32 ONNX 导出
| 决策 | 备选 | 理由 |
|---|---|---|
| 把 mean/std 嵌入图内 | sidecar JSON | 避免"导出 .onnx 时忘了打包 stats"导致部署侧推理结果错（实际遇到的 bug 类型） |
| metadata_props 存标签 | 文件名约定 | 文件复制 / 重命名后仍能查到原始 system_type；linter 可以验证 |
| `dynamo=False` | dynamo exporter | dynamo exporter 需要 `onnxscript`（额外重依赖），且 PyTorch 2.11 上对小模型仍处于实验态 |
| opset=17 | 18+ | onnxruntime 1.26 对 17 支持最稳定；仍包含 BatchNorm/Conv1d 全部 op |
| 加 `do_constant_folding=True` | 关 | 让标准化的 Sub/Div 折叠到 Conv weights，启动更快 |
| 校验 ONNX 标签与 schema 一致 | 信任导出器 | rule §3：schema 是单一信源 |

### INT8 静态量化（S12 新增）
| 决策 | 备选 | 理由 |
|---|---|---|
| 静态 PTQ（quantize_static） | 动态量化 | 动态量化对 Conv1d 收益小（只量化 MatMul/Linear）；CNN 必须用静态 |
| QDQ format | QOperator | QDQ 在 ORT CPU EP 上最稳定，可被任意支持 ONNX 的 runtime 解读 |
| QInt8 weights + activations | QUInt8 | 我们的传感器读数有负值（电流、温度差），signed 范围更合适 |
| `MinMax` 校准 | `Entropy` (KL) / `Percentile` | 训练集已平衡 + 50k 样本量充足；MinMax 简单可重现 |
| 每类 30 样本（PV=210 / BESS=150） | 全训练集 | 50k 全集会让 ORT 校准跑超过 1 分钟；30 张已足够覆盖每类的激活分布 |
| 校准 batch_size=1 | 32+ | 单样本校准与边缘部署一致；ORT 默认即可 |
| 不做 per-channel | per-channel 权重 | per-tensor 实现简单确定，模型小（180 KB）也没必要切 per-channel |
| 元数据回填（含 `agentpv.precision=int8`） | 重新 attach 全部 | 复用 FP32 导出的 metadata，0 风险地保留下游契约 |

---

## 7. 反例（What NOT to put here）

- ❌ ONNX Runtime 的推理代码（属于 `inference/`）
- ❌ 训练 / 微调（属于 `training/`）
- ❌ 评估指标对比表（属于 `evaluation/`）

---

## 8. 教学注释

- **为什么 wrap 一层 `StandardizingClassifier`？** 因为 PyTorch 把 `register_buffer` 的张量视为图常量，`torch.onnx.export` 会把 `(x - mean) / std` 直接嵌进图，constant folding 后甚至可以与第一层 Conv 融合。
- **为什么导出时 `model.eval()`？** BatchNorm 在 train 模式用 batch 统计、eval 模式用 running 统计；导出 train 模式的图会让推理结果完全错——曾在 S07 的单元测试里遇到过 0.1 量级的数值偏差，就是这个原因。
- **为什么 dummy_input shape `(1, 60, 8)`？** trace 一次得到 graph，但通过 `dynamic_axes={'sensor_window': {0: 'batch'}}` 解锁 batch 维度，部署时可以 batch=1 也可以 batch=N。

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `training/train.py` 产 .pt | `inference/onnx_runner.py` 加载 .onnx |
| `training/data.py::FeatureStats` 提供 stats schema | `api/edge_service.py` 启动时读 |

---

## 10. 测试覆盖（Tests）

`tests/unit/test_onnx_export.py`（6 项）：

- `StandardizingClassifier` 数学正确性
- 导出文件存在 + ONNX checker 通过
- metadata 完整且字符串可 JSON-decode
- ONNX vs PyTorch（含 stats）数值平价 ≤ 1e-3
- 旧 checkpoint（无 feature_stats）必须报错
- 导出图支持动态 batch（1/5/32 都能跑）

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 当前 FP32 | 当前 INT8 |
|---|---|---|---|
| 文件大小 | < 50 MiB | 0.18 MiB（PV / BESS） | **0.058 MiB（×3.16 压缩）** |
| 导出时间 | < 30 s | < 5 s | < 8 s（含校准） |
| 数值平价（vs PyTorch） | max abs diff < 1e-3 | 4.9e-4（PV）/ 5.0e-5（BESS） | — |
| Macro-F1（test split） | ≥ 0.85（作业阈值） | PV 0.9993 / BESS 0.9851 | **PV 0.9995 / BESS 0.9778** |
| p95 延迟 | ≤ 100 ms | 0.168 ms（PV）/ 0.148 ms（BESS） | **0.098 / 0.090 ms** |

---

## 12. 未来扩展（Future work）

- `prune.py`：剪 30% Conv 通道 → 微调 5 epoch → 重新导出 → 第 4 个 variant
- `kd.py`：teacher (FP32 CNN) → student (smaller CNN)，配合 INT8 实现极致压缩
- 从 `onnxruntime.quantization.MinMax` 升级到 `Entropy`，并对比是否能挽回 BESS INT8 的 -0.7% F1 漂移
- INT8 量化时支持 per-channel 权重，看能否进一步减小 BESS 漂移

---

## 13. 运行示例（Usage example）

### FP32 导出

```bash
python -m quantization.onnx_export \
    --checkpoint quantization/artifacts/cnn1d_pv_best.pt \
    --output    quantization/artifacts/cnn1d_pv.onnx \
    --opset 17

python -m quantization.onnx_export \
    --checkpoint quantization/artifacts/cnn1d_bess_best.pt \
    --output    quantization/artifacts/cnn1d_bess.onnx
```

### INT8 静态量化（S12 新增）

```bash
# PV: 用 train split 每类 30 样本（共 210）做校准
python -m quantization.int8_static --system pv --samples-per-class 30 --seed 42

# BESS: 同上（共 150 样本）
python -m quantization.int8_static --system bess --samples-per-class 30 --seed 42
```

### 完整流水线（FP32 → INT8 → 三变体评估）

```bash
# 1) 训练 → 2) ONNX → 3) INT8 → 4) 三变体对照
python -m training.train --system pv  --epochs 25 --batch-size 256 --lr 1e-3
python -m training.train --system bess --epochs 25 --batch-size 256 --lr 1e-3
python -m quantization.onnx_export --checkpoint quantization/artifacts/cnn1d_pv_best.pt
python -m quantization.onnx_export --checkpoint quantization/artifacts/cnn1d_bess_best.pt
python -m quantization.int8_static --system pv
python -m quantization.int8_static --system bess
python -m evaluation --systems pv bess --variants pytorch_fp32 onnx_fp32 onnx_int8 --compare
```

---

## 14. 已知限制（Known limitations）

- TorchScript exporter 对自定义 Python 控制流（如 `if x.shape[-1] != ...`）会抛 TracerWarning。当前的输入校验是运行时-only，不会被 trace 进图——这正是我们要的（运行时校验仅在 PyTorch 侧防呆）
- INT8 静态量化目前只对 ONNX FP32 模型有效；如果换 LSTM / TCN，需要重新评估 `quantize_static` 在该架构上的支持度
- INT8 量化产物的 BESS Macro-F1 较 FP32 略掉 0.73%（**仍 ≫ 0.85 阈值**）；可在 polish 阶段尝试 `Entropy` 校准 / per-channel 权重恢复
- 静态量化前的 `quant_pre_process` 会临时写一个 `*.pre.onnx` 中间文件，正常会清理；如果磁盘满了可能残留
- 结构化剪枝 / KD **未实现**；polish 阶段补
- 不支持非 CNN 模型（LSTM / TCN）；polish 时若加新架构需扩展 `_build_model_from_checkpoint`
