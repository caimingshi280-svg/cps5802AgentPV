# `training/` — 训练流水线（Component 2 part 2）

本模块按规则 §24 提供完整工程文档。它把 `simulation/` 产出的 npz 数据集与
`models/` 的网络架构组合起来，跑出可部署的 checkpoint。

---

## 1. 模块目的（Purpose）

提供一个**幂等可复现**的训练入口：相同 seed + 相同数据 + 相同超参 → 相同 .pt 文件
（参数差异仅在数值精度内）。

---

## 2. 为什么需要这个模块（Why it exists）

作业 §3.2 要求"baseline + 压缩 + 量化"三轮训练 / 微调，并要求每一轮的
评估指标可对比。把训练循环抽离成独立模块后：

- 同一份 `Trainer` 既能跑 baseline 也能跑压缩后的微调
- 训练日志结构化（rule §10），可由组件 3 直接消费做 tensorboard / 表格
- 测试时只用 fixture 注入小数据集即可，不需要完整数据 IO

---

## 3. 架构概览（Architecture）

```text
training/
├── data.py        ← TimeSeriesNpzDataset + LabelMap + FeatureStats + class_weights
├── losses.py      ← WeightedCrossEntropyLoss + FocalLoss
├── trainer.py     ← Trainer 类（fit / 保存 / early stop）
└── train.py       ← argparse CLI：python -m training.train --system pv
```

数据流：

```text
data/processed/X_pv.npz ──┐
data/splits/train.csv     │   _load_split_arrays
                          ▼
                  FeatureStats.fit (训练集)  ──存到 checkpoint──┐
                          │                                       │
                          ▼                                       │
                  TimeSeriesNpzDataset(train, std=stats)          │
                  TimeSeriesNpzDataset(val,   std=stats)          │
                          │                                       │
                          ▼                                       │
                       DataLoader                                  │
                          │                                       │
                          ▼                                       │
                  Trainer(model, loss, optim).fit() ───→ best.pt ─┘
```

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `data.py` | 把 npz + split csv 包成 `torch.utils.data.Dataset`，并提供 per-channel 标准化 |
| `losses.py` | 加权交叉熵（默认）+ Focal Loss（备用） |
| `trainer.py` | Trainer 类：epoch 循环、early stop、checkpoint 保存 |
| `train.py` | CLI 入口；`python -m training.train --system pv` |

---

## 5. 输入输出契约（Inputs & Outputs）

### 输入
- 命令行参数（见 `python -m training.train -h`）
- `data/processed/X_{pv,bess}.npz`、`y_{pv,bess}.npz`、`meta_{pv,bess}.csv`
- `data/splits/{train,val,test}.csv`

### 输出
- `quantization/artifacts/cnn1d_{pv,bess}_best.pt` — 含：
    - `model_state_dict`
    - `epoch`, `val_macro_f1`, `val_accuracy`
    - `system_type`, `n_classes`, `label_classes`, `feature_stats`
    - `model_arch`, `in_channels`, `dropout`
- stdout：含每个 epoch 指标的 JSON 摘要

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 选择理由 |
|---|---|---|
| 训练前先 `FeatureStats.fit` | 在 BatchNorm 上依赖 | BN 仅归一化卷积输出；输入通道幅值悬殊（BESS 能量 vs 电压相差 5 数量级）会让前两层卷积梯度被吞掉 |
| stats 存进 checkpoint | 单独 json 文件 | 保证 .pt 自包含：拿到 .pt 就能复现推理 |
| AdamW + Cosine LR | SGD / OneCycle | 默认稳定；对 LR 不敏感；30 epoch 够用 |
| Macro F1 选 best | accuracy / loss | 类别不平衡时 macro F1 才反映"少数类没被忽略" |
| Early stopping patience=8 | 10 / 5 | 保守一点，防止小数据集波动误判 |

---

## 7. 反例（What NOT to put here）

- ❌ 模型架构定义（属于 `models/`）
- ❌ 推理后处理 / 严重度映射（属于 `inference/postprocess.py`）
- ❌ 评估报告生成（属于 `evaluation/`）
- ❌ ONNX 导出（属于 `quantization/`）

---

## 8. 教学注释（Why each step）

`train.py` 的关键步骤都加了行内注释解释为什么要这一步（rule §25/§26）：
特别是 `FeatureStats.fit` 必须只用 train 集，否则验证集泄漏会让 F1 数字虚高。

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `simulation/generate_dataset.py` 产出 npz/csv | `inference/onnx_runner.py` 加载 .pt 推理（S07） |
| `models/cnn1d.py` 提供 nn.Module | `quantization/quantize.py` 加载 .pt 做 INT8 量化（S08） |
| `api/schemas.py` 提供 SystemType / SplitName / 类标签元组 | `evaluation/run_eval.py` 加载 .pt 跑 test split（S08） |

---

## 10. 测试覆盖（Tests）

`tests/unit/test_training.py`（15 项）：

- LabelMap 双向映射
- TimeSeriesNpzDataset 形状 / 长度对齐 / 错误维度拒绝
- FeatureStats fit / round-trip / zero-variance 通道
- Dataset 应用标准化结果
- Inverse-frequency class weights 正确性
- WeightedCrossEntropyLoss / FocalLoss 数值正确性
- **Trainer 集成 smoke test**：在小型可分离数据集上 12 epoch 内 F1 ≥ 0.95

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 当前（n=2400, 30 epoch, CPU） |
|---|---|---|
| 训练时间 | < 5 min | ~17 s（PV）/ ~15 s（BESS） |
| 模型大小 | < 1 MB | ~190 KB |
| 内存占用 | < 1 GB | < 400 MB |

完整 51k 数据集预计训练时间 ~5 min（PV）/ ~3 min（BESS）。

---

## 12. 未来扩展（Future work）

- 加 `training/callbacks.py` 把 early stop / checkpoint 抽成回调
- 加 TensorBoard / WandB writer（保留可关闭）
- 微调入口（pruned / quantized 模型继续训练）
- Hyperparam search（optuna）

---

## 13. 运行示例（Usage example）

```bash
# 1. 先生成数据集
python -m simulation.generate_dataset --seed 42 --n-pv 1400 --n-bess 1000 \
    --n-pv-normal 200 --n-bess-normal 200

# 2. 训 PV
python -m training.train --system pv --epochs 30 --batch-size 64

# 3. 训 BESS
python -m training.train --system bess --epochs 30 --batch-size 64
```

输出 checkpoint 在 `quantization/artifacts/cnn1d_{pv,bess}_best.pt`。

---

## 14. 已知限制（Known limitations）

- BESS 在 n=200/类 时 val Macro F1 ~0.82；polish 阶段需扩大到 n≈3000/类才能稳定 ≥0.85
- 训练目前只支持 single-GPU/CPU；多 GPU 分布式延后到 polish 阶段
- 未做数据增强（噪声注入、时间裁剪）；如果 polish 阶段过拟合需要再加
