# AgentPV Component 3 — BESS `onnx_int8` evaluation summary

- **Variant**: `onnx_int8`
- **Model artefact**: `C:\Users\Mansycc\Desktop\omar\quantization\artifacts\cnn1d_bess_int8.onnx`
- **Split**: `test`
- **Samples**: 3395
- **Classes**: 5

## Aggregate metrics

- Accuracy: **0.7016**
- Macro-F1: **0.7016** ⚠️ below 0.90 target
- Weighted-F1: 0.6918

### BESS classification report — split=`test` (N=3395, n_classes=5)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `BESS_Normal` | 0.5055 | 0.5392 | 0.5218 | 766 |
| `Capacity_fade` | 0.9747 | 0.9732 | 0.9740 | 634 |
| `Internal_resistance_increase` | 0.5378 | 0.7616 | 0.6305 | 709 |
| `Thermal_anomaly` | 0.5969 | 0.2938 | 0.3938 | 650 |
| `Cell_imbalance` | 1.0000 | 0.9764 | 0.9881 | 636 |

| Aggregate | Value |
|---|---:|
| Accuracy | 0.7016 |
| Macro-F1 | **0.7016** |
| Weighted-F1 | 0.6918 |

## Confusion matrix

![confusion matrix](confusion_matrix.png)

## CPU latency benchmark

- Runs: 1000 (warm-up 50, batch=1)
- Mean: 0.061 ms
- p50: 0.056 ms
- p95: **0.078 ms** ✅ ≤ 100 ms
- p99: 0.111 ms
- min / max: 0.050 ms / 1.323 ms

## Model size

- File: `C:\Users\Mansycc\Desktop\omar\quantization\artifacts\cnn1d_bess_int8.onnx`
- Size: 59.50 KiB (0.0581 MiB)
- Budget: 50 MiB — ✅ within budget