# utils — 跨模块通用工具

与业务无关的**横切能力**：路径常量、日志、随机种子、计时。  
所有服务与脚本应优先使用此处定义，避免在业务代码里硬编码路径字符串（项目规则 §4）。

## 模块说明

| 模块 | 作用 |
|------|------|
| `paths.py` | `PROJECT_ROOT`、`DATA_DIR`、`ARTIFACTS_DIR`、`REPORTS_DIR`、`KB_DOCS_DIR` 等。 |
| `logging_config.py` | 统一 `structlog` / 标准库 logging 配置与 `get_logger(__name__)`。 |
| `seeds.py` | 固定 `numpy` / `torch` / `random` 种子，保证可复现。 |
| `timing.py` | 简单计时器上下文，用于 benchmark 与诊断。 |

## 使用建议

- 读写数据、报告、ONNX 路径：一律基于 `utils.paths` 或 `get_settings().project_root`。  
- 新脚本入口：首行配置日志后再打业务日志。  
- 训练 / 仿真入口：在创建随机数生成器前调用 `seeds` 中的固定函数（与 `simulation`、`training` 文档一致）。
