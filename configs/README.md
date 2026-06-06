# configs — 全局配置

本目录存放 **YAML 分层默认值** 与 **Python 加载逻辑**（`settings.py`），供 edge / agent / orchestrator / 脚本统一读取。

## 文件说明

| 文件 | 作用 |
|------|------|
| `base.yaml` | 全环境共用默认（路径、日志级别、模型文件名等）。 |
| `dev.yaml` | 开发环境覆盖（如 Ollama 地址、调试选项）。 |
| `test.yaml` | 单元测试用（确定性、尽量无外部网络）。 |
| `prod.yaml` | Docker / 生产形态默认值。 |
| `settings.py` | **唯一入口**：`get_settings()` 合并环境变量 → `.env` → `APP_ENV` 对应 yaml → 代码默认值。 |

## 加载优先级（从高到低）

1. 环境变量前缀 `AGENTPV_*`（Pydantic Settings）  
2. 项目根 `.env`  
3. `configs/<APP_ENV>.yaml`（`APP_ENV` 缺省为 `dev`）  
4. `configs/base.yaml`  
5. `Settings` 类内 `Field` 默认值  

## 常用环境变量（示例）

- `APP_ENV`：`dev` | `test` | `prod`  
- `AGENTPV_LLM_BACKEND`：`mock` | `ollama` 等  
- `AGENTPV_OLLAMA_*`：本机 Ollama 主机与模型名（见 `dev.yaml`）  

评测用裁判见 `agent_eval/README.md`（`AGENTPV_JUDGE_*`）。

## 代码引用

```python
from configs.settings import get_settings

settings = get_settings()
print(settings.knowledge_base_dir)
```
