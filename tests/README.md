# tests — 自动化测试

本目录保证 **契约、训练管线、评测、智能体编排、HTTP 服务** 等在 CI 与本地可重复验证。  
默认使用 `APP_ENV=test`（见 `conftest.py`），避免写真实 `data/version.txt` 或外网调用。

## 结构

| 目录 | 内容 |
|------|------|
| `conftest.py` | 全局 fixture、环境变量设置。 |
| `unit/` | 单元测试：单模块、无完整服务栈。 |
| `integration/` | 集成测试：可能启动 ASGI 应用或使用 `httpx`/`TestClient`。 |

## 运行

```powershell
cd <项目根目录>
pip install -e ".[dev]"
pytest tests -q                 # 全量
pytest tests/unit/test_agent_eval.py -q   # 单文件示例
```

## 注意

- 部分集成测试在缺少 ONNX 权重或特定环境时会 **skip** 或走 mock，属预期行为。  
- 不要依赖测试产生的 `data/` 覆盖你本地完整数据集；大规模数据请用 `simulation.generate_dataset` 单独生成。
