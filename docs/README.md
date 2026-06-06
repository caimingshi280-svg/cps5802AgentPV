# docs — 对外文档与模式

与「运行时代码」并列的**说明性资产**：数据卡片、答辩材料、复现指南等。

## 答辩 / 作业

| 文件 | 说明 |
|------|------|
| `复现指南.md` | 全链路复现命令与自检清单 |
| `ppt制作指南.md` | PPT 逐页英文内容 + 中英旁白源稿 |
| `ppt旁白.md` | 仅旁白（主汇报 Slide 1–30） |
| `Q&A.md` | 答辩 Q&A（中英 + 对应幻灯片） |
| `网页演示指南.md` | Streamlit 现场演示步骤 |
| `文件解读.md` | 仓库目录与文件索引 |
| `开发记录.md` | 开发过程日志 |
| `AgentPV-项目方案.md` | 模块设计方案 |
| `assignment.md` / `assignmenchinese.md` | 课程作业原文 |

## 数据与契约

| 文件 | 说明 |
|------|------|
| `data_card.md` | **Data Card**（Component 1 交付物） |
| `alert_schema.json` | 边缘 → 云端告警 JSON Schema |

## 维护建议

- 修改仿真参数或类别后，同步更新 `data_card.md`。  
- 修改 `ppt制作指南.md` 后运行：`python scripts/extract_presentation_narration.py`  
- 生成 PPT：`python scripts/render_presentation.py --verify`
