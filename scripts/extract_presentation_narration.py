#!/usr/bin/env python3
"""Extract bilingual speaker notes from ppt制作指南.md into ppt旁白.md.

Usage (repo root):
    python scripts/extract_presentation_narration.py
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUIDE = ROOT / "docs" / "ppt制作指南.md"
DEFAULT_OUT = ROOT / "docs" / "ppt旁白.md"


@dataclass
class NarrationSpec:
    slide_id: str
    title: str
    notes_cn: str
    notes_en: str
    qa: str


def parse_narrations(path: Path) -> list[NarrationSpec]:
    text = path.read_text(encoding="utf-8")
    start = text.find("### Slide 1 ·")
    if start == -1:
        raise ValueError("Could not find slide section in guide")
    end = text.find("\n## 四、")
    if end == -1:
        end = text.find("\n## 五、")
    if end == -1:
        end = len(text)
    text = text[start:end]

    headers = list(
        re.finditer(r"^### ((?:Slide \d+|Appendix A\d+)) · (.+)$", text, re.MULTILINE)
    )
    specs: list[NarrationSpec] = []
    for i, m in enumerate(headers):
        block_start = m.start()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[block_start:block_end]
        slide_id = m.group(1)
        title = m.group(2).strip()

        cn_m = re.search(
            r"\*\*旁白（中文，详细）\*\*\s*\n(.*?)(?=\n\*\*Narration|\n\*\*Anticipated|\n---|\Z)",
            block,
            re.DOTALL,
        )
        en_m = re.search(
            r"\*\*Narration \(English, detailed\)\*\*\s*\n(.*?)(?=\n\*\*Anticipated|\n---|\Z)",
            block,
            re.DOTALL,
        )
        qa_m = re.search(r"\*\*Anticipated Q&A\*\*:\s*(.+?)(?=\n---|\Z)", block, re.DOTALL)

        specs.append(
            NarrationSpec(
                slide_id=slide_id,
                title=title,
                notes_cn=_clean(cn_m.group(1) if cn_m else ""),
                notes_en=_clean(en_m.group(1) if en_m else ""),
                qa=_clean(qa_m.group(1) if qa_m else ""),
            )
        )
    return specs


def _clean(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", s.strip())


def render_markdown(specs: list[NarrationSpec], guide: Path) -> str:
    main = sum(1 for s in specs if s.slide_id.startswith("Slide"))
    appendix = len(specs) - main
    today = date.today().isoformat()

    lines = [
        "# AgentPV 答辩旁白（Speaker Scripts）",
        "",
        f"> 从 [`{guide.name}`]({guide.name}) 提取 · **{len(specs)} 页**"
        f"（正文 {main} + 附录 {appendix}）· 生成日期 {today}",
        "",
        "## 怎么用",
        "",
        "- **幻灯片上只有英文**；本文件是口头讲解稿，不要贴到 PPT 里。",
        "- 每页先 **中文旁白**（答辩主用），再 **English Narration**（双语答辩或练习用）。",
        "- 部分页含 **预设 Q&A**，可在追问时扫一眼。",
        "- 网页演示不在本 deck 内，见 [`docs/网页演示指南.md`](docs/网页演示指南.md)。",
        "- 更新 `ppt制作指南.md` 后重新生成：`python scripts/extract_presentation_narration.py`",
        "",
        "---",
        "",
    ]

    for i, spec in enumerate(specs, start=1):
        lines.append(f"## {spec.slide_id} · {spec.title}")
        lines.append("")
        lines.append(f"<!-- 页码 {i} / {len(specs)} -->")
        lines.append("")
        lines.append("### 中文旁白")
        lines.append("")
        lines.append(spec.notes_cn if spec.notes_cn else "_（无）_")
        lines.append("")
        lines.append("### English Narration")
        lines.append("")
        lines.append(spec.notes_en if spec.notes_en else "_（none）_")
        lines.append("")
        if spec.qa:
            lines.append("### 预设 Q&A")
            lines.append("")
            lines.append(spec.qa)
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract PPT narration to ppt旁白.md")
    ap.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.guide.is_file():
        raise SystemExit(f"Guide not found: {args.guide}")

    specs = parse_narrations(args.guide)
    args.out.write_text(render_markdown(specs, args.guide), encoding="utf-8")
    print(f"[ok] Wrote {args.out} ({len(specs)} slides)")


if __name__ == "__main__":
    main()
