"""Build ``reports/final_report.pdf`` from ``reports/final_report.md``.

Component 8 deliverable. The master Markdown is hand-written
(``reports/final_report.md``); this script *only*:

1. Converts the Markdown to HTML via :mod:`mistune` (with GitHub-flavoured
   tables enabled);
2. Wraps it in a self-contained HTML document with an academic print
   stylesheet (A4, 11-pt serif body, monospace code, page numbers in the
   footer, page-break-before on every ``<h2>``);
3. Saves the HTML next to the Markdown so it is human-inspectable;
4. Invokes a headless Chromium / Chrome (auto-detected) with
   ``--print-to-pdf`` to produce the final PDF.

The Markdown's image references are relative to ``reports/`` so saving the
HTML there means the browser resolves them with no extra rewriting.

Reproduce::

    python scripts/render_final_report.py

Override Chrome explicitly::

    python scripts/render_final_report.py --chrome "C:/path/to/chrome.exe"

The exit code is non-zero whenever the PDF could not be produced (chrome
missing, mistune missing, HTML render error). The HTML is still written so
the user can open it in any browser and print to PDF manually.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_MD = _ROOT / "reports" / "final_report.md"
DEFAULT_HTML = _ROOT / "reports" / "final_report.html"
DEFAULT_PDF = _ROOT / "reports" / "final_report.pdf"

_PRINT_CSS = """
:root {
  --ink:        #1a1a1a;
  --muted:      #555;
  --rule:       #d5d5d5;
  --code-bg:    #f4f4f4;
  --table-head: #f6f6f6;
  --link:       #2E86AB;
}

@page {
  size: A4;
  margin: 18mm 18mm 22mm 18mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 9pt;
    color: #888;
  }
}

html, body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 11pt;
  line-height: 1.45;
  color: var(--ink);
  margin: 0 auto;
  max-width: 720px;
  text-rendering: optimizeLegibility;
  /* Georgia defaults to old-style figures (3/5/9 hang below baseline) — use lining nums. */
  font-variant-numeric: lining-nums;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

h1, h2, h3, h4 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: #111;
  line-height: 1.2;
  font-weight: 600;
}

h1 {
  font-size: 22pt;
  margin: 0 0 0.4em 0;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 0.2em;
}

h2 {
  font-size: 16pt;
  margin-top: 1.2em;
  padding-top: 0.2em;
  border-top: 1px solid var(--rule);
  page-break-before: always;
  break-before: page;
}

/* Don't break the page before the very first H2 (right after the abstract). */
h1 + hr + h2,
h1 + h2,
h1 ~ h2:first-of-type {
  page-break-before: auto;
  break-before: auto;
}

h3 { font-size: 13pt; margin-top: 1.1em; }
h4 { font-size: 11.5pt; margin-top: 0.8em; }

p { margin: 0.55em 0; text-align: justify; hyphens: auto; }

hr {
  border: none;
  border-top: 1px solid var(--rule);
  margin: 1.2em 0;
}

a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }

code, pre, kbd, samp {
  font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
}

code {
  background: var(--code-bg);
  padding: 0 0.25em;
  border-radius: 3px;
  font-size: 0.92em;
}

pre {
  background: var(--code-bg);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 0.6em 0.8em;
  overflow: auto;
  font-size: 9pt;
  line-height: 1.4;
  page-break-inside: avoid;
  break-inside: avoid;
}

pre code { background: transparent; padding: 0; border-radius: 0; }

blockquote {
  border-left: 3px solid var(--rule);
  margin: 0.6em 0 0.6em 0.2em;
  padding: 0.1em 0.8em;
  color: var(--muted);
  font-style: italic;
}

table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.7em 0;
  font-size: 9.5pt;
  font-variant-numeric: lining-nums tabular-nums;
  page-break-inside: avoid;
  break-inside: avoid;
}

th, td {
  border: 1px solid var(--rule);
  padding: 4px 7px;
  vertical-align: top;
  text-align: left;
}

thead th, table tr:first-child th {
  background: var(--table-head);
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-weight: 600;
  font-size: 9.5pt;
}

tr:nth-child(even) td { background: #fbfbfb; }

img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.6em auto;
  page-break-inside: avoid;
  break-inside: avoid;
}

ul, ol { padding-left: 1.4em; margin: 0.55em 0; }
li { margin: 0.2em 0; }

/* Title block — first H1 + first <p> serve as the title page. */
body > h1:first-of-type {
  text-align: center;
  border-bottom: 3px double var(--ink);
}

body > h1:first-of-type + p {
  text-align: center;
  color: var(--muted);
  margin-bottom: 1em;
}
"""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _render_markdown_to_html(md_text: str) -> str:
    try:
        import mistune  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "mistune is required (pip install mistune>=3.0); got: " + str(exc)
        ) from exc

    md = mistune.create_markdown(
        plugins=["table", "task_lists", "strikethrough", "url"],
        escape=False,
    )
    out = md(md_text)
    if isinstance(out, tuple):  # mistune occasionally returns (html, state)
        out = out[0]
    return out


def _wrap_html(body_html: str, *, title: str, generated: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_PRINT_CSS}</style>
<meta name="generator" content="scripts/render_final_report.py">
<meta name="generated-at" content="{generated}">
</head>
<body>
{body_html}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Chromium discovery + invocation
# ---------------------------------------------------------------------------


def _candidate_chrome_paths() -> list[Path]:
    """Probe common installation locations on Windows + macOS + Linux."""

    paths: list[Path] = []
    # Explicit overrides win.
    for env in ("CHROME_PATH", "GOOGLE_CHROME_PATH", "EDGE_PATH"):
        val = os.environ.get(env)
        if val:
            paths.append(Path(val))
    # PATH lookup.
    for exe in ("chrome", "google-chrome", "google-chrome-stable",
                "chromium", "chromium-browser", "msedge"):
        located = shutil.which(exe)
        if located:
            paths.append(Path(located))
    # Common Windows install dirs.
    for base in (
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ):
        p = Path(base)
        if p.exists():
            paths.append(p)
    # macOS.
    for base in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ):
        p = Path(base)
        if p.exists():
            paths.append(p)
    # De-duplicate while keeping order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve(strict=False)
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    return unique


def _print_to_pdf(
    *,
    chrome_path: Path,
    html_path: Path,
    pdf_path: Path,
    timeout_s: float = 120.0,
) -> None:
    """Run a headless Chromium-derived browser to convert HTML → PDF."""

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                f"Cannot overwrite {pdf_path}: file is open in another "
                f"application (e.g. PDF reader). Close it and re-run, or "
                f"use --pdf to write to a different path. Original error: {exc}"
            ) from exc

    # Use a throwaway user-data-dir so we never collide with the user's
    # real Chrome profile (avoids "profile locked" on Windows).
    with tempfile.TemporaryDirectory(prefix="agentpv-chrome-") as user_dir:
        # Use the modern `--headless=new` (Chrome 109+) for stable PDF output.
        # We deliberately keep --no-pdf-header-footer so the @page CSS rule
        # is the sole owner of header/footer styling.
        url = html_path.resolve().as_uri()
        # Chrome's headless mode resolves `--print-to-pdf` relative to its own
        # working directory (often a temp dir on Windows), so we always pass
        # an absolute path. Without this, custom --pdf overrides silently fail
        # with "Failed to write file" / system error 0x3.
        pdf_abs = pdf_path.resolve()
        cmd = [
            str(chrome_path),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--user-data-dir={user_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            f"--print-to-pdf={pdf_abs}",
            url,
        ]
        print(f"  -> chrome cmd: {cmd[0]} ... --print-to-pdf={pdf_abs.name}")
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=timeout_s,
        )

    if not pdf_path.exists() or pdf_path.stat().st_size < 1024:
        raise RuntimeError(
            f"PDF was not produced (or is suspiciously small). "
            f"chrome exit={proc.returncode}\n"
            f"stderr-tail: {proc.stderr.decode('utf-8', 'ignore')[-800:]}"
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _safe_relpath(path: Path) -> str:
    """Best-effort project-root-relative display path.

    ``Path.relative_to`` raises ``ValueError`` when ``path`` is outside the
    project root *or* mixes relative/absolute paths. For logging purposes we
    want something readable without crashing the run; fall back to the
    absolute path when relativisation fails.
    """

    try:
        return str(path.resolve().relative_to(_ROOT))
    except (ValueError, OSError):
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD,
                        help="Master Markdown source (default: %(default)s).")
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML,
                        help="Intermediate HTML output (default: %(default)s).")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF,
                        help="Final PDF output (default: %(default)s).")
    parser.add_argument("--chrome", type=Path, default=None,
                        help="Override Chromium / Chrome / Edge binary path.")
    parser.add_argument("--skip-pdf", action="store_true",
                        help="Render HTML only; do not invoke Chromium.")
    args = parser.parse_args()

    if not args.md.exists():
        parser.error(f"Master Markdown missing: {args.md}")

    md_text = _read_text(args.md)
    print(f"-> Loaded {_safe_relpath(args.md)} ({len(md_text):,} chars)")

    body_html = _render_markdown_to_html(md_text)
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    full_html = _wrap_html(
        body_html,
        title="AgentPV — Final Report",
        generated=generated,
    )
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(full_html, encoding="utf-8")
    print(
        f"-> Wrote {_safe_relpath(args.html)} "
        f"({args.html.stat().st_size:,} bytes)"
    )

    if args.skip_pdf:
        print("  (skipping PDF render — --skip-pdf passed)")
        return 0

    if args.chrome is not None:
        candidates = [args.chrome]
    else:
        candidates = _candidate_chrome_paths()

    if not candidates:
        print(
            "[warn] No Chromium / Chrome / Edge binary located.\n"
            "  Open the HTML in any browser -> File > Print > Save as PDF.\n"
            "  Tip: set CHROME_PATH or pass --chrome explicitly.",
            file=sys.stderr,
        )
        return 2

    print(f"-> Found {len(candidates)} browser candidate(s); using "
          f"{candidates[0]}")
    last_error: Exception | None = None
    for chrome in candidates:
        try:
            _print_to_pdf(
                chrome_path=chrome, html_path=args.html, pdf_path=args.pdf
            )
            print(
                f"[ok] Wrote {_safe_relpath(args.pdf)} "
                f"({args.pdf.stat().st_size:,} bytes)"
            )
            return 0
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            last_error = exc
            print(f"  [x] {chrome} failed: {exc}", file=sys.stderr)

    print(
        f"[warn] All browser candidates failed; last error: {last_error}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
