"""Gera MODEL_CARD.pdf e ML_CANVAS.pdf a partir do Markdown (HTML estilizado -> Chrome headless)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

from src.config import PROJECT_ROOT
from src.logger import get_logger

logger = get_logger(__name__)
DOCS = ["MODEL_CARD", "ML_CANVAS"]
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
]
CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;font-size:11pt;line-height:1.45;color:#1d1d1f;margin:0}
h1{font-size:20pt;border-bottom:3px solid #2f5d8a;padding-bottom:6px;color:#2f5d8a}
h2{font-size:14pt;color:#2f5d8a;margin-top:18px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:9.5pt}
th,td{border:1px solid #c9d3de;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#eaf0f6}
blockquote{border-left:4px solid #b26a00;background:#fff6e6;margin:10px 0;padding:6px 12px}
code{background:#f1f3f5;padding:1px 4px;border-radius:3px;font-size:9.5pt}
@page{size:A4;margin:16mm}
"""


def render_html(text: str, title: str) -> str:
    """Converte Markdown (com tabelas) em uma página HTML estilizada."""
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    return (
        f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>{title}</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def find_chrome() -> str | None:
    """Localiza um navegador com modo headless (variável CHROME_PATH tem prioridade)."""
    env = os.environ.get("CHROME_PATH")
    candidates = ([env] if env else []) + CHROME_CANDIDATES
    return next((c for c in candidates if Path(c).exists() or shutil.which(c)), None)


def print_pdf(chrome: str, html_path: Path, pdf_path: Path) -> None:
    """Imprime o HTML em PDF com o Chrome headless."""
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True, timeout=120,
    )  # fmt: skip


def main() -> int:
    """Gera os PDFs em docs/. Retorna 1 se não houver navegador."""
    chrome = find_chrome()
    if chrome is None:
        logger.error("navegador_nao_encontrado", extra={"hint": "defina CHROME_PATH"})
        return 1
    docs = PROJECT_ROOT / "docs"
    for name in DOCS:
        source = docs / f"{name}.md"
        html_path = docs / f"{name}.html"
        html_path.write_text(
            render_html(source.read_text(encoding="utf-8"), name.replace("_", " ").title()),
            encoding="utf-8",
        )
        print_pdf(chrome, html_path, docs / f"{name}.pdf")
        html_path.unlink()
        logger.info("pdf_gerado", extra={"doc": name})
    return 0


if __name__ == "__main__":
    sys.exit(main())
