"""
build_pdf.py
------------
Rigenera i due PDF nella root del progetto a partire dai sorgenti Markdown
in questa cartella:

    docs/guida-al-codice.md        -> tiny-llm-demo-guida-al-codice.pdf
    docs/principi-rete-neurale.md  -> tiny-llm-demo-principi-rete-neurale.pdf

Pipeline: Markdown -> HTML (libreria `markdown` + Pygments per la sintassi)
-> PDF (Chrome o Edge in modalità headless, già presenti su quasi tutti i PC).

Uso (dalla root del progetto):
    pip install markdown pygments
    python docs/build_pdf.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

DOCS_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCS_DIR.parent

DOCUMENTI = {
    "guida-al-codice.md": "tiny-llm-demo-guida-al-codice.pdf",
    "principi-rete-neurale.md": "tiny-llm-demo-principi-rete-neurale.pdf",
}

# Candidati per la conversione HTML -> PDF, in ordine di preferenza.
BROWSER_CANDIDATES = [
    "chrome",
    "google-chrome",
    "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm; }

html { font-size: 10.5pt; }
body {
  font-family: Georgia, "Times New Roman", "DejaVu Serif", serif;
  color: #1a1a1a;
  line-height: 1.5;
  text-align: justify;
  margin: 0;
}

h1 { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 2em; color: #1f3a5f;
     margin: 0 0 0.2em 0; line-height: 1.15; }
h2 { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 1.45em; color: #1f3a5f;
     margin: 1.6em 0 0.5em 0; padding-bottom: 0.15em; border-bottom: 2px solid #c9d6e6;
     page-break-after: avoid; break-after: avoid; }
h3 { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 1.12em; color: #2c5282;
     margin: 1.3em 0 0.4em 0; page-break-after: avoid; break-after: avoid; }
p.subtitle { font-style: italic; color: #555; margin: 0 0 1.2em 0; }

p { margin: 0.55em 0; orphans: 3; widows: 3; }
ul, ol { margin: 0.4em 0 0.6em 0; padding-left: 1.6em; }
li { margin: 0.25em 0; }
li > ul { margin: 0.2em 0; }
blockquote { margin: 0.8em 1.5em; padding-left: 0.8em; border-left: 3px solid #c9d6e6;
             color: #444; font-style: italic; }
a { color: #2c5282; text-decoration: none; }
li:has(> a) { text-align: left; }
strong { font-weight: bold; }

code, pre {
  font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Menlo", monospace;
  font-size: 0.86em;
}
p code, li code, td code, h2 code, h3 code {
  background: #f2f4f7; padding: 0.05em 0.3em; border-radius: 3px; white-space: nowrap;
}
pre {
  background: #f7f8fa; border-left: 3px solid #4a7ab5; padding: 0.7em 1em;
  margin: 0.7em 0 0.9em 0; line-height: 1.4; overflow: hidden; white-space: pre-wrap;
  page-break-inside: avoid; break-inside: avoid;
}
pre code { background: none; padding: 0; }

table { border-collapse: collapse; width: 100%; margin: 0.8em 0 1em 0; font-size: 0.93em;
        page-break-inside: avoid; }
th, td { border: 1px solid #d0d7e2; padding: 0.35em 0.6em; vertical-align: top; text-align: left; }
th { background: #eef2f7; font-family: "Segoe UI", Helvetica, Arial, sans-serif; }

/* Indice generato da [TOC] */
.toc { background: #f7f8fa; border: 1px solid #e1e6ee; padding: 0.8em 1.2em; margin: 1em 0 1.5em 0;
       font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 0.92em; page-break-inside: avoid; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc ul ul { padding-left: 1.3em; }
.toc li { margin: 0.12em 0; }
.toc a { color: #1f3a5f; }
"""


def trova_browser() -> str:
    for candidato in BROWSER_CANDIDATES:
        trovato = shutil.which(candidato) or (candidato if Path(candidato).exists() else None)
        if trovato:
            return trovato
    sys.exit(
        "Nessun Chrome/Edge trovato per la conversione in PDF. "
        "Installa Google Chrome o Microsoft Edge, oppure aggiungi il percorso in BROWSER_CANDIDATES."
    )


def markdown_to_html(sorgente: Path) -> str:
    md = markdown.Markdown(
        extensions=["fenced_code", "codehilite", "tables", "toc", "sane_lists"],
        extension_configs={
            "codehilite": {"guess_lang": False, "noclasses": False},
            "toc": {"toc_depth": "2-3", "title": "Indice"},
        },
    )
    corpo = md.convert(sorgente.read_text(encoding="utf-8"))
    titolo = sorgente.stem
    pygments_css = HtmlFormatter(style="default").get_style_defs(".codehilite")
    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>{titolo}</title>
<style>{CSS}{pygments_css}</style>
</head>
<body>
{corpo}
</body>
</html>"""


def html_to_pdf(browser: str, html_path: Path, pdf_path: Path) -> None:
    comando = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    subprocess.run(comando, check=True, capture_output=True)


def main() -> None:
    browser = trova_browser()
    build_dir = DOCS_DIR / "_build"
    build_dir.mkdir(exist_ok=True)

    for nome_md, nome_pdf in DOCUMENTI.items():
        sorgente = DOCS_DIR / nome_md
        html_path = build_dir / (sorgente.stem + ".html")
        pdf_path = ROOT_DIR / nome_pdf

        html_path.write_text(markdown_to_html(sorgente), encoding="utf-8")
        html_to_pdf(browser, html_path, pdf_path)
        print(f"{nome_md} -> {pdf_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
