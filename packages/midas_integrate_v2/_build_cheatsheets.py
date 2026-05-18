"""Render every cheat-sheet markdown to a printable PDF.

Cheat sheets live in two places: the canonical ``dev/paper/meetings/``
folder (for SRS-group materials) and ``dev/softmatter_drift/`` (for
TRR-group materials). Techniques cheat sheets live inside each
meeting pack (since they were written meeting-specific). This script
renders all of them into ``.pdf`` files alongside their source ``.md``
and copies the resulting PDFs into the two meeting packs so they can
be printed straight from the pack folders.

Run:

    python _build_cheatsheets.py

Requires pandoc and a TeX Live installation with xelatex. Uses
TeX Gyre Termes as the main font (has full Greek coverage for σ, Γ,
β, η, χ², etc.). Page geometry is letter / 0.65-inch margins / 10pt
body, tuned for a single-column reference printout.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRS_PACK = ROOT / "dev" / "meeting_packs" / "SRS_2026-05-13"
TRR_PACK = ROOT / "dev" / "meeting_packs" / "TRR_2026-05-13"

# Source-of-truth cheat sheets that get rendered in their canonical
# location AND copied into the appropriate meeting pack.
CHEATSHEETS = [
    {
        "src": ROOT / "dev" / "paper" / "meetings" / "SRS_TLDR.md",
        "title": "SRS group meeting — TL;DR",
        "copy_to": SRS_PACK / "briefs",
    },
    {
        "src": ROOT / "dev" / "paper" / "meetings" / "SRS_QA_cheatsheet.md",
        "title": "SRS group meeting — Q&A cheat sheet",
        "copy_to": SRS_PACK / "briefs",
    },
    {
        "src": ROOT / "dev" / "paper" / "meetings" / "SRS_presenter_notes.md",
        "title": "SRS slides — per-slide presenter notes",
        "copy_to": SRS_PACK / "briefs",
    },
    {
        "src": ROOT / "dev" / "paper" / "meetings" / "DEMO_cheatsheet.md",
        "title": "MIDAS demo cheat sheet",
        "copy_to": SRS_PACK / "briefs",
    },
    {
        "src": ROOT / "dev" / "softmatter_drift" / "TRR_TLDR.md",
        "title": "TRR group meeting — TL;DR",
        "copy_to": TRR_PACK / "briefs",
    },
    {
        "src": ROOT / "dev" / "softmatter_drift" / "TRR_QA_cheatsheet.md",
        "title": "TRR group meeting — Q&A cheat sheet",
        "copy_to": TRR_PACK / "briefs",
    },
    {
        "src": ROOT / "dev" / "softmatter_drift" / "TRR_presenter_notes.md",
        "title": "TRR slides — per-slide presenter notes",
        "copy_to": TRR_PACK / "briefs",
    },
    # Pack-only cheat sheets (the techniques primers written for each meeting).
    {
        "src": SRS_PACK / "briefs" / "SRS_techniques_cheatsheet.md",
        "title": "SRS techniques cheat sheet — for an HEDM person meeting the SRS group",
        "copy_to": None,
    },
    {
        "src": TRR_PACK / "briefs" / "TRR_techniques_cheatsheet.md",
        "title": "TRR techniques cheat sheet — for an HEDM person meeting the TRR group",
        "copy_to": None,
    },
]


# Pandoc settings — letter paper, tight margins, body font with Greek.
PANDOC_OPTS = [
    "--pdf-engine=xelatex",
    "-V", "papersize=letter",
    "-V", "geometry:margin=0.65in",
    "-V", "fontsize=10pt",
    "-V", "linkcolor=blue",
    "-V", "urlcolor=blue",
    "-V", "colorlinks=true",
    "-V", "mainfont=texgyretermes-regular.otf",
    "-V", "monofont=Menlo",
    "-V", "boldfont=texgyretermes-bold.otf",
    "-V", "italicfont=texgyretermes-italic.otf",
    "-V", "bolditalicfont=texgyretermes-bolditalic.otf",
    "-H", "/dev/stdin",   # extra LaTeX preamble piped in below
]

PREAMBLE = r"""
\usepackage{microtype}
\usepackage{setspace}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries\color{black!75}}{}{0pt}{}
\titlespacing*{\section}{0pt}{0.9em}{0.25em}
\titleformat{\subsection}{\normalsize\bfseries\color{black!75}}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{0.65em}{0.20em}
\renewcommand{\arraystretch}{1.10}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textsf{MIDAS · \jobname}}
\fancyhead[R]{\small\textsf{2026-05-13}}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
"""


def render_one(src_md: Path, title: str) -> Path | None:
    """Render one cheat-sheet .md → .pdf alongside the source."""
    if not src_md.exists():
        print(f"  MISSING: {src_md}")
        return None
    out_pdf = src_md.with_suffix(".pdf")
    # LaTeX title metadata is processed as LaTeX, so & / % / # need
    # escaping. Easiest: route through pandoc's frontmatter mechanism.
    safe_title = title.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    cmd = [
        "pandoc", str(src_md),
        "-V", f"title={safe_title}",
        *PANDOC_OPTS,
        "--resource-path", str(src_md.parent),
        "-o", str(out_pdf),
    ]
    try:
        subprocess.run(cmd, input=PREAMBLE, text=True,
                        check=True, capture_output=True)
        print(f"  wrote {out_pdf.relative_to(ROOT)}")
        return out_pdf
    except subprocess.CalledProcessError as e:
        print(f"  FAIL {src_md.name}: {e.stderr[-600:] if e.stderr else e}")
        return None


def main() -> int:
    print("Rendering cheat sheets to printable PDFs...\n")
    n_ok = 0
    n_fail = 0
    for spec in CHEATSHEETS:
        src = spec["src"]
        out_pdf = render_one(src, spec["title"])
        if out_pdf is None:
            n_fail += 1
            continue
        n_ok += 1
        copy_to_dir = spec["copy_to"]
        if copy_to_dir is not None:
            copy_to_dir.mkdir(parents=True, exist_ok=True)
            dst = copy_to_dir / out_pdf.name
            shutil.copy2(out_pdf, dst)
            print(f"    → {dst.relative_to(ROOT)}")
    print(f"\nDone. {n_ok} rendered, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
