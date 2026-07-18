#!/bin/bash
# compile.sh — Full LaTeX build (physics documentation)
set -e
DOC="main"

echo "=== Figures (regenerate from the SPETC source) ==="
( cd figures && for f in fig_0*.py; do python3 "$f"; done )

echo "=== Pass 1: lualatex ==="
lualatex --interaction=nonstopmode "$DOC.tex"

echo "=== BibTeX ==="
bibtex "$DOC"

echo "=== Pass 2: lualatex ==="
lualatex --interaction=nonstopmode "$DOC.tex"

echo "=== Pass 3: lualatex ==="
lualatex --interaction=nonstopmode "$DOC.tex"

echo "=== Done: $DOC.pdf ==="
