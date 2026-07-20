#!/bin/bash
# compile.sh — preprint rendering of the RNAAS note.
# The AAS submission source is spetc_rnaas_aastex.tex (needs aastex631.cls).
set -e
DOC="main"
lualatex --interaction=nonstopmode "$DOC.tex"
bibtex "$DOC"
lualatex --interaction=nonstopmode "$DOC.tex"
lualatex --interaction=nonstopmode "$DOC.tex"
echo "=== Done: $DOC.pdf ==="
