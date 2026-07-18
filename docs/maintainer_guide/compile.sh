#!/bin/bash
# compile.sh — Full LaTeX build
set -e
DOC="main"
lualatex --interaction=nonstopmode "$DOC.tex"
lualatex --interaction=nonstopmode "$DOC.tex"
echo "=== Done: $DOC.pdf ==="
