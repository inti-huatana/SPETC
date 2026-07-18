# SPETC v10 documentation

Three LuaLaTeX documents, each self-contained in its folder with a
`compile.sh` build script (lualatex + bibtex where needed):

- `physics/` -> `SPETC_physics.pdf` — scientific documentation: every model
  and formula of the ETC, with bibliography, tables, and publication-quality
  PNG figures. The figures are generated from the released SPETC code by the
  scripts in `physics/figures/` (`python3 fig_0X_*.py`; `figstyle.py` holds
  the shared style and puts the repository root on `sys.path`).
- `user_guide/` -> `SPETC_user_guide.pdf` — end-user operating guide.
- `maintainer_guide/` -> `SPETC_maintainer_guide.pdf` — code structure,
  module reference, data formats, modification recipes, testing, and
  Windows/macOS porting and packaging.

Requirements to rebuild: TeX Live with lualatex, tex-gyre fonts, siunitx,
natbib; Python with the SPETC requirements for the figures.
