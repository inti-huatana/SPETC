"""Shared publication style for all SPETC documentation figures (PNG output)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The figure scripts import the SPETC physics modules directly, so every
# curve in the documentation is produced by the released code itself.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Okabe-Ito colourblind-safe palette.
COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "black": "#000000",
}

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 9, "font.size": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.2, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 300, "savefig.dpi": 300,
    "legend.frameon": False,
})

SINGLE = (3.46, 2.60)   # 88 mm single column
DOUBLE = (7.09, 3.20)   # 180 mm double column


def save(fig, name):
    out = Path(__file__).resolve().parent / f"{name}.png"
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
