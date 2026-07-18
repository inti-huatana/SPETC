#!/usr/bin/env python3
"""Pickering (2002) airmass against plane-parallel sec z."""

import numpy as np
import matplotlib.pyplot as plt

from figstyle import COLORS, DOUBLE, save
from ephemeris import pickering_airmass


def main():
    alt = np.linspace(3.0, 90.0, 500)
    secz = 1.0 / np.sin(np.radians(alt))
    pick = pickering_airmass(alt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
    ax1.plot(alt, secz, color=COLORS["red"], linestyle="--", label=r"plane-parallel $\sec z$")
    ax1.plot(alt, pick, color=COLORS["blue"], label="Pickering (2002)")
    ax1.set_xlabel("apparent altitude [deg]")
    ax1.set_ylabel("airmass $X$")
    ax1.set_xlim(3, 90)
    ax1.set_ylim(1, 15)
    ax1.axvline(5.0, color="0.6", linewidth=0.7, linestyle=":")
    ax1.annotate("ETC 5$^\\circ$ cutoff", xy=(5.5, 12.5), fontsize=7, color="0.35")
    ax1.legend()

    ax2.plot(alt, 100.0 * (secz - pick) / pick, color=COLORS["green"])
    ax2.set_xlabel("apparent altitude [deg]")
    ax2.set_ylabel(r"$(\sec z - X_{\rm P})/X_{\rm P}$ [%]")
    ax2.set_xlim(3, 40)
    ax2.axvline(5.0, color="0.6", linewidth=0.7, linestyle=":")
    save(fig, "fig_02_airmass")


if __name__ == "__main__":
    main()
