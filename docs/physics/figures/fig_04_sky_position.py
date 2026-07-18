#!/usr/bin/env python3
"""Position dependence of the dark sky and the van Rhijn airglow law."""

import numpy as np
import matplotlib.pyplot as plt

from figstyle import COLORS, DOUBLE, save
from sky_brightness import dark_sky_position_correction_mag, van_rhijn_factor


def main():
    beta = np.linspace(0.0, 90.0, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)

    for b_gal, color in ((90.0, COLORS["blue"]), (30.0, COLORS["green"]), (0.0, COLORS["red"])):
        dm = [dark_sky_position_correction_mag(0.0, bb, b_gal) for bb in beta]
        ax1.plot(beta, dm, color=color, label=fr"$|b| = {b_gal:.0f}^\circ$")
    ax1.set_xlabel(r"ecliptic latitude $|\beta|$ [deg]")
    ax1.set_ylabel(r"$\Delta m_{\rm sky}$ at zenith [mag arcsec$^{-2}$]")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 90)
    ax1.legend(title="galactic latitude")

    z = np.linspace(0.0, 90.0, 300)
    ax2.plot(z, van_rhijn_factor(z), color=COLORS["blue"], label="van Rhijn (1921)")
    ax2.plot(z, np.minimum(1.0 / np.cos(np.radians(np.clip(z, 0, 89.0))), 40), color=COLORS["red"],
             linestyle="--", label=r"linear $\sec z$ (pre-v10)")
    ax2.set_xlabel(r"zenith distance $z$ [deg]")
    ax2.set_ylabel("airglow enhancement factor")
    ax2.set_xlim(0, 90)
    ax2.set_ylim(1, 12)
    ax2.legend(loc="upper left")
    save(fig, "fig_04_sky_position")


if __name__ == "__main__":
    main()
