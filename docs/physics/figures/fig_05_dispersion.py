#!/usr/bin/env python3
"""Filippenko (1982) differential refraction and the fixed-slit loss."""

import numpy as np
import matplotlib.pyplot as plt

from figstyle import COLORS, DOUBLE, save
from etc_physics import psf_slit_throughput
from observing_conditions import differential_refraction_arcsec

REFERENCE_AA = 5500.0
SEEING = 1.5
SLIT = 1.5


def main():
    wave = np.linspace(3500.0, 9500.0, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)

    for x, color in ((1.2, COLORS["blue"]), (1.5, COLORS["green"]), (2.0, COLORS["red"])):
        dr = differential_refraction_arcsec(wave, REFERENCE_AA, x)
        ax1.plot(wave / 1e4, dr, color=color, label=f"$X={x:.1f}$")
        loss = psf_slit_throughput(SLIT, SEEING, "gaussian", offset_arcsec=np.abs(dr))
        centre = psf_slit_throughput(SLIT, SEEING, "gaussian")
        ax2.plot(wave / 1e4, loss / centre, color=color, label=f"$X={x:.1f}$")

    ax1.axhline(0, color="0.7", linewidth=0.6)
    ax1.set_xlabel(r"wavelength [$\mu$m]")
    ax1.set_ylabel(r"$\Delta R(\lambda,5500\,\mathrm{\AA})$ [arcsec]")
    ax1.legend()

    ax2.set_xlabel(r"wavelength [$\mu$m]")
    ax2.set_ylabel("relative slit throughput")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    save(fig, "fig_05_dispersion")


if __name__ == "__main__":
    main()
