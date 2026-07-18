#!/usr/bin/env python3
"""Gaussian versus Moffat encircled energy and decentred slit coupling."""

import numpy as np
import matplotlib.pyplot as plt

from figstyle import COLORS, DOUBLE, save
from etc_physics import psf_encircled_energy, psf_slit_throughput

SEEING = 1.0  # arcsec FWHM


def main():
    radius = np.linspace(0.05, 3.0, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)

    for model, beta, color, label in (
            ("gaussian", None, COLORS["blue"], "Gaussian"),
            ("moffat", 2.5, COLORS["red"], r"Moffat $\beta=2.5$"),
            ("moffat", 4.7, COLORS["green"], r"Moffat $\beta=4.7$")):
        ee = [psf_encircled_energy(r, SEEING, model, beta or 2.5) for r in radius]
        ax1.plot(radius, ee, color=color, label=label)
    ax1.set_xlabel(r"aperture radius / seeing FWHM")
    ax1.set_ylabel("encircled energy")
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc="lower right")

    offset = np.linspace(0.0, 2.0, 300)
    for model, beta, color, label in (
            ("gaussian", None, COLORS["blue"], "Gaussian"),
            ("moffat", 2.5, COLORS["red"], r"Moffat $\beta=2.5$"),
            ("moffat", 4.7, COLORS["green"], r"Moffat $\beta=4.7$")):
        t = psf_slit_throughput(1.0, SEEING, model, beta or 2.5, offset_arcsec=offset)
        ax2.plot(offset, t, color=color, label=label)
    ax2.set_xlabel(r"PSF decentre / seeing FWHM (1$''$ slit, 1$''$ seeing)")
    ax2.set_ylabel("slit throughput")
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 0.85)
    ax2.legend(loc="upper right")
    save(fig, "fig_03_psf")


if __name__ == "__main__":
    main()
