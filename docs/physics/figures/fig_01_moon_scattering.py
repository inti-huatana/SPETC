#!/usr/bin/env python3
"""Krisciunas & Schaefer (1991) scattering function and the historical bug."""

import numpy as np
import matplotlib.pyplot as plt

from figstyle import COLORS, SINGLE, save
from sky_brightness import moonlight_scattering_function


def main():
    rho = np.linspace(5.0, 150.0, 500)
    correct = moonlight_scattering_function(rho)
    buggy = 10.0 ** (5.36 * (1.06 + np.cos(np.radians(rho)) ** 2)) + 10.0 ** (6.15 - rho / 40.0)

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.semilogy(rho, correct, color=COLORS["blue"],
                label=r"K\&S (1991): $10^{5.36}(1.06+\cos^2\rho)+10^{6.15-\rho/40}$")
    ax.semilogy(rho, buggy, color=COLORS["red"], linestyle="--",
                label=r"misparenthesized $10^{5.36(1.06+\cos^2\rho)}+\ldots$ (pre-v10)")
    ax.set_xlabel(r"Moon--target separation $\rho$ [deg]")
    ax.set_ylabel(r"scattering function $f(\rho)$")
    ax.set_xlim(5, 150)
    ax.legend(loc="upper right", fontsize=6.5)
    save(fig, "fig_01_moon_scattering")


if __name__ == "__main__":
    main()
