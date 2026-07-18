#!/usr/bin/env python3
"""Broad-band S/N versus magnitude: full noise budget and the scintillation limit.

Runs the released PhotometryETC on the 358 mm reference configuration.
"""

import numpy as np
import matplotlib.pyplot as plt

from figstyle import COLORS, SINGLE, save
from detector import Detector
from photometry import PhotometryETC

T_EXP = 60.0


def build_etc():
    wave = np.linspace(3500.0, 10500.0, 1000)
    template = np.column_stack((wave, np.full_like(wave, 3.60e-9)))
    band = np.column_stack((wave, ((wave >= 5000.0) & (wave <= 6000.0)).astype(float)))
    qe = np.column_stack((wave, np.full_like(wave, 0.90)))
    atmo_curve = np.column_stack((wave, np.full_like(wave, 0.85)))
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.70,
                 "focal_length_mm": 2000.0}
    detector = Detector(13.5, 2.5, 80000.0, 16, read_noise_e=5.0, dark_current_e_s_pix=0.02)
    atmosphere = {"airmass": 1.2, "seeing_arcsec": 1.5, "transmission_curve": atmo_curve,
                  "psf_model": "moffat", "moffat_beta": 2.5, "elevation_m": 1366.0}
    sky = {"sky_mag": 21.2, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True,
           "aperture_radius_arcsec": 2.0}
    return PhotometryETC(telescope, detector, atmosphere, sky), template, band, qe


def main():
    etc, template, band, qe = build_etc()
    mags = np.arange(2.0, 18.1, 0.5)
    full, poisson = [], []
    for mag in mags:
        r = etc.compute_photometry_single(template, band, qe, mag, T_EXP)
        full.append(r["snr"])
        s = r["photons_source_es"] * T_EXP
        b = r["photons_sky_es"] * T_EXP
        d = etc.detector.dark_current_e_s_pix * r["n_pixels"] * T_EXP
        poisson.append(s / np.sqrt(s + b + d + r["n_pixels"] * etc.detector.read_noise_e ** 2))

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.semilogy(mags, poisson, color=COLORS["red"], linestyle="--",
                label="photon + read + dark only")
    ax.semilogy(mags, full, color=COLORS["blue"],
                label="full budget (+ scintillation, ADC)")
    ax.set_xlabel(r"target magnitude (AB, $V$-like band)")
    ax.set_ylabel(f"S/N in {T_EXP:.0f} s")
    ax.set_xlim(2, 18)
    ax.legend(loc="lower left", fontsize=7)
    ax.annotate("scintillation\nlimit", xy=(4.0, 1.5e3), fontsize=7, color=COLORS["blue"])
    save(fig, "fig_06_snr_budget")


if __name__ == "__main__":
    main()
