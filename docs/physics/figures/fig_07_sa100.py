#!/usr/bin/env python3
"""Star Analyser 100 slitless prediction from the released SpectroscopyETC."""

import numpy as np
import matplotlib.pyplot as plt

from figstyle import COLORS, DOUBLE, save
from detector import Detector
from spectroscopy import SpectroscopyETC


def main():
    wave = np.linspace(3500.0, 10500.0, 1000)
    template = np.column_stack((wave, np.full_like(wave, 3.60e-9)))
    band = np.column_stack((wave, np.ones_like(wave)))
    qe = np.column_stack((wave, 0.9 * np.exp(-0.5 * ((wave - 5500.0) / 2500.0) ** 2)))
    telescope = {"diameter_mm": 200.0, "obstruction_mm": 70.0, "efficiency": 0.85,
                 "focal_length_mm": 1000.0}
    detector = Detector(4.63, 0.8, 25000.0, 14, read_noise_e=6.0, dark_current_e_s_pix=0.05)
    atmosphere = {"airmass": 1.3, "seeing_arcsec": 2.5, "transmission_curve": None,
                  "psf_model": "moffat", "moffat_beta": 2.5, "elevation_m": 300.0}
    sky = {"sky_mag": 19.5, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True}
    etc = SpectroscopyETC(telescope, detector, atmosphere, sky)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
    for mag, color in ((6.0, COLORS["blue"]), (8.0, COLORS["green"]), (10.0, COLORS["red"])):
        result = etc.compute_spectroscopy(
            template, 100.0, 1.0, 120.0, (4000.0, 8000.0), mag, qe, band,
            spectroscopy_mode="slitless", slitless_extraction_width_arcsec=5.0,
            slitless_dispersion_aa_pix=10.0, slitless_intrinsic_fwhm_pix=1.0,
            visual_band=band, slitless_grating_lines_mm=100.0,
            slitless_grating_distance_mm=42.0, grating_efficiency=0.5)
        ax1.plot(result["wavelength_aa"] / 1e4, result["photons_source_es"],
                 color=color, label=f"$V={mag:.0f}$")
        ax2.plot(result["wavelength_aa"] / 1e4, result["snr"], color=color,
                 label=f"$V={mag:.0f}$")
    resolution = result.attrs["effective_resolution_R"]
    dispersion = result.attrs["dispersion_aa_pix"]
    ax1.set_xlabel(r"wavelength [$\mu$m]")
    ax1.set_ylabel(r"source rate [e$^-$ s$^{-1}$ resel$^{-1}$]")
    ax1.set_yscale("log")
    ax1.legend()
    ax2.set_xlabel(r"wavelength [$\mu$m]")
    ax2.set_ylabel("S/N per resolution element (120 s)")
    ax2.legend()
    ax2.set_title(fr"SA100: {dispersion:.1f} A pix$^{{-1}}$, median $R \simeq {resolution:.0f}$",
                  fontsize=8)
    save(fig, "fig_07_sa100")


if __name__ == "__main__":
    main()
