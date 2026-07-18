"""Numerical smoke test independent of the project data files."""

import numpy as np

from detector import Detector
from photometry import PhotometryETC
from spectroscopy import SpectroscopyETC


def main():
    wave = np.linspace(3500.0, 10500.0, 1000)
    # Flux-calibrated visual-zero template, matching the BPGS convention.
    template = np.column_stack((wave, np.full_like(wave, 3.60e-9)))
    v_band = np.column_stack((wave, ((wave >= 5000.0) & (wave <= 6000.0)).astype(float)))
    qe = np.column_stack((wave, np.full_like(wave, 0.90)))
    atmo = np.column_stack((wave, np.full_like(wave, 0.85)))
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.70,
                 "focal_length_mm": 2000.0}
    detector = Detector(13.5, 2.5, 80000.0, 16, read_noise_e=5.0)
    atmosphere = {"airmass": 1.0, "seeing_arcsec": 0.8, "transmission_curve": atmo}
    sky = {"sky_mag": 21.7, "sky_zero_point_jy": 3631.0,
           "sky_at_telescope": True, "aperture_radius_arcsec": 1.0}
    phot = PhotometryETC(telescope, detector, atmosphere, sky).compute_photometry_single(
        template, v_band, qe, target_mag=5.0, t_exp_s=60.0)
    spec = SpectroscopyETC(telescope, detector, atmosphere, sky).compute_spectroscopy(
        template, 10000.0, 1.0, 60.0, (5000.0, 6000.0), 5.0, qe, v_band,
        pixels_per_resel=2.0, extraction_height_arcsec=1.0)
    max_spec_snr = float(spec["snr"].max())
    assert 150.0 < max_spec_snr < 800.0, max_spec_snr
    # A V=5 target in 60 s is scintillation-limited, not photon-limited: the
    # Young-law fractional rms (~7.6e-4 for 358 mm, X=1, sea level) caps the
    # broad-band S/N near 1/sigma_scint ~ 1.3e3.
    assert 8.0e2 < phot["snr"] < 3.0e3, phot["snr"]
    print(f"PASS: photometric S/N={phot['snr']:.1f} (scintillation-limited); "
          f"spectral max S/N/resel={max_spec_snr:.1f}")


if __name__ == "__main__":
    main()
