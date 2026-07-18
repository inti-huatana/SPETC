"""Spectroscopic ETC, reporting counts and S/N per resolution element."""

import numpy as np
import pandas as pd
import astropy.units as u
from math import erf

from etc_physics import (as_angstrom_curve, calibrated_template_magnitude, magnitude_f_lambda,
                         slit_throughput, snr, collecting_area, atmospheric_transmission, instrument_transmission)
from astropy.constants import h, c
from spectral_utils import require_coverage, interpolate_checked


class SpectroscopyETC:
    def __init__(self, telescope, detector, atmosphere, sky_model):
        self.telescope = telescope
        self.detector = detector
        self.atmosphere = atmosphere
        self.sky_model = sky_model

    def compute_spectroscopy(self, star_spec, resolution_R, slit_width_arcsec, t_exp_s,
                             wavelength_range, target_mag, qe_curve, magnitude_band,
                             pixels_per_resel=2.0, extraction_height_arcsec=None,
                             target_zero_point_jy=3631.0, reference_filter=None,
                             template_mv0=0.0, visual_band=None,
                             visual_zero_point_jy=3631.0, spectroscopy_mode="slit",
                             slitless_extraction_width_arcsec=None,
                             slitless_dispersion_aa_pix=None,
                             slitless_intrinsic_fwhm_pix=1.0,
                             reference_detector_type=1, visual_detector_type=1,
                             observing_filter=None):
        if resolution_R <= 0 or t_exp_s <= 0 or pixels_per_resel <= 0:
            raise ValueError("Resolution, exposure time, and sampling must be positive.")
        lo, hi = map(float, wavelength_range)
        if lo <= 0 or hi <= lo:
            raise ValueError("Invalid wavelength range.")
        mode = str(spectroscopy_mode).strip().lower()
        if mode not in {"slit", "slitless"}:
            raise ValueError("Spectroscopy mode must be 'slit' or 'slitless'.")
        plate_scale = 206265.0 * (self.detector.pixel_size_um * 1e-3) / float(self.telescope["focal_length_mm"])
        seeing = float(self.atmosphere["seeing_arcsec"])
        if mode == "slitless":
            dispersion = float(slitless_dispersion_aa_pix)
            if dispersion <= 0:
                raise ValueError("Slitless dispersion (Angstrom/pixel) must be positive.")
            lsf_pixels = np.hypot(seeing / plate_scale, float(slitless_intrinsic_fwhm_pix))
            if lsf_pixels <= 0:
                raise ValueError("Slitless intrinsic FWHM must be non-negative.")
            dlam_value = dispersion * lsf_pixels
            n = max(int(np.ceil((hi - lo) / dlam_value)) + 1, 2)
            wave = np.linspace(lo, hi, n) * u.AA
            dlam = np.full(n, dlam_value) * u.AA
            effective_resolution = wave.value / dlam_value
            dispersion_pixels_per_resel = lsf_pixels
        else:
            resolution_curve = self.telescope.get("slit_resolution_curve")
            if resolution_curve is not None:
                curve = np.asarray(resolution_curve, dtype=float)
                if curve.ndim != 2 or curve.shape[1] < 2:
                    raise ValueError("Slit resolution curve must be width_arcsec, resolving_power.")
                curve = curve[np.argsort(curve[:, 0])]
                slit_width = float(slit_width_arcsec)
                if slit_width < curve[0, 0] or slit_width > curve[-1, 0]:
                    raise ValueError("Slit width lies outside the calibrated resolution-curve range.")
                resolution_R = float(np.interp(slit_width, curve[:, 0], curve[:, 1]))
                if resolution_R <= 0:
                    raise ValueError("Calibrated slit resolving power must be positive.")
            # One sample is one slit-spectrograph resolution element, not one detector pixel.
            n = max(int(np.ceil(resolution_R * np.log(hi / lo))), 2)
            wave = lo * np.exp(np.arange(n) * np.log(hi / lo) / (n - 1)) * u.AA
            dlam = wave / resolution_R
            effective_resolution = np.full(n, float(resolution_R))
            dispersion_pixels_per_resel = float(pixels_per_resel)
        reference_filter = magnitude_band if reference_filter is None else reference_filter
        spec_wave, spec_flam = calibrated_template_magnitude(
            star_spec, target_mag, reference_filter, target_zero_point_jy,
            template_mv0, visual_band, visual_zero_point_jy,
            reference_detector_type, visual_detector_type)
        source_curve = np.column_stack((spec_wave.to_value(u.AA), spec_flam.to_value(spec_flam.unit)))
        require_coverage(wave.to_value(u.AA), source_curve, "template spectrum")
        source_flam = interpolate_checked(wave.to_value(u.AA), source_curve, "template spectrum") * spec_flam.unit
        qe = interpolate_checked(wave.to_value(u.AA), qe_curve, "QE curve", clip=(0.0, 1.0))
        if observing_filter is None:
            observing_transmission = np.ones(wave.size)
        else:
            observing_transmission = interpolate_checked(
                wave.to_value(u.AA), observing_filter, "spectroscopic observing filter", clip=(0.0, 1.0))
        if extraction_height_arcsec is None:
            extraction_height_arcsec = seeing
        extraction_height_arcsec = float(extraction_height_arcsec)
        if extraction_height_arcsec <= 0:
            raise ValueError("Extraction height must be positive.")
        sky_mag = float(self.sky_model.get("sky_mag", self.sky_model.get("sky_mag_ab_arcsec2")))
        sky_zero_point_jy = float(self.sky_model.get("sky_zero_point_jy", 3631.0))
        if mode == "slit":
            slit_width_arcsec = float(slit_width_arcsec)
            if slit_width_arcsec <= 0:
                raise ValueError("Slit width must be positive.")
            # Width is the slit loss; height is the finite cross-dispersion
            # extraction loss.  Both dimensions contribute sky area.
            source_fraction = (slit_throughput(slit_width_arcsec, seeing)
                               * slit_throughput(extraction_height_arcsec, seeing))
            sky_area = slit_width_arcsec * extraction_height_arcsec
            spatial_pixels = max(extraction_height_arcsec / plate_scale, 1.0)
        else:
            # In slitless data the dispersion-direction width is already the
            # LSF/resolution element.  A second angular "width" must not be
            # used as an aperture loss or sky width.  The sole free aperture
            # is the cross-dispersion extraction height.
            cross_dispersion_height = float(seeing if slitless_extraction_width_arcsec is None
                                            else slitless_extraction_width_arcsec)
            if cross_dispersion_height <= 0:
                raise ValueError("Slitless cross-dispersion extraction must be positive.")
            source_fraction = slit_throughput(cross_dispersion_height, seeing)
            spatial_pixels = max(cross_dispersion_height / plate_scale, 1.0)
            sky_area = (dispersion_pixels_per_resel * plate_scale) * cross_dispersion_height
        n_pixels = max(dispersion_pixels_per_resel * spatial_pixels, 1.0)

        # Midpoint integration is accurate for each narrow resolution element
        # and, unlike an Astropy loop for every bin, remains responsive at
        # R=100000 across a broad spectral range.
        area = collecting_area(self.telescope)
        efficiency = float(self.telescope.get("efficiency", 1.0))
        if not 0.0 <= efficiency <= 1.0:
            raise ValueError("Telescope efficiency must be in [0, 1].")
        atmosphere_trans = atmospheric_transmission(wave, self.atmosphere)
        instrument_trans = instrument_transmission(wave, self.telescope)
        photon_energy = (h * c / wave).to(u.erg)
        source_rates_unextracted = (source_flam * observing_transmission * qe * instrument_trans * atmosphere_trans * area * efficiency
                                    / photon_energy * dlam).to_value(1 / u.s)
        source_rates = source_rates_unextracted * source_fraction
        spectral_sky_flam = self.sky_model.get("spectral_sky_f_lambda")
        spectral_sky = self.sky_model.get("spectral_sky_mag_offsets")
        if spectral_sky_flam is not None:
            sky_wave, sky_flam_values = as_angstrom_curve(spectral_sky_flam, "spectral sky F_lambda")
            sky_flam = (np.interp(wave.to_value(u.AA), sky_wave.to_value(u.AA), sky_flam_values,
                                  left=sky_flam_values[0], right=sky_flam_values[-1])
                        * (u.erg / (u.s * u.cm**2 * u.AA)) * sky_area)
        elif spectral_sky is None:
            # Compatibility fallback for a manually specified broad-band sky.
            sky_mag_at_wave = np.full(wave.size, sky_mag)
            sky_flam = magnitude_f_lambda(wave, sky_zero_point_jy) * 10.0**(-0.4 * sky_mag_at_wave) * sky_area
        else:
            colour_wave, colour_mag = as_angstrom_curve(spectral_sky, "spectral sky colour")
            sky_mag_at_wave = sky_mag + np.interp(wave.to_value(u.AA), colour_wave.to_value(u.AA), colour_mag,
                                                   left=colour_mag[0], right=colour_mag[-1])
            sky_flam = magnitude_f_lambda(wave, sky_zero_point_jy) * 10.0**(-0.4 * sky_mag_at_wave) * sky_area
        sky_transmission = np.ones_like(atmosphere_trans) if self.sky_model.get("sky_at_telescope", False) else atmosphere_trans
        sky_rates = (sky_flam * observing_transmission * qe * instrument_trans * sky_transmission * area * efficiency / photon_energy
                     * dlam).to_value(1 / u.s)
        source_e_unextracted = source_rates_unextracted * t_exp_s
        source_e = source_rates * t_exp_s
        sky_e = sky_rates * t_exp_s
        dark_e = self.detector.dark_current_e_s_pix * n_pixels * t_exp_s
        snrs = source_e / np.sqrt(np.maximum(source_e + sky_e + dark_e + n_pixels * self.detector.read_noise_e**2, 1e-300))
        # Brightest pixel of separable Gaussian spectral and spatial profiles.
        sigma_disp_pix = dispersion_pixels_per_resel / 2.354820045
        sigma_spatial_pix = seeing / plate_scale / 2.354820045
        peak_dispersion_fraction = erf(0.5 / (np.sqrt(2.0) * sigma_disp_pix))
        peak_spatial_fraction = erf(0.5 / (np.sqrt(2.0) * sigma_spatial_pix))
        peak_e = (source_e_unextracted * peak_dispersion_fraction * peak_spatial_fraction + sky_e / n_pixels
                  + self.detector.dark_current_e_s_pix * t_exp_s)
        peak_rate_e_s = peak_e / t_exp_s
        saturation_limit_e = min(self.detector.full_well_e, self.detector.max_electrons)
        max_unsaturated_exptime_s = np.divide(saturation_limit_e, peak_rate_e_s,
                                              out=np.full_like(peak_rate_e_s, np.inf, dtype=float),
                                              where=peak_rate_e_s > 0)
        adu, saturated = self.detector.counts_to_adu(peak_e)
        peak_adu_unclipped = peak_e / self.detector.gain_e_adu
        result = pd.DataFrame({"wavelength_aa": wave.value, "resolution_element_aa": dlam.value,
                             "effective_resolution_R": effective_resolution,
                             "photons_source_es": source_rates, "photons_sky_es": sky_rates,
                             "snr": snrs, "adu": adu, "saturated": saturated.astype(int),
                             "peak_e_unclipped": peak_e, "peak_adu_unclipped": peak_adu_unclipped,
                             "full_well_fraction": peak_e / self.detector.full_well_e,
                             "saturation_flag": self.detector.saturation_flag(peak_e),
                             "peak_rate_e_s": peak_rate_e_s,
                             "max_unsaturated_exptime_s": max_unsaturated_exptime_s})
        result.attrs["n_pixels_per_resel"] = n_pixels
        result.attrs["spectroscopy_mode"] = mode
        result.attrs["experimental"] = mode == "slitless"
        result.attrs["effective_resolution_R"] = float(np.median(effective_resolution))
        return result
