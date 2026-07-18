"""Broad-band ETC with separate reference and observing passbands.

Point sources use the selected PSF model (Gaussian or Moffat) for aperture
losses and peak-pixel prediction.  Extended sources (galaxies, nebulae,
planets) are treated as uniform surface brightness over a stated angular
area: the entered magnitude remains the integrated magnitude, the aperture
captures an area fraction, and the peak pixel holds the mean surface rate.
Noise includes photon, dark, read, scintillation and ADC quantization terms.
"""

import numpy as np
import astropy.units as u

from etc_physics import (as_angstrom_curve, calibrated_template_magnitude, electron_rate,
                         magnitude_f_lambda, psf_encircled_energy, psf_slit_throughput,
                         snr, synthetic_magnitude)
from observing_conditions import digitization_noise_e, scintillation_variance_e2
from spectral_utils import require_coverage, interpolate_checked, interpolate_zero_filled


class PhotometryETC:
    def __init__(self, telescope, detector, atmosphere, sky_model):
        self.telescope = telescope
        self.detector = detector
        self.atmosphere = atmosphere
        self.sky_model = sky_model

    def compute_photometry_single(self, star_spec, observing_filter, qe_curve, target_mag, t_exp_s,
                                  target_zero_point_jy=3631.0, reference_filter=None,
                                  template_mv0=0.0, visual_band=None,
                                  visual_zero_point_jy=3631.0,
                                  observing_zero_point_jy=3631.0,
                                  reference_detector_type=1, visual_detector_type=1,
                                  observing_detector_type=1,
                                  source_geometry="point", source_area_arcsec2=None):
        if t_exp_s <= 0:
            raise ValueError("Exposure time must be positive.")
        geometry = str(source_geometry).strip().lower()
        if geometry not in {"point", "extended"}:
            raise ValueError("Source geometry must be 'point' or 'extended'.")
        wave, transmission = as_angstrom_curve(observing_filter, "photometric observing filter")
        transmission = np.clip(transmission, 0.0, 1.0)
        active = transmission > 0.0
        if active.sum() < 2:
            raise ValueError("Photometric observing filter has no positive transmission samples.")
        wave, transmission = wave[active], transmission[active]
        qe = interpolate_zero_filled(wave.to_value(u.AA), qe_curve, "QE curve", clip=(0.0, 1.0))

        reference_filter = observing_filter if reference_filter is None else reference_filter
        spec_wave, spec_flam = calibrated_template_magnitude(
            star_spec, target_mag, reference_filter, target_zero_point_jy,
            template_mv0, visual_band, visual_zero_point_jy,
            reference_detector_type, visual_detector_type)
        #require_coverage(wave.to_value(u.AA),
        #                 np.column_stack((spec_wave.to_value(u.AA), spec_flam.to_value(spec_flam.unit))),
        #                 "template spectrum")
        target_flam = interpolate_zero_filled(wave.to_value(u.AA),
                                          np.column_stack((spec_wave.to_value(u.AA), spec_flam.to_value(spec_flam.unit))),
                                          "template spectrum") * spec_flam.unit
        total_source_rate = electron_rate(wave, target_flam, transmission, qe, self.telescope, self.atmosphere)
        standard_observing_mag = synthetic_magnitude(
            np.column_stack((spec_wave.value, spec_flam.to_value(spec_flam.unit))), observing_filter,
            observing_zero_point_jy, observing_detector_type)
        zero_rate = electron_rate(wave, magnitude_f_lambda(wave, observing_zero_point_jy), transmission, qe,
                                  self.telescope, self.atmosphere)
        #instrumental_response_mag = float(-2.5 * np.log10(
        #    total_source_rate.to_value(1 / u.s) / zero_rate.to_value(1 / u.s)))
        source_rate_value = total_source_rate.to_value(1 / u.s)
        zero_rate_value = zero_rate.to_value(1 / u.s)
        
        instrumental_response_mag = (
            float(-2.5 * np.log10(source_rate_value / zero_rate_value))
            if source_rate_value > 0 and zero_rate_value > 0
            else np.nan
        )
    
        seeing = float(self.atmosphere["seeing_arcsec"])
        psf_model = str(self.atmosphere.get("psf_model", "gaussian"))
        moffat_beta = float(self.atmosphere.get("moffat_beta", 2.5))
        aperture_radius = float(self.sky_model.get("aperture_radius_arcsec", 1.0))
        aperture_area = np.pi * aperture_radius**2
        plate_scale = 206265.0 * (self.detector.pixel_size_um * 1e-3) / float(self.telescope["focal_length_mm"])
        total_rate_es = total_source_rate.to_value(1 / u.s)
        if geometry == "extended":
            # Uniform surface-brightness approximation: valid when the source
            # is much larger than the seeing disc, so PSF blur only moves
            # light across the source edge.
            source_area = float(source_area_arcsec2 or 0.0)
            if source_area <= 0:
                raise ValueError("Extended sources require a positive angular area in arcsec^2.")
            captured_fraction = min(aperture_area / source_area, 1.0)
            source_rate = total_source_rate * captured_fraction
            surface_rate_es = total_rate_es / source_area  # e-/s/arcsec^2
            peak_source_rate_es = surface_rate_es * plate_scale**2
        else:
            source_rate = total_source_rate * psf_encircled_energy(
                aperture_radius, seeing, psf_model, moffat_beta)
            # Pixel saturation is set by the untruncated PSF, not by the
            # aperture-extracted source counts used for S/N.  The central-pixel
            # fraction is the separable product of two 1-pixel-wide couplings.
            central_pixel_fraction = psf_slit_throughput(plate_scale, seeing, psf_model, moffat_beta)**2
            peak_source_rate_es = total_rate_es * central_pixel_fraction

        sky_mag = float(self.sky_model.get("sky_mag", self.sky_model.get("sky_mag_ab_arcsec2")))
        sky_zero_point_jy = float(self.sky_model.get("sky_zero_point_jy", 3631.0))
        sky_flam = magnitude_f_lambda(wave, sky_zero_point_jy) * 10.0**(-0.4 * sky_mag) * aperture_area
        # Sky models and manually entered sky brightness are already observed
        # at the telescope.  Applying source extinction to them again would
        # spuriously darken the background.
        sky_atmosphere = ({"airmass": 1.0, "seeing_arcsec": seeing, "transmission_curve": None}
                          if self.sky_model.get("sky_at_telescope", False) else self.atmosphere)
        sky_rate = electron_rate(wave, sky_flam, transmission, qe, self.telescope, sky_atmosphere)

        n_pixels = max(aperture_area / plate_scale**2, 1.0)
        source_e = source_rate.to_value(1 / u.s) * t_exp_s
        sky_e = sky_rate.to_value(1 / u.s) * t_exp_s
        dark_e = self.detector.dark_current_e_s_pix * n_pixels * t_exp_s
        scintillation_var = scintillation_variance_e2(
            source_e, float(self.telescope["diameter_mm"]),
            float(self.atmosphere.get("airmass", 1.0)),
            float(self.atmosphere.get("elevation_m", 0.0)), t_exp_s)
        digitization_var = n_pixels * digitization_noise_e(self.detector.gain_e_adu)**2
        result_snr = snr(source_e, sky_e, dark_e, self.detector.read_noise_e, n_pixels,
                         extra_variance_e2=scintillation_var + digitization_var)
        peak_total = (peak_source_rate_es * t_exp_s + sky_e / n_pixels
                      + self.detector.dark_current_e_s_pix * t_exp_s)
        peak_rate_e_s = peak_total / t_exp_s
        saturation_limit_e = min(self.detector.full_well_e, self.detector.max_electrons)
        max_unsaturated_exptime_s = saturation_limit_e / peak_rate_e_s if peak_rate_e_s > 0 else np.inf
        adu, is_sat = self.detector.counts_to_adu(peak_total)
        peak_adu_unclipped = peak_total / self.detector.gain_e_adu
        return {
            "mag": float(target_mag), "photons_source_es": source_rate.to_value(1 / u.s),
            "photons_sky_es": sky_rate.to_value(1 / u.s), "snr": result_snr,
            "adu": adu, "saturated": int(is_sat), "source_rate_per_s": source_rate.to_value(1 / u.s),
            "sky_rate_per_s": sky_rate.to_value(1 / u.s), "n_pixels": n_pixels,
            "peak_e_unclipped": float(peak_total), "peak_adu_unclipped": float(peak_adu_unclipped),
            "full_well_fraction": float(peak_total / self.detector.full_well_e),
            "saturation_flag": self.detector.saturation_flag(peak_total),
            "peak_rate_e_s": float(peak_rate_e_s),
            "max_unsaturated_exptime_s": float(max_unsaturated_exptime_s),
            "estimated_observing_magnitude": standard_observing_mag,
            "instrumental_response_magnitude": instrumental_response_mag,
            "scintillation_noise_e": float(np.sqrt(scintillation_var)),
            "digitization_noise_e": float(np.sqrt(digitization_var)),
            "source_geometry": geometry,
            "sky_mag_arcsec2": sky_mag,
        }
