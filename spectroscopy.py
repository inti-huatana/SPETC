"""Spectroscopic ETC, reporting counts and S/N per resolution element.

Slit mode applies the selected PSF model (Gaussian/Moffat) to slit and
extraction losses and, when the slit is not at the parallactic angle, the
per-wavelength Filippenko (1982) differential-refraction offset loss.
Slitless mode supports transmission-grating physics (Star Analyser 100/200):
the dispersion follows from the groove density and grating-to-sensor
distance, and the resolution element from seeing, intrinsic LSF and the
atmospheric-dispersion smear.  Noise includes photon, dark, read,
scintillation and ADC quantization terms.
"""

import numpy as np
import pandas as pd
import astropy.units as u
from scipy.special import erf

from etc_physics import (as_angstrom_curve, background_noise_factor, calibrated_template_magnitude,
                         magnitude_f_lambda, psf_slit_throughput, collecting_area,
                         atmospheric_transmission, instrument_transmission, transformed_template,
                         FWHM_TO_SIGMA)
from observing_conditions import (differential_refraction_arcsec, digitization_noise_e,
                                  effective_seeing_arcsec, scintillation_variance_e2)
from astropy.constants import h, c
from ephemeris import site_pressure_hpa, site_temperature_c
from spectral_utils import require_coverage, interpolate_checked, interpolate_zero_filled


def grating_dispersion_aa_per_pixel(lines_per_mm, grating_to_sensor_mm, pixel_size_um,
                                    diffraction_order=1):
    """First-order transmission-grating dispersion at the detector [A/pixel].

    For a converging-beam grating such as the Star Analyser, x = L m n lambda,
    so dlambda/dpixel = 1e4 p_um / (L_mm n_mm m).  A Star Analyser 100 at
    L = 42 mm on 4.63 um pixels gives ~11 A/pixel, matching its published
    calibration.
    """
    lines = float(lines_per_mm)
    distance = float(grating_to_sensor_mm)
    order = int(diffraction_order)
    if lines <= 0 or distance <= 0 or order < 1:
        raise ValueError("Grating groove density, distance and order must be positive.")
    return 1.0e4 * float(pixel_size_um) / (distance * lines * order)


def slit_spectrograph_resolving_power(wavelength_aa, grating_lines_mm, collimator_fl_mm,
                                      camera_fl_mm, telescope_fl_mm, slit_width_arcsec,
                                      seeing_arcsec, diffraction_order=1):
    """Resolving power of a classical slit spectrograph (Littrow approximation).

    The grating equation in Littrow configuration gives
    sin(beta) = m n lambda / 2 and a reciprocal linear dispersion
    dlambda/dx = cos(beta) 1e7 / (m n f_cam) [A/mm].  The resolution element
    on the detector is the narrower of the geometric slit image and the
    seeing image, both reimaged by f_cam/f_coll (anamorphism = 1 in
    Littrow).  Amateurs know these numbers from their hardware; the returned
    R feeds the ETC's resolving-power field, which can still be edited.
    """
    lines = float(grating_lines_mm)
    coll = float(collimator_fl_mm)
    cam = float(camera_fl_mm)
    tel = float(telescope_fl_mm)
    slit = float(slit_width_arcsec)
    seeing = float(seeing_arcsec)
    order = int(diffraction_order)
    lam = float(wavelength_aa)
    if min(lines, coll, cam, tel, slit, seeing, lam) <= 0 or order < 1:
        raise ValueError("All spectrograph geometry values must be positive.")
    sin_beta = order * lines * (lam * 1.0e-7) / 2.0
    if sin_beta >= 1.0:
        raise ValueError(
            f"{lam:.0f} Å is not diffracted by {lines:.0f} lines/mm in order {order} "
            "(grating equation exceeds 90 deg).")
    cos_beta = np.sqrt(1.0 - sin_beta**2)
    dispersion_aa_mm = cos_beta * 1.0e7 / (order * lines * cam)
    slit_focal_mm = slit / 206265.0 * tel
    seeing_focal_mm = seeing / 206265.0 * tel
    limited_by = "slit" if slit_focal_mm <= seeing_focal_mm else "seeing"
    width_detector_mm = min(slit_focal_mm, seeing_focal_mm) * cam / coll
    dlam = width_detector_mm * dispersion_aa_mm
    if dlam <= 0:
        raise ValueError("Computed resolution element is not positive.")
    return {"resolving_power": lam / dlam, "resolution_element_aa": dlam,
            "dispersion_aa_mm": dispersion_aa_mm, "limited_by": limited_by,
            "blaze_angle_deg": float(np.degrees(np.arcsin(sin_beta)))}


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
                             observing_filter=None,
                             slitless_grating_lines_mm=None,
                             slitless_grating_distance_mm=None,
                             grating_efficiency=1.0,
                             slit_at_parallactic=True,
                             include_atmospheric_dispersion=True,
                             radial_velocity_kms=0.0, ebv=0.0,
                             slit_geometry=None, grating_efficiency_curve=None):
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
        # Guiding/image-motion blur: an rms tracking error sigma_g adds a
        # Gaussian of FWHM 2.355 sigma_g in quadrature to the seeing FWHM.
        guiding_rms = float(self.atmosphere.get("guiding_rms_arcsec", 0.0))
        if guiding_rms > 0.0:
            seeing = float(np.hypot(seeing, FWHM_TO_SIGMA * guiding_rms))
        psf_model = str(self.atmosphere.get("psf_model", "gaussian"))
        moffat_beta = float(self.atmosphere.get("moffat_beta", 2.5))
        airmass = float(self.atmosphere.get("airmass", 1.0))
        # Optional Kolmogorov scaling of the seeing with wavelength and
        # airmass (FWHM ~ X^0.6 (lambda/5000)^-0.2, the ETC-42 convention);
        # the entered seeing is then the zenith V value.  When disabled the
        # seeing is flat, as before.
        seeing_scaling = bool(self.atmosphere.get("seeing_wavelength_scaling", False))
        elevation_m = float(self.atmosphere.get("elevation_m", 0.0))
        pressure_hpa = float(self.atmosphere.get("pressure_hpa", site_pressure_hpa(elevation_m)))
        temperature_c = float(self.atmosphere.get("temperature_c", site_temperature_c(elevation_m)))
        dispersion_reference_aa = 0.5 * (lo + hi)
        grating_efficiency = float(grating_efficiency)
        if not 0.0 < grating_efficiency <= 1.0:
            raise ValueError("Grating efficiency must be in (0, 1].")
        resolution_R_geometry = None
        if mode == "slitless":
            if slitless_grating_lines_mm and slitless_grating_distance_mm:
                dispersion = grating_dispersion_aa_per_pixel(
                    slitless_grating_lines_mm, slitless_grating_distance_mm,
                    self.detector.pixel_size_um)
            else:
                dispersion = float(slitless_dispersion_aa_pix)
            if dispersion <= 0:
                raise ValueError("Slitless dispersion (Angstrom/pixel) must be positive.")
            seeing_lsf = (float(effective_seeing_arcsec(seeing, dispersion_reference_aa, airmass))
                          if seeing_scaling else seeing)
            lsf_pixels = np.hypot(seeing_lsf / plate_scale, float(slitless_intrinsic_fwhm_pix))
            if lsf_pixels <= 0:
                raise ValueError("Slitless intrinsic FWHM must be non-negative.")
            if include_atmospheric_dispersion:
                # Atmospheric dispersion elongates the zero-deviation image
                # along the refraction direction; within one resolution
                # element it adds in quadrature to the LSF.
                resel_aa = dispersion * lsf_pixels
                smear_arcsec = abs(float(
                    differential_refraction_arcsec(
                        dispersion_reference_aa + 0.5 * resel_aa, dispersion_reference_aa - 0.5 * resel_aa,
                        airmass, pressure_hpa, temperature_c)))
                lsf_pixels = np.hypot(lsf_pixels, smear_arcsec / plate_scale)
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
            # Engine-side sanity clamp: when the spectrograph geometry is
            # supplied, the entered R cannot exceed what the slit/seeing and
            # the grating deliver at the band centre.
            if slit_geometry:
                geometry = slit_spectrograph_resolving_power(
                    0.5 * (lo + hi), float(slit_geometry["grating_lines_mm"]),
                    float(slit_geometry["collimator_fl_mm"]), float(slit_geometry["camera_fl_mm"]),
                    float(self.telescope["focal_length_mm"]), float(slit_width_arcsec), seeing,
                    int(slit_geometry.get("diffraction_order", 1)))
                resolution_R_geometry = float(geometry["resolving_power"])
                if resolution_R > resolution_R_geometry:
                    resolution_R = resolution_R_geometry
            # One sample is one slit-spectrograph resolution element, not one detector pixel.
            n = max(int(np.ceil(resolution_R * np.log(hi / lo))), 2)
            wave = lo * np.exp(np.arange(n) * np.log(hi / lo) / (n - 1)) * u.AA
            dlam = wave / resolution_R
            effective_resolution = np.full(n, float(resolution_R))
            dispersion_pixels_per_resel = float(pixels_per_resel)
        reference_filter = magnitude_band if reference_filter is None else reference_filter
        # Optional radial-velocity shift and CCM89 reddening are applied to
        # the template *before* calibration: the observed reference magnitude
        # of a reddened, moving star already contains both effects, so the
        # transformation changes the template's colours, not its level.
        star_spec = transformed_template(star_spec, radial_velocity_kms, ebv)
        spec_wave, spec_flam = calibrated_template_magnitude(
            star_spec, target_mag, reference_filter, target_zero_point_jy,
            template_mv0, visual_band, visual_zero_point_jy,
            reference_detector_type, visual_detector_type)
        source_curve = np.column_stack((spec_wave.to_value(u.AA), spec_flam.to_value(spec_flam.unit)))
        # Observing-side curves are zero-filled outside their tabulated
        # coverage (same policy as photometry): truncation only loses counts,
        # it never rescales the calibration.
        qe = interpolate_zero_filled(wave.to_value(u.AA), qe_curve, "QE curve", clip=(0.0, 1.0))
        if observing_filter is None:
            observing_transmission = np.ones(wave.size)
        else:
            observing_transmission = interpolate_zero_filled(
                wave.to_value(u.AA), observing_filter, "spectroscopic observing filter", clip=(0.0, 1.0))
        if extraction_height_arcsec is None:
            extraction_height_arcsec = seeing
        extraction_height_arcsec = float(extraction_height_arcsec)
        if extraction_height_arcsec <= 0:
            raise ValueError("Extraction height must be positive.")
        # Per-wavelength seeing for the light-loss integrals; the extraction
        # geometry (fixed number of pixels the observer sums over) stays on
        # the reference seeing.
        seeing_eff = (effective_seeing_arcsec(seeing, wave.to_value(u.AA), airmass)
                      if seeing_scaling else np.full(wave.size, seeing))
        sky_mag = float(self.sky_model.get("sky_mag", self.sky_model.get("sky_mag_ab_arcsec2")))
        sky_zero_point_jy = float(self.sky_model.get("sky_zero_point_jy", 3631.0))
        if mode == "slit":
            slit_width_arcsec = float(slit_width_arcsec)
            if slit_width_arcsec <= 0:
                raise ValueError("Slit width must be positive.")
            if include_atmospheric_dispersion and not slit_at_parallactic:
                # Worst-case geometry: the full differential refraction is
                # perpendicular to the slit and decentres the PSF per
                # wavelength (Filippenko 1982).  At the parallactic angle the
                # offset runs along the slit and is recovered by extraction.
                dispersion_offsets = differential_refraction_arcsec(
                    wave.to_value(u.AA), dispersion_reference_aa, airmass,
                    pressure_hpa, temperature_c)
            else:
                dispersion_offsets = np.zeros(wave.size)
            # Width is the slit loss; height is the finite cross-dispersion
            # extraction loss.  Both dimensions contribute sky area.
            source_fraction = (psf_slit_throughput(slit_width_arcsec, seeing_eff, psf_model,
                                                   moffat_beta, offset_arcsec=dispersion_offsets)
                               * psf_slit_throughput(extraction_height_arcsec, seeing_eff,
                                                     psf_model, moffat_beta))
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
            source_fraction = psf_slit_throughput(cross_dispersion_height, seeing_eff,
                                                  psf_model, moffat_beta)
            spatial_pixels = max(cross_dispersion_height / plate_scale, 1.0)
            sky_area = (dispersion_pixels_per_resel * plate_scale) * cross_dispersion_height
        # OSC single-channel extraction: rates and channel-pixel counts scale
        # by the Bayer fill fraction; the peak pixel (from the unextracted
        # rates further below) does not.
        fill = self.detector.channel_fill_fraction
        source_fraction = source_fraction * fill
        n_pixels = max(dispersion_pixels_per_resel * spatial_pixels * fill, 1.0)

        area = collecting_area(self.telescope)
        efficiency = float(self.telescope.get("efficiency", 1.0))
        if not 0.0 <= efficiency <= 1.0:
            raise ValueError("Telescope efficiency must be in [0, 1].")
        area_cm2 = area.to_value(u.cm**2)
        atmosphere_trans = atmospheric_transmission(wave, self.atmosphere)
        instrument_trans = instrument_transmission(wave, self.telescope)
        photon_energy_erg = (h * c / wave).to_value(u.erg)
        wave_aa = wave.to_value(u.AA)
        dlam_aa = dlam.to_value(u.AA)

        # --- Source: exact per-resel integral of the calibrated template ---
        # The template is *integrated* over each resolution element instead
        # of point-sampled at its centre, so total line counts are invariant
        # under the choice of R (a 2 A emission line yields the same
        # electrons at R = 300 and R = 20000).  A single cumulative trapezoid
        # over a fine union grid keeps this O(n) even at R = 100000.
        edges_lo = np.clip(wave_aa - 0.5 * dlam_aa, lo, hi)
        edges_hi = np.clip(wave_aa + 0.5 * dlam_aa, lo, hi)
        template_wave = source_curve[:, 0]
        inside = (template_wave >= lo) & (template_wave <= hi)
        fine = np.union1d(np.union1d(template_wave[inside],
                                     np.concatenate((edges_lo, edges_hi))),
                          np.linspace(lo, hi, 2048))
        fine_q = fine * u.AA
        fine_flam = np.interp(fine, template_wave, source_curve[:, 1], left=0.0, right=0.0)
        fine_filter = (np.ones(fine.size) if observing_filter is None else
                       interpolate_zero_filled(fine, observing_filter,
                                               "spectroscopic observing filter", clip=(0.0, 1.0)))
        fine_qe = interpolate_zero_filled(fine, qe_curve, "QE curve", clip=(0.0, 1.0))
        fine_instrument = instrument_transmission(fine_q, self.telescope)
        fine_atmosphere = atmospheric_transmission(fine_q, self.atmosphere)
        fine_energy_erg = (h * c / fine_q).to_value(u.erg)
        # Grating efficiency: a wavelength-dependent curve (if supplied) takes
        # precedence over the scalar and folds directly into the integrands.
        if grating_efficiency_curve is not None:
            geff_wave, geff_values = as_angstrom_curve(grating_efficiency_curve, "grating efficiency")
            geff_wave_aa = geff_wave.to_value(u.AA)
            fine_grating_eff = np.clip(np.interp(fine, geff_wave_aa, geff_values,
                                                 left=geff_values[0], right=geff_values[-1]), 0.0, 1.0)
        else:
            fine_grating_eff = np.full(fine.size, float(grating_efficiency))
        fine_integrand = (fine_flam * fine_filter * fine_qe * fine_instrument * fine_atmosphere
                          * fine_grating_eff * area_cm2 * efficiency / fine_energy_erg)
        cumulative = np.concatenate(([0.0], np.cumsum(
            0.5 * (fine_integrand[1:] + fine_integrand[:-1]) * np.diff(fine))))
        source_rates_unextracted = (np.interp(edges_hi, fine, cumulative)
                                    - np.interp(edges_lo, fine, cumulative))
        source_rates = source_rates_unextracted * source_fraction

        # --- Sky ---
        spectral_sky_flam = self.sky_model.get("spectral_sky_f_lambda")
        spectral_sky = self.sky_model.get("spectral_sky_mag_offsets")

        def sky_flam_arcsec2(wavelengths_aa):
            """Sky surface brightness F_lambda per arcsec^2 [erg/s/cm^2/A/arcsec^2]."""
            grid = np.asarray(wavelengths_aa, dtype=float)
            if spectral_sky_flam is not None:
                sky_wave, sky_values = as_angstrom_curve(spectral_sky_flam, "spectral sky F_lambda")
                return np.interp(grid, sky_wave.to_value(u.AA), sky_values,
                                 left=sky_values[0], right=sky_values[-1])
            if spectral_sky is None:
                mags = np.full(grid.size, sky_mag)
            else:
                colour_wave, colour_mag = as_angstrom_curve(spectral_sky, "spectral sky colour")
                mags = sky_mag + np.interp(grid, colour_wave.to_value(u.AA), colour_mag,
                                           left=colour_mag[0], right=colour_mag[-1])
            zero = magnitude_f_lambda(grid * u.AA, sky_zero_point_jy).to_value(
                u.erg / (u.s * u.cm**2 * u.AA))
            return zero * 10.0 ** (-0.4 * mags)

        sky_observed_at_telescope = bool(self.sky_model.get("sky_at_telescope", False))
        if mode == "slit":
            # The slit restricts the sky to slit width x extraction height,
            # and each resel sees only its own dlam of sky.
            sky_transmission = np.ones(n) if sky_observed_at_telescope else atmosphere_trans
            grating_eff_resel = np.interp(wave_aa, fine, fine_grating_eff)
            sky_rates = (sky_flam_arcsec2(wave_aa) * sky_area * observing_transmission * qe
                         * instrument_trans * sky_transmission * area_cm2 * efficiency
                         / photon_energy_erg * dlam_aa) * grating_eff_resel * fill
        else:
            # Slitless: dispersing a spatially uniform source leaves the
            # detector uniformly illuminated, so *every pixel* receives sky
            # light integrated over the entire grating bandpass — the
            # classical reason objective-grating spectra are sky-limited.
            # Booking only dlam of sky per resel (as a slit would allow)
            # underestimates the background by ~(band width / resel width),
            # two orders of magnitude in a typical Star Analyser setup.
            fine_sky_trans = np.ones(fine.size) if sky_observed_at_telescope else fine_atmosphere
            sky_integrand = (sky_flam_arcsec2(fine) * plate_scale**2 * fine_filter * fine_qe
                             * fine_instrument * fine_sky_trans * fine_grating_eff * area_cm2 * efficiency
                             / fine_energy_erg)
            trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            sky_rate_per_pixel = float(trapezoid(sky_integrand, fine))
            sky_rates = np.full(n, sky_rate_per_pixel * dispersion_pixels_per_resel
                                * spatial_pixels * fill)
        source_e_unextracted = source_rates_unextracted * t_exp_s
        source_e = source_rates * t_exp_s
        sky_e = sky_rates * t_exp_s
        dark_e = self.detector.dark_current_e_s_pix * n_pixels * t_exp_s
        # Generic extra background per pixel (detector glow, stray light,
        # ghosting): a catch-all Poisson term (cf. ETC-42 ExtraBackgroundNoise).
        extra_bg_rate = float(self.sky_model.get("extra_background_e_s_pixel", 0.0))
        extra_bg_e = extra_bg_rate * n_pixels * t_exp_s
        scintillation_var = scintillation_variance_e2(
            source_e, float(self.telescope["diameter_mm"]), airmass, elevation_m, t_exp_s)
        digitization_var = n_pixels * digitization_noise_e(self.detector.gain_e_adu)**2
        # Sky subtraction from finite sky windows along the slit (or beside
        # the slitless trace) multiplies every per-pixel background variance
        # by (1 + n_pix/n_sky) (Merline & Howell 1995); scintillation is a
        # source-borne term and stays outside the factor.
        subtraction_factor = background_noise_factor(
            n_pixels, self.sky_model.get("sky_annulus_pixels", 0.0))
        snrs = source_e / np.sqrt(np.maximum(
            source_e + subtraction_factor * (sky_e + dark_e
                                             + n_pixels * self.detector.read_noise_e**2
                                             + digitization_var + extra_bg_e)
            + scintillation_var, 1e-300))
        # Cayrel (1988) equivalent-width uncertainty: sigma(EW) =
        # 1.5 sqrt(FWHM dx) / (S/N per pixel), with the resolution element as
        # the line FWHM and dx the dispersion-pixel width.  Reported in mA.
        snr_per_pixel = snrs / np.sqrt(max(dispersion_pixels_per_resel, 1.0))
        pixel_aa = dlam.to_value(u.AA) / max(dispersion_pixels_per_resel, 1e-12)
        sigma_ew_mangstrom = 1000.0 * 1.5 * np.sqrt(dlam.to_value(u.AA) * pixel_aa) / np.maximum(snr_per_pixel, 1e-12)
        # Brightest pixel: separable product of the (Gaussian) instrumental
        # LSF along dispersion and the selected PSF model across it.
        sigma_disp_pix = dispersion_pixels_per_resel / 2.354820045
        peak_dispersion_fraction = erf(0.5 / (np.sqrt(2.0) * sigma_disp_pix))
        peak_spatial_fraction = psf_slit_throughput(plate_scale, seeing_eff, psf_model, moffat_beta)
        peak_e = (source_e_unextracted * peak_dispersion_fraction * peak_spatial_fraction + sky_e / n_pixels
                  + self.detector.dark_current_e_s_pix * t_exp_s + extra_bg_rate * t_exp_s)
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
                             "snr": snrs, "sigma_ew_mangstrom": sigma_ew_mangstrom,
                             "adu": adu, "saturated": saturated.astype(int),
                             "peak_e_unclipped": peak_e, "peak_adu_unclipped": peak_adu_unclipped,
                             "full_well_fraction": peak_e / self.detector.full_well_e,
                             "saturation_flag": self.detector.saturation_flag(peak_e),
                             "peak_rate_e_s": peak_rate_e_s,
                             "max_unsaturated_exptime_s": max_unsaturated_exptime_s})
        result.attrs["n_pixels_per_resel"] = n_pixels
        result.attrs["spectroscopy_mode"] = mode
        result.attrs["experimental"] = mode == "slitless"
        result.attrs["effective_resolution_R"] = float(np.median(effective_resolution))
        result.attrs["psf_model"] = psf_model
        result.attrs["sky_mag_arcsec2"] = float(sky_mag)
        result.attrs["scintillation_noise_e_median"] = float(np.median(np.sqrt(scintillation_var)))
        result.attrs["digitization_noise_e"] = float(np.sqrt(digitization_var))
        result.attrs["sky_subtraction_factor"] = float(subtraction_factor)
        result.attrs["radial_velocity_kms"] = float(radial_velocity_kms)
        result.attrs["ebv"] = float(ebv)
        result.attrs["telluric_bands_included"] = bool(
            self.atmosphere.get("include_telluric_bands", False))
        if resolution_R_geometry is not None:
            result.attrs["resolution_R_geometry"] = resolution_R_geometry
            result.attrs["resolution_clamped"] = bool(
                np.isclose(float(resolution_R), resolution_R_geometry))
        if mode == "slit" and seeing < 0.7 * float(slit_width_arcsec):
            # The star underfills the slit: the delivered resolution is set
            # by the seeing image, i.e. finer than the slit-limited R — the
            # reported R is then conservative.
            result.attrs["resolution_note"] = (
                "seeing underfills the slit; delivered resolution is "
                "seeing-limited (finer than the slit-limited R used here)")
        if mode == "slitless":
            result.attrs["dispersion_aa_pix"] = float(dispersion)
            result.attrs["grating_efficiency"] = grating_efficiency
            result.attrs["sky_rate_per_pixel_e_s"] = float(sky_rate_per_pixel)
        return result
