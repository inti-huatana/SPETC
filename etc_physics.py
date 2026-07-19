"""Unit-safe radiometry shared by the photometric and spectroscopic ETCs.

Template files are calibrated spectral distributions.  ``template_mv0`` is
the visual magnitude represented by the stored distribution (zero for the
BPGS spectra, but not necessarily for other catalogue entries).  It is used
before the requested target magnitude is applied, as in ``interpola_spad``.
"""

import math
import numpy as np
import astropy.units as u
from astropy.constants import c, h
from spectral_utils import (as_curve, require_coverage, interpolate_checked, interpolate_zero_filled,
                            interpolate_edge_extended)


FLAM_UNIT = u.erg / (u.s * u.cm**2 * u.AA)
AB_FNU = 3631.0 * u.Jy


def trapz_quantity(y, x):
    """Integrate a Quantity using the unit of the supplied abscissa."""
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return integrate(y.value, x.to_value(x.unit)) * y.unit * x.unit


def as_angstrom_curve(curve, name, wavelength_unit="Angstrom"):
    """Return a validated numerical curve in Å; input units are explicit."""
    item = as_curve(curve, name=name, wavelength_unit_name=wavelength_unit)
    return item.wavelength_aa * u.AA, item.values


def magnitude_f_lambda(wavelength, zero_point_jy=3631.0):
    """Flux density of magnitude zero for a supplied f_nu zero point in Jy."""
    zero_point = float(zero_point_jy) * u.Jy
    if not np.isfinite(zero_point.value) or zero_point <= 0 * u.Jy:
        raise ValueError("Magnitude zero point must be finite and positive.")
    return (zero_point * c / wavelength**2).to(FLAM_UNIT)


def ab_f_lambda(wavelength):
    """AB=0 spectral flux density evaluated at ``wavelength``."""
    return magnitude_f_lambda(wavelength, 3631.0)


def interpolate_quantity(wavelength, source_wavelength, source_values, unit=FLAM_UNIT):
    values = np.interp(wavelength.to_value(u.AA), source_wavelength.to_value(u.AA),
                       source_values, left=0.0, right=0.0)
    return values * unit


def _band_integral(wavelength, f_lambda, band_wave, band, detector_type=1):
    """Return the SVO-defined in-band integral for an ``F_lambda`` SED."""
    band = np.clip(np.asarray(band, dtype=float), 0.0, 1.0)
    valid = band > 0.0
    if not np.any(valid):
        raise ValueError("The magnitude bandpass has zero transmission.")
    lo, hi = band_wave[valid].min(), band_wave[valid].max()
    grid = np.linspace(lo.to_value(u.AA), hi.to_value(u.AA), 4096) * u.AA
    source = np.interp(grid.to_value(u.AA), wavelength.to_value(u.AA),
                       f_lambda.to_value(FLAM_UNIT), left=0.0, right=0.0) * FLAM_UNIT
    response = np.interp(grid.to_value(u.AA), band_wave.to_value(u.AA), band, left=0.0, right=0.0)
    if int(detector_type) not in {0, 1}:
        raise ValueError("SVO DetectorType must be 0 (energy) or 1 (photon).")
    weighting = grid if int(detector_type) == 1 else np.ones(grid.size)
    integral = trapz_quantity(source * response * weighting, grid)
    if not np.isfinite(integral.value) or integral <= 0 * integral.unit:
        raise ValueError(
            "The template has no positive flux in the selected magnitude bandpass "
            f"(template {wavelength.min().value:.0f}-{wavelength.max().value:.0f} Å; "
            f"band {lo.value:.0f}-{hi.value:.0f} Å).")
    return grid, response, integral


def synthetic_magnitude(template, magnitude_band, zero_point_jy=3631.0, detector_type=1):
    """Synthetic magnitude using the SVO energy/photon response convention."""
    wave, values = as_angstrom_curve(template, "template spectrum")
    f_lambda = values * FLAM_UNIT
    band_wave, band = as_angstrom_curve(magnitude_band, "magnitude bandpass")
    grid, response, measured = _band_integral(wave, f_lambda, band_wave, band, detector_type)
    weighting = grid if int(detector_type) == 1 else np.ones(grid.size)
    reference = trapz_quantity(magnitude_f_lambda(grid, zero_point_jy) * response * weighting, grid)
    return float(-2.5 * np.log10((measured / reference).to_value(u.dimensionless_unscaled)))


def covered_response_fraction(template_wave_aa, band_wave_aa, band_response, detector_type=1):
    """Response-weighted fraction of a bandpass covered by a template's grid.

    The weighting matches the synthetic-magnitude integrand (photon or
    energy), so the returned fraction is exactly the share of the calibration
    integral that the template can actually supply.
    """
    band = np.clip(np.asarray(band_response, dtype=float), 0.0, 1.0)
    valid = band > 0.0
    if not np.any(valid):
        raise ValueError("The magnitude bandpass has zero transmission.")
    band_wave = np.asarray(band_wave_aa, dtype=float)
    grid = np.linspace(band_wave[valid].min(), band_wave[valid].max(), 2048)
    response = np.interp(grid, band_wave, band, left=0.0, right=0.0)
    weighting = grid if int(detector_type) == 1 else np.ones(grid.size)
    integrand = response * weighting
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    total = integrate(integrand, grid)
    template_wave = np.asarray(template_wave_aa, dtype=float)
    covered = (grid >= template_wave.min()) & (grid <= template_wave.max())
    if not np.any(covered):
        return 0.0
    return float(integrate(np.where(covered, integrand, 0.0), grid) / total)


CALIBRATION_COVERAGE_MINIMUM = 0.99


def _require_calibration_coverage(template_wave, band_wave, band_response, detector_type, name):
    """Calibration bands must be essentially fully covered by the template.

    Zero-filling is acceptable on the observing side, where truncation only
    loses counts; in the calibration integral it silently rescales the whole
    spectrum (a truncated synthetic magnitude comes out too faint and the
    template is scaled up to compensate), so it stays a hard requirement.
    """
    fraction = covered_response_fraction(template_wave.to_value(u.AA), band_wave.to_value(u.AA),
                                         band_response, detector_type)
    if fraction < CALIBRATION_COVERAGE_MINIMUM:
        raise ValueError(
            f"The template covers only {100.0 * fraction:.0f}% of the {name} response "
            f"({template_wave.min().value:.0f}-{template_wave.max().value:.0f} Å). A partially "
            "covered calibration band silently biases the flux scaling; select a reference "
            "filter inside the template coverage or a template with wider coverage.")


def calibrated_template_magnitude(template, target_magnitude, reference_band,
                                  reference_zero_point_jy=3631.0,
                                  template_mv0=0.0,
                                  visual_band=None, visual_zero_point_jy=3631.0,
                                  reference_detector_type=1, visual_detector_type=1):
    """Scale a calibrated template to the target reference measurement.

    The catalogue ``template_mv0`` is the visual magnitude represented by the
    input file.  The first factor below reproduces the original Fortran
    convention, which converts it to a visual-zero distribution.  A synthetic
    colour from that distribution then allows a user magnitude in *any*
    reference filter, independently of the observing filter.
    """
    wave, values = as_angstrom_curve(template, "template spectrum")
    visual_zero_template = values * (10.0 ** (0.4 * float(template_mv0)))
    if visual_band is None:
        # Backward-compatible V-only usage: the supplied magnitude is visual.
        reference_colour = 0.0
    else:
        ref_wave, ref_response = as_angstrom_curve(reference_band, "reference magnitude bandpass")
        vis_wave, vis_response = as_angstrom_curve(visual_band, "visual magnitude bandpass")
        same_response = (int(reference_detector_type) == int(visual_detector_type)
                         and ref_wave.shape == vis_wave.shape and np.allclose(ref_wave.value, vis_wave.value)
                         and np.allclose(ref_response, vis_response))
        if same_response:
            # The original Fortran V case needs no numerical convolution at
            # all.  For the same response in a different magnitude system,
            # only the two zero points supply the constant colour offset.
            reference_colour = -2.5 * np.log10(float(visual_zero_point_jy) /
                                                 float(reference_zero_point_jy))
        else:
            _require_calibration_coverage(wave, ref_wave, ref_response,
                                          reference_detector_type, "reference magnitude filter")
            _require_calibration_coverage(wave, vis_wave, vis_response,
                                          visual_detector_type, "visual calibration filter")
            ref_mag = synthetic_magnitude(np.column_stack((wave.value, visual_zero_template)),
                                          reference_band, reference_zero_point_jy, reference_detector_type)
            visual_mag = synthetic_magnitude(np.column_stack((wave.value, visual_zero_template)),
                                             visual_band, visual_zero_point_jy, visual_detector_type)
            reference_colour = ref_mag - visual_mag
    scale = 10.0 ** (-0.4 * (float(target_magnitude) - reference_colour))
    return wave, visual_zero_template * scale * FLAM_UNIT


def normalise_template_magnitude(template, magnitude, magnitude_band, zero_point_jy=3631.0):
    """Compatibility wrapper for callers not yet passing template ``mv0``."""
    return calibrated_template_magnitude(template, magnitude, magnitude_band, zero_point_jy)


def normalise_template_ab(template, magnitude, magnitude_band):
    """Backward-compatible AB wrapper."""
    return normalise_template_magnitude(template, magnitude, magnitude_band, 3631.0)


def collecting_area(telescope):
    diameter = float(telescope["diameter_mm"]) * u.mm
    obstruction = float(telescope.get("obstruction_mm", 0.0)) * u.mm
    if diameter <= 0 * u.mm or obstruction < 0 * u.mm or obstruction >= diameter:
        raise ValueError("Require diameter > obstruction >= 0.")
    return (np.pi / 4.0 * (diameter**2 - obstruction**2)).to(u.cm**2)


def atmospheric_transmission(wavelength, atmosphere):
    """Transmission at the current airmass.

    ``transmission_curve`` is a zenith transmission curve; it is raised to X,
    not applied once and then supplemented by an unrelated empirical
    extinction curve.
    """
    airmass = float(atmosphere.get("airmass", 1.0))
    if not np.isfinite(airmass) or airmass < 1.0:
        raise ValueError("Airmass must be finite and >= 1.")
    curve = atmosphere.get("transmission_curve")
    if curve is None:
        return np.ones(wavelength.size)
    # Outside the curve's tabulated range the atmosphere is certainly not
    # opaque: extend the edge values rather than zero-filling.
    zenith = interpolate_edge_extended(wavelength.to_value(u.AA), curve,
                                       "atmospheric transmission curve", clip=(0.0, 1.0))
    return np.clip(zenith, 0.0, 1.0) ** airmass


def instrument_transmission(wavelength, telescope):
    """Optional wavelength-dependent optics/instrument transmission curve."""
    curve = telescope.get("throughput_curve")
    if curve is None:
        return np.ones(wavelength.size)
    return interpolate_checked(wavelength.to_value(u.AA), curve, "instrument throughput curve", clip=(0.0, 1.0))


def electron_rate(wavelength, f_lambda, throughput, qe, telescope, atmosphere):
    """Detected electron rate integrated over a wavelength grid."""
    area = collecting_area(telescope)
    efficiency = float(telescope.get("efficiency", 1.0))
    if not 0.0 <= efficiency <= 1.0:
        raise ValueError("Telescope efficiency must be in [0, 1].")
    photon_energy = (h * c / wavelength).to(u.erg)
    integrand = f_lambda * np.clip(throughput, 0.0, 1.0) * np.clip(qe, 0.0, 1.0)
    integrand *= (atmospheric_transmission(wavelength, atmosphere) * instrument_transmission(wavelength, telescope)
                  * area * efficiency / photon_energy)
    return trapz_quantity(integrand, wavelength).to(1 / u.s)


FWHM_TO_SIGMA = 2.354820045


def _moffat_alpha(seeing_arcsec, beta):
    """Moffat core radius alpha from the FWHM: FWHM = 2 alpha sqrt(2^(1/beta) - 1).

    ``seeing_arcsec`` may be a scalar or an array (wavelength-dependent seeing).
    """
    beta = float(beta)
    if beta <= 1.0:
        raise ValueError("Moffat beta must exceed 1 for a finite total flux.")
    return np.asarray(seeing_arcsec, dtype=float) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def psf_encircled_energy(radius_arcsec, seeing_arcsec, psf_model="gaussian", moffat_beta=2.5):
    """Circular-aperture encircled energy for a Gaussian or Moffat PSF.

    Moffat: EE(r) = 1 - [1 + (r/alpha)^2]^(1-beta), the analytic integral of
    the circular Moffat profile.  Small beta (2.5-3) reproduces the extended
    wings of real seeing-limited images that a Gaussian underestimates.
    """
    if radius_arcsec <= 0 or seeing_arcsec <= 0:
        raise ValueError("Aperture radius and seeing must be positive.")
    model = str(psf_model).strip().lower()
    if model == "gaussian":
        sigma = seeing_arcsec / FWHM_TO_SIGMA
        return 1.0 - np.exp(-0.5 * (radius_arcsec / sigma) ** 2)
    if model == "moffat":
        alpha = _moffat_alpha(seeing_arcsec, moffat_beta)
        return 1.0 - (1.0 + (radius_arcsec / alpha) ** 2) ** (1.0 - float(moffat_beta))
    raise ValueError("PSF model must be 'gaussian' or 'moffat'.")


def psf_slit_throughput(width_arcsec, seeing_arcsec, psf_model="gaussian", moffat_beta=2.5,
                        offset_arcsec=0.0):
    """Fraction of PSF light through an infinite slit, optionally decentred.

    The Gaussian case is the erf coupling integral.  For a circular Moffat
    profile the one-dimensional marginal is exactly a Student-t distribution
    with nu = 2 beta - 2 degrees of freedom and scale alpha/sqrt(nu), so the
    throughput uses its CDF with no numerical integration.
    ``offset_arcsec`` decentres the PSF (e.g. atmospheric-dispersion drift).
    """
    seeing = np.asarray(seeing_arcsec, dtype=float)
    if width_arcsec <= 0 or np.any(seeing <= 0):
        raise ValueError("Slit width and seeing must be positive.")
    half = 0.5 * float(width_arcsec)
    offset = np.abs(np.asarray(offset_arcsec, dtype=float))
    model = str(psf_model).strip().lower()
    if model == "gaussian":
        from scipy.special import erf
        sigma = seeing / FWHM_TO_SIGMA
        scale = np.sqrt(2.0) * sigma
        result = 0.5 * (erf((half - offset) / scale) + erf((half + offset) / scale))
    elif model == "moffat":
        from scipy.stats import t as student_t
        beta = float(moffat_beta)
        alpha = _moffat_alpha(seeing, beta)
        nu = 2.0 * beta - 2.0
        scale = alpha / np.sqrt(nu)
        result = (student_t.cdf((half - offset) / scale, df=nu)
                  - student_t.cdf((-half - offset) / scale, df=nu))
    else:
        raise ValueError("PSF model must be 'gaussian' or 'moffat'.")
    return float(result) if np.ndim(result) == 0 else np.asarray(result, dtype=float)


def gaussian_encircled_energy(radius_arcsec, seeing_arcsec):
    """Backward-compatible Gaussian wrapper for :func:`psf_encircled_energy`."""
    return psf_encircled_energy(radius_arcsec, seeing_arcsec, "gaussian")


def slit_throughput(width_arcsec, seeing_arcsec):
    """Backward-compatible Gaussian wrapper for :func:`psf_slit_throughput`."""
    return psf_slit_throughput(width_arcsec, seeing_arcsec, "gaussian")


def snr(source_e, sky_e, dark_e, read_noise_e, n_pixels, extra_variance_e2=0.0):
    """CCD-equation S/N with an optional extra variance term [e^-2].

    ``extra_variance_e2`` collects non-Poisson contributions such as
    scintillation and ADC quantization noise.
    """
    variance = source_e + sky_e + dark_e + n_pixels * read_noise_e**2 + extra_variance_e2
    return source_e / np.sqrt(max(variance, 1e-300))
