"""Scintillation, atmospheric dispersion and digitization noise terms.

References
----------
* Young A.T. (1967) AJ 72, 747: scintillation power law; the Osborn et al.
  (2015, MNRAS 452, 1707) revision keeps the same functional form with a
  site-dependent coefficient.
* Filippenko A.V. (1982) PASP 94, 715: refractive index of air and
  differential atmospheric refraction.
* Janesick, Scientific Charge-Coupled Devices: quantization noise q/sqrt(12).
"""

import numpy as np
import astropy.units as u

# Scale height of the atmospheric turbulence relevant to scintillation [m].
SCINTILLATION_SCALE_HEIGHT_M = 8000.0


def scintillation_fractional_rms(diameter_mm, airmass, elevation_m, t_exp_s,
                                 coefficient=0.09):
    """Young (1967) fractional scintillation rms for a single aperture.

    sigma/I = C d^(-2/3) X^1.75 exp(-h/h0) / sqrt(2 t) with d in cm.  The
    default C=0.09 is Young's classical value; Osborn et al. (2015) find
    median site coefficients of similar size (their C_Y ~ 1.3-1.6 corrects
    the same formula multiplicatively and can be folded into ``coefficient``).
    """
    diameter_cm = (float(diameter_mm) * u.mm).to_value(u.cm)
    if diameter_cm <= 0 or t_exp_s <= 0:
        raise ValueError("Aperture diameter and exposure time must be positive.")
    x = max(float(airmass), 1.0)
    return (float(coefficient) * diameter_cm ** (-2.0 / 3.0) * x ** 1.75
            * np.exp(-float(elevation_m) / SCINTILLATION_SCALE_HEIGHT_M)
            / np.sqrt(2.0 * float(t_exp_s)))


def scintillation_variance_e2(source_e, diameter_mm, airmass, elevation_m, t_exp_s,
                              coefficient=0.09):
    """Scintillation variance in electrons^2 for detected source electrons."""
    fraction = scintillation_fractional_rms(diameter_mm, airmass, elevation_m, t_exp_s,
                                            coefficient)
    return (np.asarray(source_e, dtype=np.float64) * fraction) ** 2


def scintillation_variance_rate_e2_s(source_rate_e_s, diameter_mm, airmass, elevation_m,
                                     coefficient=0.09):
    """Scintillation variance accumulation rate [e-^2 per second].

    With sigma/I = C (2t)^{-1/2} the variance of S_rate * t electrons is
    (S_rate C)^2 t / 2 - *linear* in time, exactly like a Poisson term, so
    closed-form exposure solvers can absorb it into the total rate.  It is
    also independent of how the time is split into frames (scintillation
    averages down with total time).
    """
    diameter_cm = (float(diameter_mm) * u.mm).to_value(u.cm)
    if diameter_cm <= 0:
        raise ValueError("Aperture diameter must be positive.")
    x = max(float(airmass), 1.0)
    c_factor = (float(coefficient) * diameter_cm ** (-2.0 / 3.0) * x ** 1.75
                * np.exp(-float(elevation_m) / SCINTILLATION_SCALE_HEIGHT_M))
    return (float(source_rate_e_s) * c_factor) ** 2 / 2.0


def effective_seeing_arcsec(seeing_zenith_v_arcsec, wavelength_aa, airmass,
                            reference_wavelength_aa=5000.0):
    """Seeing FWHM scaled to wavelength and airmass (Kolmogorov turbulence).

    ``FWHM(lambda, X) = seeing_V(zenith) * X^0.6 * (lambda/5000)^(-0.2)``,
    the standard Fried-parameter scaling (r0 ~ lambda^1.2 cos(z)^0.6, seeing
    ~ lambda/r0).  This is the exact form used by ETC-42
    (``calculator/psf/Seeing.java``, CeSAM/LAM); the reference seeing is the
    zenith value in V.  Blue light is more blurred than red, and the seeing
    degrades as the target descends towards the horizon.
    """
    seeing = float(seeing_zenith_v_arcsec)
    if seeing <= 0:
        raise ValueError("Reference seeing must be positive.")
    x = max(float(airmass), 1.0)
    scale = x ** 0.6 * (np.asarray(wavelength_aa, dtype=float)
                        / float(reference_wavelength_aa)) ** (-0.2)
    return seeing * scale


# Parametric telluric absorption bands: centre [A], Gaussian sigma [A],
# zenith depth (fractional absorption at band centre, X = 1) and curve-of-
# growth exponent p in depth(X) = depth * X^p (p = 1 for optically thin
# H2O lines, p ~ 0.55 for the saturated O2 bands).  Values approximate a
# low-resolution smoothing of a sea-level transmission model; a measured
# site spectrum can replace this model where per-line accuracy matters.
TELLURIC_BANDS = (
    # (centre_aa, sigma_aa, zenith_depth, growth_exponent)
    (6280.0, 25.0, 0.06, 0.55),   # O2 gamma band
    (6870.0, 30.0, 0.35, 0.55),   # O2 B band
    (7186.0, 60.0, 0.15, 1.0),    # H2O
    (7605.0, 30.0, 0.75, 0.55),   # O2 A band
    (8164.0, 70.0, 0.20, 1.0),    # H2O
    (8946.0, 90.0, 0.35, 1.0),    # H2O
    (9400.0, 110.0, 0.55, 1.0),   # H2O
)


def telluric_transmission(wavelength_aa, airmass):
    """Smoothed telluric O2/H2O band transmission at the current airmass.

    Each band is a Gaussian absorption profile of the stated centre, width
    and zenith depth; the depth grows with airmass as X^p (curve of growth:
    linear for unsaturated H2O, ~sqrt for the saturated O2 A/B bands).
    Below ~6200 A the model returns 1 (the visible telluric bands are weak).
    This is a band-averaged model for S/N budgeting at the ETC's resolution,
    not a line-by-line radiative transfer.
    """
    wave = np.asarray(wavelength_aa, dtype=np.float64)
    x = max(float(airmass), 1.0)
    transmission = np.ones_like(wave)
    for centre, sigma, depth, growth in TELLURIC_BANDS:
        band_depth = min(depth * x ** growth, 0.98)
        transmission *= 1.0 - band_depth * np.exp(-0.5 * ((wave - centre) / sigma) ** 2)
    return np.clip(transmission, 0.0, 1.0)


def digitization_noise_e(gain_e_adu):
    """Quantization noise of the ADC in electrons rms per pixel: g/sqrt(12)."""
    gain = float(gain_e_adu)
    if gain <= 0:
        raise ValueError("Detector gain must be positive.")
    return gain / np.sqrt(12.0)


def refractive_index_minus_one(wavelength_aa, pressure_hpa=1013.25, temperature_c=15.0,
                               water_vapour_hpa=8.0):
    """(n - 1) of air at the supplied conditions (Filippenko 1982, eqs. 1-3)."""
    wavelength_um = (np.asarray(wavelength_aa, dtype=np.float64) * u.AA).to_value(u.um)
    if np.any(wavelength_um <= 0.2):
        raise ValueError("Refractive index formula is valid above 2000 Angstrom.")
    sigma2 = (1.0 / wavelength_um) ** 2
    n_stp = (64.328 + 29498.1 / (146.0 - sigma2) + 255.4 / (41.0 - sigma2)) * 1e-6
    pressure_mmhg = float(pressure_hpa) * 0.750062
    temperature = float(temperature_c)
    pt_factor = (pressure_mmhg * (1.0 + (1.049 - 0.0157 * temperature) * 1e-6 * pressure_mmhg)
                 / (720.883 * (1.0 + 0.003661 * temperature)))
    water_mmhg = float(water_vapour_hpa) * 0.750062
    water_term = ((0.0624 - 0.000680 * sigma2) / (1.0 + 0.003661 * temperature)
                  * water_mmhg) * 1e-6
    return n_stp * pt_factor - water_term


def differential_refraction_arcsec(wavelength_aa, reference_wavelength_aa, airmass,
                                   pressure_hpa=1013.25, temperature_c=15.0,
                                   water_vapour_hpa=8.0):
    """Atmospheric dispersion Delta R(lambda) relative to a reference [arcsec].

    R(lambda) ~ 206265 (n(lambda) - 1) tan z, so the differential term is
    206265 [n(lambda) - n(ref)] tan z with tan z = sqrt(X^2 - 1).  Positive
    values displace blue light towards the zenith relative to the reference.
    """
    x = max(float(airmass), 1.0)
    tan_z = np.sqrt(x * x - 1.0)
    n_lambda = refractive_index_minus_one(wavelength_aa, pressure_hpa, temperature_c,
                                          water_vapour_hpa)
    n_ref = refractive_index_minus_one(float(reference_wavelength_aa), pressure_hpa,
                                       temperature_c, water_vapour_hpa)
    return 206265.0 * (n_lambda - n_ref) * tan_z
