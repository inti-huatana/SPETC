"""
atmospheric.py
Seeing, PSF, and atmospheric extinction calculations.

References:
- Young, A. T. (1974). Atmospheric Extinction
- Kolmogorov turbulence theory via Fried parameter r0
"""

import numpy as np
from scipy import special


# ============================================================================
# Physical constants
# ============================================================================

H_PLANCK = 6.62606876e-27   # erg*s
C_LIGHT = 2.99792458e10     # cm/s


# ============================================================================
# Seeing and PSF
# ============================================================================

def seeing_young(wavelength_um, h_turb_m):
    """
    Atmospheric seeing using Young (1974) formula.
    
    seeing(λ) = 0.364 * (λ/500nm)^(-0.2) * sqrt(h_turb / 8000)
    
    Parameters
    ----------
    wavelength_um : float or array
        Wavelength [μm]
    h_turb_m : float
        Turbulence scale height [m] (typical: 8000 m)
    
    Returns
    -------
    seeing_arcsec : float or array
        Seeing FWHM [arcsec]
    """
    wavelength_nm = wavelength_um * 1000
    factor = 0.364 * (wavelength_nm / 500.0)**(-0.2) * np.sqrt(h_turb_m / 8000.0)
    return factor


def plate_scale_arcsec_per_pixel(focal_length_mm, pixel_size_um):
    """
    Geometric plate scale, arcsec/pixel.

    plate_scale = 206265 * pixel_size_mm / focal_length_mm

    This is a purely geometric relation (focal-plane image scale) and
    does NOT depend on wavelength -- wavelength only affects the
    diffraction-limited component of the PSF, which is a separate,
    smaller effect not modeled here (seeing dominates in the optical/NIR
    for any ground-based aperture larger than ~10-15 cm).

    Parameters
    ----------
    focal_length_mm : float
    pixel_size_um : float

    Returns
    -------
    float, arcsec/pixel
    """
    pixel_size_mm = pixel_size_um * 1e-3
    return 206265.0 * pixel_size_mm / focal_length_mm


def psf_gaussian(seeing_arcsec, focal_length_mm, pixel_size_um, wavelength_um=None):
    """
    Gaussian PSF parameters derived from seeing.

    PSF FWHM [pix] = seeing_arcsec / plate_scale_arcsec_per_pixel

    Parameters
    ----------
    seeing_arcsec : float
        Atmospheric seeing FWHM [arcsec]
    focal_length_mm : float
        Focal length of telescope [mm]
    pixel_size_um : float
        Detector pixel size [um]
    wavelength_um : float, optional
        Unused (kept for backward-compatible call signatures); the
        geometric plate scale does not depend on wavelength. Previously
        this parameter was (incorrectly) used AS the plate scale's
        numerator instead of pixel_size_um, producing a plate scale off
        by many orders of magnitude and an unusably large PSF in pixels
        (and therefore near-zero slit throughput downstream).

    Returns
    -------
    fwhm_pixels : float
        PSF FWHM [pixels]
    sigma_pixels : float
        PSF standard deviation [pixels] (FWHM / 2.355)
    """
    plate_scale = plate_scale_arcsec_per_pixel(focal_length_mm, pixel_size_um)

    fwhm_pixels = seeing_arcsec / plate_scale
    sigma_pixels = fwhm_pixels / 2.355

    return fwhm_pixels, sigma_pixels


def slit_loss_rectangular(slit_width_um, psf_sigma_pixels, pixel_size_um):
    """
    Slit transmission for rectangular aperture with Gaussian PSF.

    For an infinite slit perpendicular to the dispersion direction:
        efficiency = erf(W / (2 * sqrt(2) * sigma))

    where W is the slit half-width in pixels and sigma is the PSF
    standard deviation in pixels.

    Parameters
    ----------
    slit_width_um : float
        Slit width [um] on the detector
    psf_sigma_pixels : float
        PSF standard deviation [pixels]
    pixel_size_um : float
        Detector pixel size [um] -- needed to convert slit_width_um into
        pixels. Previously this function took a "plate_scale_arcsec_pix"
        parameter it never actually used, and never converted
        slit_width_um to pixels at all (see the removed comments in the
        prior version, which said as much) -- it silently returned a
        throughput computed from a slit "width" still in microns compared
        against a PSF sigma in pixels, two incompatible units, only by
        coincidence not erroring out. This is now a proper unit
        conversion, matching the equivalent inline calculation in
        spectroscopy.py.

    Returns
    -------
    throughput : float
        Fraction of light transmitted [0..1]
    """
    slit_width_pixels = slit_width_um / pixel_size_um

    arg = slit_width_pixels / (2 * np.sqrt(2) * psf_sigma_pixels)
    arg = np.clip(arg, -10, 10)

    return special.erf(arg)


# ============================================================================
# Atmospheric Extinction
# ============================================================================

def airmass_extinction(airmass, wavelength_um, extinction_coeff=None):
    """
    Atmospheric extinction using Bouguer law.
    
    A(λ) = k(λ) * (X - 1)
    
    where X is airmass, k(λ) is extinction coefficient.
    
    Parameters
    ----------
    airmass : float
        Airmass (1.0 at zenith)
    wavelength_um : float
        Wavelength [μm]
    extinction_coeff : float, optional
        Extinction coefficient [mag/airmass]. If None, estimate from wavelength.
    
    Returns
    -------
    extinction_mag : float
        Extinction [mag]
    """
    if extinction_coeff is None:
        # Simple wavelength-dependent extinction
        # k(λ) ~ λ^(-0.4) for optical
        k_ext = 0.15 * (0.55 / wavelength_um)**0.4
    else:
        k_ext = extinction_coeff
    
    extinction = k_ext * (airmass - 1)
    return extinction


# ============================================================================
# Angle calculations (convenience)
# ============================================================================

def zenith_distance_from_altitude(altitude_deg):
    """Convert altitude to zenith distance."""
    return 90 - altitude_deg


def altitude_from_zenith_distance(zenith_distance_deg):
    """Convert zenith distance to altitude."""
    return 90 - zenith_distance_deg
