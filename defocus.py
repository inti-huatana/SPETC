"""
defocus.py
Defocused-PSF (donut) model for defocused photometry.

A defocused telescope images a point source as a broad "donut".  In the pure
geometric-optics limit that donut is a uniformly-illuminated annulus (the
projection of the pupil), but the image a camera actually records is that
annulus **convolved with the atmospheric point-spread function** -- the
seeing (and any guiding/tracking blur).  The convolution is what turns the
geometric annulus, with its unphysical vertical edges, into the smoothly
rounded, winged ring seen in real defocused frames.  This module builds the
realistic (seeing-convolved) profile and its encircled-energy curve, and
keeps the geometric annulus available as the ideal limit.

Geometry.  For a primary of diameter ``D`` and focal length ``F`` (focal
ratio ``N = F / D``), a defocus of ``delta`` gives a blur circle of full
diameter ``delta / N = delta * D / F`` on the focal plane, i.e. an outer
radius ``r_out = delta * D / (2 F)``.  A central obstruction of linear
fraction ``eps = obstruction / D`` (a classical Ritchey-Chretien or
Cassegrain, ``obstruction > 0``) removes the inner ``r_in = eps * r_out``; a
classical reflector modelled with ``obstruction = 0`` gives a filled disc.
Only ``|delta|`` matters, so intra- and extra-focal positions of equal
magnitude give identical images.

Realism and limits.  The dominant blur from the ground is the atmosphere, so
the model convolves the annulus with the selected seeing PSF (Gaussian or
Moffat) after adding any guiding rms in quadrature to the seeing FWHM.  It
still neglects diffraction ringing at the edges and optical aberrations
(spherical, coma), which are sub-dominant once the donut is much larger than
the seeing and diffraction scales -- the regime defocused photometry uses.
When the seeing is set to zero the profile collapses to the ideal geometric
annulus.
"""

import numpy as np

# Radians-to-arcsec, so that 1 um on the focal plane subtends
# (1e-3 mm / F_mm) rad -> ARCSEC_PER_RAD * 1e-3 / F_mm arcsec.
ARCSEC_PER_RAD = 206264.806247
# FWHM = 2 sqrt(2 ln 2) sigma for a Gaussian.
_FWHM_TO_SIGMA = 2.3548200450309493


def _donut_radii_um(defocus_um, diameter_mm, focal_length_mm, obstruction_mm):
    """Return (r_in, r_out) of the geometric donut on the focal plane [um]."""
    delta = abs(float(defocus_um))
    diameter = float(diameter_mm)
    focal = float(focal_length_mm)
    if diameter <= 0 or focal <= 0:
        raise ValueError("Telescope diameter and focal length must be positive.")
    if delta <= 0:
        raise ValueError("Defocus position must be non-zero to compute a defocused PSF.")
    r_out_um = delta * diameter / (2.0 * focal)
    eps = 0.0
    if float(obstruction_mm) > 0:
        eps = float(obstruction_mm) / diameter
        if not 0.0 <= eps < 1.0:
            raise ValueError("Obstruction diameter must be between 0 and the primary diameter.")
    r_in_um = eps * r_out_um
    return r_in_um, r_out_um


def defocus_encircled_energy(aperture_radius_arcsec, defocus_um, diameter_mm,
                             focal_length_mm, obstruction_mm):
    """Ideal *geometric* encircled-energy fraction (0..1) of a flat annulus
    within a circular aperture [arcsec radius].  This is the seeing-free
    limit; the realistic value comes from :func:`defocused_star_profile`."""
    r_in_um, r_out_um = _donut_radii_um(defocus_um, diameter_mm, focal_length_mm, obstruction_mm)
    arcsec_per_um = ARCSEC_PER_RAD * 1e-3 / float(focal_length_mm)
    r_ap_um = float(aperture_radius_arcsec) / arcsec_per_um
    if r_ap_um <= r_in_um:
        return 0.0
    if r_ap_um >= r_out_um:
        return 1.0
    return (r_ap_um**2 - r_in_um**2) / (r_out_um**2 - r_in_um**2)


def _seeing_kernel(radius_grid, fwhm_arcsec, psf_model, moffat_beta):
    """Normalised radial seeing kernel sampled on a 2-D radius grid."""
    model = str(psf_model).strip().lower()
    if model == "moffat":
        beta = float(moffat_beta)
        if beta <= 1.0:
            raise ValueError("Moffat beta must exceed 1.")
        alpha = fwhm_arcsec / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        kernel = (1.0 + (radius_grid / alpha) ** 2) ** (-beta)
    elif model == "gaussian":
        sigma = fwhm_arcsec / _FWHM_TO_SIGMA
        kernel = np.exp(-0.5 * (radius_grid / sigma) ** 2)
    else:
        raise ValueError("PSF model must be 'gaussian' or 'moffat'.")
    total = kernel.sum()
    if total <= 0:
        raise ValueError("Degenerate seeing kernel.")
    return kernel / total


def defocused_star_profile(defocus_um, diameter_mm, focal_length_mm, obstruction_mm,
                           pixel_size_um, seeing_arcsec, psf_model="gaussian",
                           moffat_beta=2.5, guiding_rms_arcsec=0.0, n_grid=384):
    """Realistic (seeing-convolved) defocused-star radial profile and its
    cumulative encircled energy.

    The geometric annulus is built on a 2-D grid and convolved with the
    selected atmospheric PSF (guiding rms added in quadrature to the seeing
    FWHM), then azimuthally averaged.  Returns a dict over a radius grid from
    0 to the field edge:
      ``radius_arcsec`` / ``radius_px`` : the radius axis in the two units;
      ``intensity_norm``               : azimuthal-mean intensity / its peak
                                         (rounded edges and wings, not a step);
      ``ee_percent``                   : cumulative encircled energy [%];
    plus the geometric annulus radii (``r_in_*``/``r_out_*``), the obstruction
    fraction ``epsilon``, the effective blur ``fwhm_arcsec`` used, and
    ``peak_sb_per_arcsec2`` -- the central surface brightness of the profile
    normalised to unit total flux, for the detector peak-pixel prediction.
    """
    r_in_um, r_out_um = _donut_radii_um(defocus_um, diameter_mm, focal_length_mm, obstruction_mm)
    focal = float(focal_length_mm)
    pixel = float(pixel_size_um)
    if pixel <= 0:
        raise ValueError("Pixel size must be positive.")
    arcsec_per_um = ARCSEC_PER_RAD * 1e-3 / focal
    r_in = r_in_um * arcsec_per_um
    r_out = r_out_um * arcsec_per_um

    seeing = max(float(seeing_arcsec), 0.0)
    guiding = max(float(guiding_rms_arcsec), 0.0)
    fwhm = float(np.hypot(seeing, _FWHM_TO_SIGMA * guiding)) if (seeing > 0 or guiding > 0) else 0.0

    # Field: the whole donut plus several seeing widths of wings.
    pad = 4.0 * fwhm if fwhm > 0 else 0.06 * r_out
    half = 1.05 * r_out + pad
    n_grid = int(max(96, min(n_grid, 512)))
    axis = np.linspace(-half, half, n_grid)
    cell = axis[1] - axis[0]
    xx, yy = np.meshgrid(axis, axis)
    radius2d = np.hypot(xx, yy)
    annulus = ((radius2d >= r_in) & (radius2d <= r_out)).astype(float)

    if fwhm > 0:
        from scipy.signal import fftconvolve
        kernel = _seeing_kernel(radius2d, fwhm, psf_model, moffat_beta)
        image = fftconvolve(annulus, kernel, mode="same")
        image = np.clip(image, 0.0, None)
    else:
        image = annulus

    # Azimuthal average and cumulative encircled energy in one radial binning.
    n_bins = n_grid // 2
    r_flat = radius2d.ravel()
    i_flat = image.ravel()
    bin_edges = np.linspace(0.0, half, n_bins + 1)
    counts, _ = np.histogram(r_flat, bins=bin_edges)
    flux_sum, _ = np.histogram(r_flat, bins=bin_edges, weights=i_flat)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_intensity = np.where(counts > 0, flux_sum / np.maximum(counts, 1), 0.0)
    peak = mean_intensity.max()
    intensity_norm = mean_intensity / peak if peak > 0 else mean_intensity
    total_flux = image.sum()
    ee = np.cumsum(flux_sum) / total_flux if total_flux > 0 else np.zeros_like(centers)

    plate_scale_arcsec_px = arcsec_per_um * pixel
    peak_sb = float(image.max() / (total_flux * cell**2)) if total_flux > 0 else 0.0

    return {
        "radius_arcsec": centers,
        "radius_px": centers / plate_scale_arcsec_px,
        "intensity_norm": intensity_norm,
        "ee_percent": 100.0 * np.clip(ee, 0.0, 1.0),
        "ee_fraction": np.clip(ee, 0.0, 1.0),
        "epsilon": (r_in / r_out) if r_out > 0 else 0.0,
        "r_in_arcsec": r_in,
        "r_out_arcsec": r_out,
        "r_in_px": r_in / plate_scale_arcsec_px,
        "r_out_px": r_out / plate_scale_arcsec_px,
        "fwhm_arcsec": fwhm,
        "peak_sb_per_arcsec2": peak_sb,
    }


def defocus_capture_and_peak(aperture_radius_arcsec, defocus_um, diameter_mm,
                             focal_length_mm, obstruction_mm, pixel_size_um,
                             seeing_arcsec, psf_model="gaussian", moffat_beta=2.5,
                             guiding_rms_arcsec=0.0, profile=None):
    """Realistic captured-light fraction within an aperture and the peak
    surface brightness (per arcsec^2, unit total flux), for the engine.

    Pass a pre-computed ``profile`` to avoid recomputation, else one is built.
    """
    if profile is None:
        profile = defocused_star_profile(defocus_um, diameter_mm, focal_length_mm,
                                         obstruction_mm, pixel_size_um, seeing_arcsec,
                                         psf_model, moffat_beta, guiding_rms_arcsec)
    captured = float(np.interp(float(aperture_radius_arcsec),
                               profile["radius_arcsec"], profile["ee_fraction"],
                               left=0.0, right=1.0))
    return captured, profile["peak_sb_per_arcsec2"]
