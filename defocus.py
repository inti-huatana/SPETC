"""
defocus.py
Geometric defocused-PSF (donut) model for defocused photometry.

When the detector is displaced by a distance delta from the focal plane, the
geometric image of a point source is a uniformly-illuminated annulus (a
"donut"): light from the whole pupil is spread over a disc whose diameter is
set by the focal ratio, with a central hole imposed by the telescope's central
obstruction.  This is the classical model behind defocused photometry (bright
targets spread over many pixels to beat scintillation and flat-field errors).

For an aperture of diameter ``D`` and focal length ``F`` (focal ratio
``N = F / D``), a defocus of ``delta`` gives a blur circle of full diameter
``delta / N = delta * D / F`` on the focal plane, i.e. an outer radius

    r_out = delta * D / (2 F).

A central obstruction of linear fraction ``eps = obstruction / D`` (a classical
Ritchey-Chretien or Cassegrain, ``obstruction > 0``) removes the inner
``r_in = eps * r_out``; a classical reflector modelled with ``obstruction = 0``
gives a filled disc (``r_in = 0``).  The pupil is uniformly illuminated, so in
the geometric limit the intensity is constant across the annulus and the
encircled energy is analytic:

    EE(r) = 0                                    r < r_in
          = (r^2 - r_in^2) / (r_out^2 - r_in^2)  r_in <= r <= r_out
          = 1                                    r > r_out.

This is a geometric-optics model: it neglects diffraction ringing at the donut
edges, optical aberrations (small on-axis for a paraboloid or an RC) and the
seeing that softens the edges -- all negligible when the donut is much larger
than the seeing/diffraction scale, which is the regime defocused photometry is
used in.
"""

import numpy as np

# Radians-to-arcsec, so that 1 um on the focal plane subtends
# (1e-3 mm / F_mm) rad -> ARCSEC_PER_RAD * 1e-3 / F_mm arcsec.
ARCSEC_PER_RAD = 206264.806247


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
    """Fraction (0..1) of donut light within a circular aperture [arcsec radius]."""
    r_in_um, r_out_um = _donut_radii_um(defocus_um, diameter_mm, focal_length_mm, obstruction_mm)
    arcsec_per_um = ARCSEC_PER_RAD * 1e-3 / float(focal_length_mm)
    r_ap_um = float(aperture_radius_arcsec) / arcsec_per_um
    if r_ap_um <= r_in_um:
        return 0.0
    if r_ap_um >= r_out_um:
        return 1.0
    return (r_ap_um**2 - r_in_um**2) / (r_out_um**2 - r_in_um**2)


def defocus_donut_profile(defocus_um, diameter_mm, focal_length_mm, obstruction_mm,
                          pixel_size_um, n_samples=600, margin=1.15):
    """Radial profile and cumulative encircled energy of the geometric donut.

    Returns a dict with, over a radius grid from 0 to ``margin * r_out``:
      ``radius_px``, ``radius_arcsec`` : the radius axis in the two units;
      ``intensity_norm``              : intensity normalised to the peak (1 in
                                        the annulus, 0 in the hole and outside);
      ``ee_percent``                  : cumulative encircled energy [%];
    plus the scalar donut geometry (``r_in_*``/``r_out_*`` in um, arcsec and px)
    and the obstruction fraction ``epsilon``.
    """
    r_in_um, r_out_um = _donut_radii_um(defocus_um, diameter_mm, focal_length_mm, obstruction_mm)
    focal = float(focal_length_mm)
    pixel = float(pixel_size_um)
    if pixel <= 0:
        raise ValueError("Pixel size must be positive.")
    arcsec_per_um = ARCSEC_PER_RAD * 1e-3 / focal

    r_max_um = margin * r_out_um
    radius_um = np.linspace(0.0, r_max_um, int(n_samples))
    intensity = np.where((radius_um >= r_in_um) & (radius_um <= r_out_um), 1.0, 0.0)
    span = r_out_um**2 - r_in_um**2
    ee = np.clip((radius_um**2 - r_in_um**2) / span, 0.0, 1.0)
    ee[radius_um < r_in_um] = 0.0

    return {
        "radius_um": radius_um,
        "radius_px": radius_um / pixel,
        "radius_arcsec": radius_um * arcsec_per_um,
        "intensity_norm": intensity,
        "ee_percent": 100.0 * ee,
        "epsilon": (r_in_um / r_out_um) if r_out_um > 0 else 0.0,
        "r_in_um": r_in_um,
        "r_out_um": r_out_um,
        "r_in_px": r_in_um / pixel,
        "r_out_px": r_out_um / pixel,
        "r_in_arcsec": r_in_um * arcsec_per_um,
        "r_out_arcsec": r_out_um * arcsec_per_um,
    }
