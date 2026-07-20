"""Broad-band night, twilight and daylight sky model used by the ETC.

The constants are the supplied ING model values: solar minima, dark-sky
magnitudes/extinction, and the Weaver (1947) V-band daylight table.  This is
an empirical broad-band model, deliberately not a substitute for a future
site-specific spectral sky model.
"""

from datetime import datetime
import numpy as np

from sky_brightness import (SOLAR_MINIMA, SOLAR_CYCLE as SOLAR_CYCLE_YR,
                            dark_sky_position_correction_mag)

SKY_WAVELENGTH_AA = np.array([3650.0, 4450.0, 5510.0, 6580.0, 8060.0, 10000.0, 12500.0, 16500.0, 22000.0])
SKY_MAG_VEGA = np.array([22.0, 22.7, 21.9, 21.0, 20.0, 18.8, 16.1, 14.7, 12.5])
SKY_EXT_MAG = np.array([0.550, 0.250, 0.150, 0.090, 0.060, 0.050, 0.100, 0.110, 0.070])
DAYLIGHT_SUN_ZD = np.array([0.0, 30.0, 60.0])
DAYLIGHT_SKY_ZD = np.array([0.0, 30.0, 60.0, 80.0])
# Daylight sky-brightness table in the units it was supplied in (the Weaver
# lineage of the original Fortran code).  Read directly as V mag/arcsec^2 the
# values (8.5-9.2) are ~4.5 magnitudes too FAINT: the physical clear daytime
# zenith sky is ~3000-8000 cd/m^2, i.e. V ~ 3.5-4.5 mag/arcsec^2 against the
# night-sky anchor 21.9 mag/arcsec^2 ~ 2e-4 cd/m^2.  The table's *shape* with
# Sun and sky zenith distance is kept; the constant below re-anchors its
# brightest entry (Sun at the zenith, target at the zenith, 8.50) to the
# physical V = 4.0 mag/arcsec^2.  If the original Weaver (1947) quantity and
# unit are ever re-derived exactly, replace this anchor with the exact
# conversion.
DAYLIGHT_V_MAG_SUPPLIED = np.array([
    [8.73, 8.74, 8.85, 9.17],
    [8.67, 8.69, 8.84, 9.19],
    [8.50, 8.56, 8.80, 9.20],
])
DAYLIGHT_ANCHOR_V_MAG = 4.0
DAYLIGHT_ANCHOR_OFFSET_MAG = DAYLIGHT_ANCHOR_V_MAG - DAYLIGHT_V_MAG_SUPPLIED.min()
DAYLIGHT_V_MAG = DAYLIGHT_V_MAG_SUPPLIED + DAYLIGHT_ANCHOR_OFFSET_MAG


def decimal_year(value):
    start = datetime(value.year, 1, 1)
    end = datetime(value.year + 1, 1, 1)
    return value.year + (value - start).total_seconds() / (end - start).total_seconds()


def solar_cycle_activity(value):
    """ING-model airglow scale: 0.8 at a minimum, 2.0 at maximum."""
    year = decimal_year(value) if isinstance(value, datetime) else float(value)
    reference = SOLAR_MINIMA[-1]
    if year <= SOLAR_MINIMA[0]:
        phase = ((year - SOLAR_MINIMA[0]) % SOLAR_CYCLE_YR) / SOLAR_CYCLE_YR
    elif year >= SOLAR_MINIMA[-1]:
        phase = ((year - reference) % SOLAR_CYCLE_YR) / SOLAR_CYCLE_YR
    else:
        phase = 0.0
        for low, high in zip(SOLAR_MINIMA[:-1], SOLAR_MINIMA[1:]):
            if low <= year <= high:
                phase = (year - low) / (high - low)
                break
    return 0.8 + 1.2 * phase


def _daylight_v_mag(sun_altitude_deg, target_altitude_deg):
    sun_zd = np.clip(90.0 - float(sun_altitude_deg), DAYLIGHT_SUN_ZD[0], DAYLIGHT_SUN_ZD[-1])
    target_zd = np.clip(90.0 - float(target_altitude_deg), DAYLIGHT_SKY_ZD[0], DAYLIGHT_SKY_ZD[-1])
    at_target_zd = np.array([np.interp(target_zd, DAYLIGHT_SKY_ZD, row) for row in DAYLIGHT_V_MAG])
    return float(np.interp(sun_zd, DAYLIGHT_SUN_ZD, at_target_zd))


def sky_magnitude_vega(pivot_wavelength_aa, utc_datetime, target_altitude_deg, target_airmass,
                        sun_altitude_deg, ecliptic_lat_deg=90.0, galactic_lat_deg=90.0,
                        solar_elongation_deg=180.0):
    """Estimated observed sky surface brightness in Vega mag/arcsec².

    Below -18° *geometric* Sun altitude this is the supplied dark-sky table
    with a position-dependent correction: van Rhijn airglow scaled by the
    solar-cycle activity and attenuated along the slant path, zodiacal light
    at the field's |ecliptic latitude| and solar elongation, and integrated
    starlight at its |galactic latitude|.  Between -18° and the horizon, dark
    and daylight fluxes are blended smoothly.  Above the horizon, the
    re-anchored daylight table is used; the supplied broad-band dark-sky
    colour relation is interpolated to the selected filter pivot.  Twilight
    thresholds are conventionally defined on the geometric (unrefracted) Sun
    altitude; pass that value here.
    """
    wavelength = float(pivot_wavelength_aa)
    base_mag = float(np.interp(wavelength, SKY_WAVELENGTH_AA, SKY_MAG_VEGA))
    v_base = float(np.interp(5510.0, SKY_WAVELENGTH_AA, SKY_MAG_VEGA))
    colour = base_mag - v_base
    activity = solar_cycle_activity(utc_datetime)
    zenith_dist = 90.0 - float(np.clip(target_altitude_deg, 0.0, 90.0))
    night_mag = base_mag + dark_sky_position_correction_mag(
        zenith_dist, ecliptic_lat_deg, galactic_lat_deg, solar_activity=activity,
        solar_elongation_deg=solar_elongation_deg)
    daylight_mag = _daylight_v_mag(max(float(sun_altitude_deg), 0.0), target_altitude_deg) + colour
    if sun_altitude_deg >= 0.0:
        return daylight_mag
    if sun_altitude_deg <= -18.0:
        return night_mag
    # Twilight bridge between the Sun on the horizon and the end of
    # astronomical twilight.  The blend is linear in *magnitude* with Sun
    # altitude, not in flux: a flux blend is completely dominated by the very
    # bright daylight anchor even a few degrees below the horizon (it would
    # put the sky near 6 mag/arcsec2 at Sun -15 deg), whereas the observed
    # twilight brightness falls roughly one magnitude per degree of Sun
    # depression.  Magnitude-linear interpolation is monotonic, matches both
    # endpoints, and reproduces the familiar ~1 mag/deg twilight gradient.
    blend = (float(sun_altitude_deg) + 18.0) / 18.0   # 1 at horizon, 0 at -18 deg
    return float((1.0 - blend) * night_mag + blend * daylight_mag)
