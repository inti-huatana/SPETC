"""Broad-band night, twilight and daylight sky model used by the ETC.

The constants are the supplied ING model values: solar minima, dark-sky
magnitudes/extinction, and the Weaver (1947) V-band daylight table.  This is
an empirical broad-band model, deliberately not a substitute for a future
site-specific spectral sky model.
"""

from datetime import datetime
import numpy as np


SOLAR_MINIMA = np.array([1901.04, 1913.62, 1924.05, 1933.63, 1944.28, 1954.04,
                         1964.54, 1976.54, 1986.45, 1996.80, 2008.62, 2019.96])
SOLAR_CYCLE_YR = 11.00606
SKY_WAVELENGTH_AA = np.array([3650.0, 4450.0, 5510.0, 6580.0, 8060.0, 10000.0, 12500.0, 16500.0, 22000.0])
SKY_MAG_VEGA = np.array([22.0, 22.7, 21.9, 21.0, 20.0, 18.8, 16.1, 14.7, 12.5])
SKY_EXT_MAG = np.array([0.550, 0.250, 0.150, 0.090, 0.060, 0.050, 0.100, 0.110, 0.070])
DAYLIGHT_SUN_ZD = np.array([0.0, 30.0, 60.0])
DAYLIGHT_SKY_ZD = np.array([0.0, 30.0, 60.0, 80.0])
DAYLIGHT_V_MAG = np.array([
    [8.73, 8.74, 8.85, 9.17],
    [8.67, 8.69, 8.84, 9.19],
    [8.50, 8.56, 8.80, 9.20],
])


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
                        sun_altitude_deg):
    """Estimated observed sky surface brightness in Vega mag/arcsec².

    Below -18° Sun altitude this is the supplied dark-sky table corrected for
    the solar-cycle airglow factor and approximate sky airmass.  Between -18°
    and the horizon, dark and daylight fluxes are blended smoothly.  Above the
    horizon, Weaver's V-band daylight table is used; its supplied broad-band
    dark-sky colour relation is interpolated to the selected filter pivot.
    """
    wavelength = float(pivot_wavelength_aa)
    base_mag = float(np.interp(wavelength, SKY_WAVELENGTH_AA, SKY_MAG_VEGA))
    v_base = float(np.interp(5510.0, SKY_WAVELENGTH_AA, SKY_MAG_VEGA))
    colour = base_mag - v_base
    airmass = float(target_airmass) if np.isfinite(target_airmass) else 1.0
    airmass = max(airmass, 1.0)
    activity = solar_cycle_activity(utc_datetime)
    airglow_factor = (145.0 + 130.0 * (activity - 0.8) / 1.2) / 145.0
    night_mag = base_mag - 2.5 * np.log10(airglow_factor * airmass)
    daylight_mag = _daylight_v_mag(max(float(sun_altitude_deg), 0.0), target_altitude_deg) + colour
    if sun_altitude_deg >= 0.0:
        return daylight_mag
    if sun_altitude_deg <= -18.0:
        return night_mag
    blend = (float(sun_altitude_deg) + 18.0) / 18.0
    flux = (1.0 - blend) * 10.0 ** (-0.4 * night_mag) + blend * 10.0 ** (-0.4 * daylight_mag)
    return float(-2.5 * np.log10(flux))
