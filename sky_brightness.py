"""
sky_brightness.py
Sky brightness calculation in 9 photometric bands.
Uses Benn & Ellison (1998) for dark sky and Krisciunas & Schaefer (1991) for moonlight.

References:
- Benn, C. R. & Ellison, S. L. (1998). La Palma Observatory Technical Note 115
- Krisciunas, K. & Schaefer, B. E. (1991). PASP 103, 1033
"""

import numpy as np


# ============================================================================
# Data: Solar cycle, sky magnitudes, extinction coefficients, moon fudge factors
# ============================================================================

SOLAR_MINIMA = np.array([
    1901.04, 1913.62, 1924.05, 1933.63, 1944.28, 1954.04,
    1964.54, 1976.54, 1986.45, 1996.80, 2008.62, 2019.96
])
SOLAR_CYCLE = 11.00606  # years

# Dark sky reference magnitudes per arcsec² (9 bands: U, B, V, R, I, Z, J, H, K)
SKYMAG = np.array([22.0, 22.7, 21.9, 21.0, 20.0, 18.8, 16.1, 14.7, 12.5])

# Extinction coefficients per band (Bouguer law: A = k * (airmass - 1))
SKYEXT = np.array([0.550, 0.250, 0.150, 0.090, 0.060, 0.050, 0.100, 0.110, 0.070])

# Lunar brightness fudge factors (empirical normalization, Krisciunas & Schaefer)
MOON_FUDGE = np.array([2.0, 1.6, 2.4, 3.0, 5.9, 8.94, 13.06, 18.09, 24.04])

# Band names and reference wavelengths
BANDS = ['U', 'B', 'V', 'R', 'I', 'Z', 'J', 'H', 'K']
BAND_WAVELENGTH_NM = [360, 440, 550, 700, 900, 1000, 1250, 1650, 2200]

# Conventional broad-band Vega F_nu zero points (Jy), paired with the U--K
# grid above.  They convert the supplied Vega sky magnitudes to physical
# F_lambda before the spectroscopic ETC interpolates the continuum.  A
# calibrated site sky spectrum can replace this broad-band baseline later.
BAND_VEGA_ZEROPOINT_JY = np.array([1810.0, 4260.0, 3640.0, 3080.0, 2550.0,
                                   2250.0, 1594.0, 1024.0, 666.7])

# Constants in S10 units (nanoLambert scale)
MAG_ZERO_S10 = 27.78  # reference magnitude for S10 scale

# Airglow emission-layer height for the van Rhijn path-length law [m] and
# mean Earth radius [m].
AIRGLOW_LAYER_HEIGHT_M = 90.0e3
EARTH_RADIUS_M = 6.371e6

# Dark reference S10 composition used to normalise the tabulated U--K dark-sky
# magnitudes: solar-minimum airglow at the zenith plus the high-ecliptic-
# latitude zodiacal floor.  Position and zenith-distance dependent components
# are expressed relative to this reference.
Q_DARK_REFERENCE_S10 = 145.0 + 60.0


def van_rhijn_factor(zenith_dist_deg):
    """Airglow path-length enhancement for an emitting layer at ~90 km.

    V(z) = [1 - (R/(R+h) sin z)^2]^(-1/2)  (van Rhijn 1921).  Sub-linear in
    airmass and bounded at the horizon, unlike a Bouguer sec z scaling.
    """
    z = np.radians(np.clip(np.asarray(zenith_dist_deg, dtype=np.float64), 0.0, 90.0))
    ratio = EARTH_RADIUS_M / (EARTH_RADIUS_M + AIRGLOW_LAYER_HEIGHT_M)
    return 1.0 / np.sqrt(1.0 - (ratio * np.sin(z)) ** 2)


def zodiacal_s10(ecliptic_lat_deg):
    """Benn & Ellison zodiacal-light brightness in S10, symmetric in beta."""
    abs_beta = np.abs(float(ecliptic_lat_deg))
    if abs_beta < 60.0:
        return 140.0 - 90.0 * np.sin(np.radians(abs_beta))
    return 60.0


def starlight_s10(galactic_lat_deg):
    """Benn & Ellison integrated-starlight brightness in S10."""
    return 100.0 * np.exp(-np.abs(float(galactic_lat_deg)) / 10.0)


def dark_sky_position_correction_mag(zenith_dist_deg, ecliptic_lat_deg, galactic_lat_deg,
                                     solar_activity=0.8):
    """Magnitude correction of the dark sky relative to the tabulated values.

    Combines van Rhijn airglow (scaled by the solar-cycle activity), zodiacal
    light at the target's ecliptic latitude and integrated starlight at its
    galactic latitude, normalised to the dark reference composition of the
    U--K table.  Negative values brighten the sky.
    """
    qair = (145.0 + 130.0 * (float(solar_activity) - 0.8) / 1.2) * float(van_rhijn_factor(zenith_dist_deg))
    q3 = qair + zodiacal_s10(ecliptic_lat_deg) + starlight_s10(galactic_lat_deg)
    return -2.5 * np.log10(q3 / Q_DARK_REFERENCE_S10)


def moonlight_scattering_function(separation_deg):
    """Krisciunas & Schaefer (1991) f(rho): Rayleigh + Mie aureole terms.

    f(rho) = 10^5.36 (1.06 + cos^2 rho) + 10^(6.15 - rho/40)
    """
    rho = np.asarray(separation_deg, dtype=np.float64)
    return 10.0 ** 5.36 * (1.06 + np.cos(np.radians(rho)) ** 2) + 10.0 ** (6.15 - rho / 40.0)


def solar_cycle_phase(year):
    """
    Solar cycle phase (11-year cycle).
    
    Parameters
    ----------
    year : float
        Decimal year (e.g., 2026.5)
    
    Returns
    -------
    fase_sol : float
        Phase scaled to [0.8, 2.0] (0.8 = solar min, 2.0 = solar max)
    """
    # Find which 11-year interval we're in
    if year < SOLAR_MINIMA[0]:
        # Before first minimum: use periodicity
        fase_raw = ((year - SOLAR_MINIMA[0]) % SOLAR_CYCLE) / SOLAR_CYCLE
    elif year > SOLAR_MINIMA[-1]:
        # After last minimum: use periodicity
        fase_raw = ((year - SOLAR_MINIMA[-1]) % SOLAR_CYCLE) / SOLAR_CYCLE
    else:
        # Between known minima
        for i in range(len(SOLAR_MINIMA) - 1):
            if SOLAR_MINIMA[i] <= year <= SOLAR_MINIMA[i+1]:
                fase_raw = (year - SOLAR_MINIMA[i]) / (SOLAR_MINIMA[i+1] - SOLAR_MINIMA[i])
                break
    
    # Scale to [0.8, 2.0]
    fase_sol = 0.8 + 1.2 * fase_raw
    return fase_sol


def sky_brightness_total(
    year, month, day, hour, minute=0,
    ecliptic_lat_deg=90.0, galactic_lat_deg=90.0,
    airmass_target=1.0, airmass_moon=1.0,
    lunar_phase_deg=0, moon_separation_deg=180, moon_zenith_dist_deg=90,
    v_extinction_mag=0.15,
    include_moon=True,
    include_sun_twilight=False,
    sun_altitude_deg=-90,
    target_zenith_dist_deg=None,
):
    """
    Calculate 9-band sky brightness [mag/arcsec^2].

    Dark sky: Benn & Ellison airglow (van Rhijn path-length law), zodiacal
    light at the field's |ecliptic latitude| and integrated starlight at its
    |galactic latitude|.  Moonlight: Krisciunas & Schaefer (1991) with the
    published scattering function f(rho).

    ``target_zenith_dist_deg`` drives the van Rhijn airglow factor; when it is
    None it is recovered from ``airmass_target`` through sec z.
    """
    nband = 9

    # 1. Dark-sky components (Benn & Ellison), airglow on the van Rhijn law.
    fase_sol = solar_cycle_phase(year + (month - 1) / 12 + (day - 1) / 365.25)
    if target_zenith_dist_deg is None:
        secz = max(float(airmass_target), 1.0)
        target_zenith_dist_deg = np.degrees(np.arccos(np.clip(1.0 / secz, 0.0, 1.0)))
    qair = (145.0 + 130.0 * (fase_sol - 0.8) / 1.2) * float(van_rhijn_factor(target_zenith_dist_deg))
    qzod = zodiacal_s10(ecliptic_lat_deg)
    qstar = starlight_s10(galactic_lat_deg)
    q3 = qair + qzod + qstar

    # 2. Dark-sky brightness in all bands: V-band S10 plus the table colours.
    sky_v = MAG_ZERO_S10 - 2.5 * np.log10(q3)
    sky_mag = sky_v + (SKYMAG - SKYMAG[2])
    qsky = 10.0 ** ((MAG_ZERO_S10 - sky_mag) / 2.5)

    # 3. Moonlight contribution (Krisciunas & Schaefer 1991).
    qmoon = np.zeros(nband, dtype=np.float64)
    if include_moon and moon_separation_deg < 180:
        alpha = float(lunar_phase_deg)  # K&S phase angle alpha: 0 = full moon
        istar = 10.0 ** (-0.4 * (3.84 + 0.026 * abs(alpha) + 4.0e-9 * alpha ** 4))
        fr = float(moonlight_scattering_function(moon_separation_deg))
        delta_ext = (10.0 ** (-0.4 * v_extinction_mag * airmass_moon)
                     * (1.0 - 10.0 ** (-0.4 * v_extinction_mag * airmass_target)))
        bnl = istar * fr * delta_ext          # nanoLambert
        bs10 = bnl * 3.8                      # S10 units
        qmoon = MOON_FUDGE * bs10

    # 4. Sun twilight contribution (simplified linear flux bridge).
    qsun = np.zeros(nband, dtype=np.float64)
    if include_sun_twilight and sun_altitude_deg > -18 and sun_altitude_deg <= 0:
        twilight_factor = (sun_altitude_deg + 18.0) / 18.0
        twilight_mag_correction = twilight_factor * 2.0
        qsun = 10.0 ** ((MAG_ZERO_S10 - (sky_mag - twilight_mag_correction)) / 2.5)

    # 5. Total.
    qall = np.maximum(qsky + qmoon + qsun, 1e-30)
    return MAG_ZERO_S10 - 2.5 * np.log10(qall)
