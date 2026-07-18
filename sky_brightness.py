"""
sky_brightness.py
Sky brightness calculation in 9 photometric bands.
Uses Benn & Ellison (1998) for dark sky and Krisciunas & Schaefer (1991) for moonlight.

References:
- Benn, C. R. & Ellison, S. L. (1998). La Palma Observatory Technical Note 115
- Krisciunas, K. & Schaefer, B. E. (1991). PASP 103, 1033
"""

import numpy as np
from ephemeris import julian_date, julian_centuries, sun_position, moon_position, airmass_simple


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
    ecliptic_lat_deg=0, galactic_lat_deg=0,
    airmass_target=1.0, airmass_moon=1.0,
    lunar_phase_deg=0, moon_separation_deg=180, moon_zenith_dist_deg=90,
    v_extinction_mag=0.15,
    include_moon=True,
    include_sun_twilight=False,
    sun_altitude_deg=-90
):
    """
    Calculate 9-band sky brightness.
    
    Uses Benn & Ellison model for airglow, zodiacal light, starlight.
    Adds Krisciunas & Schaefer moonlight contribution.
    
    Parameters
    ----------
    year, month, day, hour, minute : int
        Observation time (UTC)
    ecliptic_lat_deg : float
        Ecliptic latitude of observation field [deg]
    galactic_lat_deg : float
        Galactic latitude of observation field [deg]
    airmass_target : float
        Airmass for the target (use 1.0 for zenith)
    airmass_moon : float
        Airmass for the moon
    lunar_phase_deg : float
        Lunar phase angle [deg] (0=full, 90=quarter, 180=new)
    moon_separation_deg : float
        Angular separation from moon to field [deg]
    moon_zenith_dist_deg : float
        Zenith distance of moon [deg]
    v_extinction_mag : float
        V-band extinction coefficient [mag/airmass]
    include_moon : bool
    include_sun_twilight : bool
    sun_altitude_deg : float
        Sun altitude [deg] (used if include_sun_twilight)
    
    Returns
    -------
    sky_mag : ndarray
        Sky brightness in 9 bands [mag/arcsec²]
    """
    nband = 9
    
    # 1. Dark sky components (Benn & Ellison)
    fase_sol = solar_cycle_phase(year + (month - 1)/12 + (day - 1)/365.25)
    
    qair = (145 + 130 * (fase_sol - 0.8) / 1.2) * airmass_target
    
    # Zodiacal light (ecliptic latitude dependent)
    abs_eclat = np.abs(ecliptic_lat_deg)
    if abs_eclat < 60:
        qzod = 140 - 90 * np.sin(np.radians(ecliptic_lat_deg))
    else:
        qzod = 60
    
    # Starlight (galactic latitude dependent)
    qstar = 100 * np.exp(-np.abs(galactic_lat_deg) / 10)
    
    # Total dark sky
    q3 = qair + qzod + qstar
    
    # 2. Dark sky brightness in all bands
    qsky = np.zeros(nband)
    sky_mag = np.zeros(nband)
    
    for i in range(nband):
        # V-band reference
        sky_v = MAG_ZERO_S10 - 2.5 * np.log10(q3)
        
        # Color correction for this band
        sky_mag[i] = sky_v + (SKYMAG[i] - SKYMAG[2])
        
        # Convert back to S10 units
        qsky[i] = 10**((MAG_ZERO_S10 - sky_mag[i]) / 2.5)
    
    # 3. Moonlight contribution (Krisciunas & Schaefer 1991)
    qmoon = np.zeros(nband)
    
    if include_moon and moon_separation_deg < 180:
        # Lunar phase factor (0=full, 90=quarter, 180=new)
        # i is the phase angle (0 = full, 180 = new)
        i_rad = np.radians(180 - lunar_phase_deg)  # convert to geocentric phase angle
        
        # Illumination fraction
        k = (1 + np.cos(i_rad)) / 2
        
        # Brightness factor from Krisciunas & Schaefer
        s = 10**(-0.4 * (3.84 + 0.026 * abs(lunar_phase_deg) + 4e-9 * lunar_phase_deg**4))
        
        # Geometric factor (phase function of scattering from moon)
        rho = moon_separation_deg
        fr = (10**(5.36 * (1.06 + np.cos(np.radians(rho))**2)) +
              10**(6.15 - rho / 40))
        
        # Atmospheric extinction
        xz = airmass_target
        xzm = airmass_moon
        
        # Extinction correction
        delta_ext = 10**(-0.4 * v_extinction_mag * xzm) * (1 - 10**(-0.4 * v_extinction_mag * xz))
        
        # Moon brightness in nanoLambert (Krisciunas & Schaefer)
        bnl = s * fr * delta_ext
        bs10 = bnl * 3.8  # convert to S10 units
        
        # Moon contribution per band (fudge factor accounts for lunar color)
        for i in range(nband):
            qmoon[i] = MOON_FUDGE[i] * bs10
    
    # 4. Sun twilight contribution (simplified)
    qsun = np.zeros(nband)
    if include_sun_twilight and sun_altitude_deg > -18:
        # Very approximate: twilight sky brightness increases rapidly below -18°
        # For now, we'll add a simple model
        if sun_altitude_deg > 0:
            # Daytime: not implemented here
            pass
        else:
            # Twilight (-18° to 0°): interpolate
            twilight_factor = (sun_altitude_deg + 18) / 18  # [0, 1]
            # Very rough: add ~1-2 mag to sky
            twilight_mag_correction = twilight_factor * 2.0
            for i in range(nband):
                qsun[i] = 10**((MAG_ZERO_S10 - (sky_mag[i] - twilight_mag_correction)) / 2.5)
    
    # 5. Total sky brightness
    qall = qsky + qmoon + qsun
    
    # Avoid log of zero
    qall = np.maximum(qall, 1e-30)
    
    sky_mag_total = MAG_ZERO_S10 - 2.5 * np.log10(qall)
    
    return sky_mag_total


def get_sky_components(
    year, month, day, hour, minute=0,
    ecliptic_lat_deg=0, galactic_lat_deg=0,
    airmass_target=1.0, airmass_moon=1.0,
    lunar_phase_deg=0, moon_separation_deg=180, moon_zenith_dist_deg=90,
    v_extinction_mag=0.15
):
    """
    Return sky brightness components separately (for debugging/display).
    
    Returns
    -------
    dict with keys: airglow, zodiacal, starlight, moon, total (all as arrays of 9 bands)
    """
    nband = 9
    fase_sol = solar_cycle_phase(year + (month - 1)/12 + (day - 1)/365.25)
    
    # Components in S10
    qair = (145 + 130 * (fase_sol - 0.8) / 1.2) * airmass_target
    abs_eclat = np.abs(ecliptic_lat_deg)
    qzod = 140 - 90 * np.sin(np.radians(ecliptic_lat_deg)) if abs_eclat < 60 else 60
    qstar = 100 * np.exp(-np.abs(galactic_lat_deg) / 10)
    
    # Convert to mag/arcsec²
    mag_air = MAG_ZERO_S10 - 2.5 * np.log10(qair + 1e-30)
    mag_zod = MAG_ZERO_S10 - 2.5 * np.log10(qzod + 1e-30)
    mag_star = MAG_ZERO_S10 - 2.5 * np.log10(qstar + 1e-30)
    
    # Moon
    mag_moon = np.zeros(nband)
    if moon_separation_deg < 180:
        i_rad = np.radians(180 - lunar_phase_deg)
        k = (1 + np.cos(i_rad)) / 2
        s = 10**(-0.4 * (3.84 + 0.026 * abs(lunar_phase_deg) + 4e-9 * lunar_phase_deg**4))
        rho = moon_separation_deg
        fr = (10**(5.36 * (1.06 + np.cos(np.radians(rho))**2)) +
              10**(6.15 - rho / 40))
        xz = airmass_target
        xzm = airmass_moon
        delta_ext = 10**(-0.4 * v_extinction_mag * xzm) * (1 - 10**(-0.4 * v_extinction_mag * xz))
        bnl = s * fr * delta_ext
        bs10 = bnl * 3.8
        
        for i in range(nband):
            qmoon_i = MOON_FUDGE[i] * bs10
            mag_moon[i] = MAG_ZERO_S10 - 2.5 * np.log10(qmoon_i + 1e-30)
    else:
        mag_moon[:] = 30.0  # very faint if moon far away
    
    # Total
    sky_total = sky_brightness_total(
        year, month, day, hour, minute,
        ecliptic_lat_deg, galactic_lat_deg,
        airmass_target, airmass_moon,
        lunar_phase_deg, moon_separation_deg, moon_zenith_dist_deg,
        v_extinction_mag
    )
    
    return {
        'airglow': np.full(nband, mag_air),
        'zodiacal': np.full(nband, mag_zod),
        'starlight': np.full(nband, mag_star),
        'lunar': mag_moon,
        'total': sky_total,
        'bands': BANDS
    }
