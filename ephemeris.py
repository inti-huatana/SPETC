"""Astropy-based coordinate, visibility and time utilities.

Azimuth follows the standard astronomical convention: North=0, East=90,
South=180, West=270 degrees.  Calculations use ICRS positions and UTC.
"""

import numpy as np
from datetime import timezone
from zoneinfo import ZoneInfo
import astropy.units as u
from astropy.coordinates import (SkyCoord, EarthLocation, AltAz, Angle, get_sun,
                                 get_body, GeocentricTrueEcliptic)
from astropy.time import Time
from astropy.utils import iers

# An ETC must also work offline.  Astropy will use its bundled IERS table.
iers.conf.auto_download = False
iers.conf.auto_max_age = None


def _location(lat_deg, lon_deg, elev_m=0.0):
    return EarthLocation.from_geodetic(float(lon_deg) * u.deg, float(lat_deg) * u.deg,
                                       float(elev_m) * u.m)


def site_pressure_hpa(elev_m):
    """ISA barometric pressure at the site elevation [hPa]."""
    return 1013.25 * np.exp(-float(elev_m) / 8435.0)


def site_temperature_c(elev_m):
    """ISA temperature at the site elevation [Celsius]."""
    return 15.0 - 0.0065 * float(elev_m)


def _refraction_frame(time, lat_deg, lon_deg, elev_m):
    """AltAz frame including ERFA atmospheric refraction at site conditions."""
    return AltAz(obstime=time, location=_location(lat_deg, lon_deg, elev_m),
                 pressure=site_pressure_hpa(elev_m) * u.hPa,
                 temperature=site_temperature_c(elev_m) * u.deg_C,
                 relative_humidity=0.2, obswl=0.55 * u.um)


def pickering_airmass(apparent_altitude_deg):
    """Pickering (2002) interpolative airmass from the *apparent* altitude.

    X = 1 / sin(h + 244 / (165 + 47 h^1.1)), accurate to the horizon, unlike
    plane-parallel sec z which diverges and overestimates below ~20 deg.
    """
    h = np.clip(np.asarray(apparent_altitude_deg, dtype=np.float64), 1e-3, 90.0)
    x = 1.0 / np.sin(np.radians(h + 244.0 / (165.0 + 47.0 * h ** 1.1)))
    return float(x) if x.ndim == 0 else x


def parallactic_angle_deg(hour_angle_deg, dec_deg, lat_deg):
    """Parallactic angle q [deg], measured from North towards East.

    q = atan2(sin H, tan(phi) cos(delta) - sin(delta) cos H).  Zero when the
    target crosses the meridian; the sign follows the hour angle.
    """
    ha = np.radians(np.asarray(hour_angle_deg, dtype=np.float64))
    dec = np.radians(float(dec_deg))
    lat = np.radians(float(lat_deg))
    q = np.arctan2(np.sin(ha), np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(ha))
    return np.degrees(q)


def julian_date(year, month, day, hour=0, minute=0, second=0):
    return Time(f"{int(year):04d}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{float(second):06.3f}",
                scale="utc").jd


def julian_centuries(jd):
    return (np.asarray(jd, dtype=float) - 2451545.0) / 36525.0


def sun_position(jd):
    sun = get_sun(Time(jd, format="jd", scale="utc")).icrs
    return np.asarray(sun.ra.deg), np.asarray(sun.dec.deg), np.asarray(sun.distance.to_value(u.au))


def moon_position(jd):
    time = Time(jd, format="jd", scale="utc")
    moon = get_body("moon", time).icrs
    sun = get_sun(time).icrs
    phase = sun.separation(moon).to_value(u.deg)
    return np.asarray(moon.ra.deg), np.asarray(moon.dec.deg), np.asarray(moon.distance.to_value(u.km)), np.asarray(phase)


def jd_to_hm_string(jd):
    return Time(jd, format="jd", scale="utc").datetime.strftime("%H:%M UT")


def parse_ra(value):
    return Angle(value, unit=u.hourangle).to_value(u.deg)


def parse_dec(value):
    return Angle(value, unit=u.deg).to_value(u.deg)


def degrees_to_sexagesimal_ra(value):
    return Angle(value * u.deg).to_string(unit=u.hourangle, sep=":", precision=2, pad=True)


def degrees_to_sexagesimal_dec(value):
    return Angle(value * u.deg).to_string(unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True)


def galactic_to_equatorial(l_deg, b_deg):
    c = SkyCoord(l=float(l_deg) * u.deg, b=float(b_deg) * u.deg, frame="galactic").icrs
    return c.ra.deg, c.dec.deg


def equatorial_to_galactic(ra_deg, dec_deg):
    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs").galactic
    return c.l.deg, c.b.deg


def ecliptic_to_equatorial(lon_deg, lat_deg):
    c = SkyCoord(lon=float(lon_deg) * u.deg, lat=float(lat_deg) * u.deg,
                 frame=GeocentricTrueEcliptic(equinox=Time("J2000"))).icrs
    return c.ra.deg, c.dec.deg


def equatorial_to_ecliptic(ra_deg, dec_deg):
    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    e = c.transform_to(GeocentricTrueEcliptic(equinox=Time("J2000")))
    return e.lon.deg, e.lat.deg


def altitude_azimuth(ra_deg, dec_deg, lat_deg, lon_deg, jd, elev_m=0.0):
    t = Time(jd, format="jd", scale="utc")
    altaz = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs").transform_to(
        AltAz(obstime=t, location=_location(lat_deg, lon_deg, elev_m), pressure=0 * u.hPa))
    return float(altaz.alt.deg), float(altaz.az.deg)


def airmass_simple(altitude_deg):
    """Geometric sec(z), returned only above the 5 degree ETC cutoff."""
    alt = np.asarray(altitude_deg, dtype=float)
    result = np.full(alt.shape, np.nan, dtype=float)
    good = alt >= 5.0
    result[good] = 1.0 / np.sin(np.deg2rad(alt[good]))
    return float(result) if result.ndim == 0 else result


def compute_target_track(ra_deg, dec_deg, lat_deg, lon_deg, jd_start, jd_end,
                         step_min=5.0, elev_m=0.0, local_utc_offset_h=0.0, timezone_name=None):
    """Compute a physically consistent track; airmass is masked below 5 deg."""
    n = max(int(np.ceil((jd_end - jd_start) * 1440.0 / step_min)) + 1, 2)
    jd = np.linspace(jd_start, jd_end, n)
    time = Time(jd, format="jd", scale="utc")
    frame = _refraction_frame(time, lat_deg, lon_deg, elev_m)
    target = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
    target_altaz = target.transform_to(frame)
    sun = get_sun(time)
    sun_altaz = sun.transform_to(frame)
    # Twilight thresholds (-6/-12/-18 deg) are conventionally defined on the
    # geometric, unrefracted Sun altitude; near the horizon the refracted
    # altitude differs by up to ~0.5 deg and would shift the twilight class.
    geometric_frame = AltAz(obstime=time, location=_location(lat_deg, lon_deg, elev_m),
                            pressure=0 * u.hPa)
    sun_geometric = sun.transform_to(geometric_frame)
    moon = get_body("moon", time, location=_location(lat_deg, lon_deg, elev_m))
    moon_altaz = moon.transform_to(frame)
    alt = target_altaz.alt.to_value(u.deg)
    airmass = np.full(n, np.nan)
    above_cutoff = alt >= 5.0
    # Pickering (2002) on the refraction-corrected apparent altitude replaces
    # plane-parallel sec z; the two agree near the zenith and Pickering stays
    # physical at low altitude.
    airmass[above_cutoff] = pickering_airmass(alt[above_cutoff])
    moon_sep = target_altaz.separation(moon_altaz).to_value(u.deg)
    phase = sun_altaz.separation(moon_altaz).to_value(u.deg)
    # Solar elongation of the target (for the zodiacal-light gradient) and
    # the topocentric Moon distance (for the moonlight illuminance).
    sun_sep = target_altaz.separation(sun_altaz).to_value(u.deg)
    moon_distance_km = moon.distance.to_value(u.km)
    lst = time.sidereal_time("apparent", longitude=float(lon_deg) * u.deg)
    hour_angle = (lst - float(ra_deg) * u.hourangle / 15.0).wrap_at(180 * u.deg).to_value(u.deg)
    parallactic = parallactic_angle_deg(hour_angle, dec_deg, lat_deg)
    if timezone_name:
        try:
            zone = ZoneInfo(str(timezone_name))
        except Exception as exc:
            raise ValueError(f"Invalid IANA timezone {timezone_name!r}.") from exc
        local_datetimes = np.asarray([value.replace(tzinfo=timezone.utc).astimezone(zone).replace(tzinfo=None)
                                      for value in time.datetime])
    else:
        local_datetimes = Time(time.jd + float(local_utc_offset_h) / 24.0, format="jd", scale="utc").datetime
    return {"jd": jd, "utc_datetime": time.datetime, "local_datetime": local_datetimes,
            "alt_target": alt, "az_target": target_altaz.az.to_value(u.deg), "airmass_target": airmass,
            "parallactic_deg": parallactic,
            "alt_sun": sun_altaz.alt.to_value(u.deg), "az_sun": sun_altaz.az.to_value(u.deg),
            "alt_sun_geometric": sun_geometric.alt.to_value(u.deg),
            "alt_moon": moon_altaz.alt.to_value(u.deg), "az_moon": moon_altaz.az.to_value(u.deg),
            "phase_moon": phase, "moon_sep_deg": moon_sep,
            "sun_sep_deg": sun_sep, "moon_distance_km": moon_distance_km}


# Standard almanac convention for sunrise/sunset: the *upper limb* touches
# the geometric horizon with standard refraction, i.e. the Sun's centre at
# geometric altitude -50' = -(34' refraction + 16' semidiameter).
SUN_RISE_SET_ALTITUDE_DEG = -50.0 / 60.0


# Compatibility wrappers retained for callers outside the GUI.
def find_altitude_at(ra_deg, dec_deg, lat_deg, lon_deg, jd):
    return altitude_azimuth(ra_deg, dec_deg, lat_deg, lon_deg, jd)[0]


def find_transit(ra_deg, dec_deg, lat_deg, lon_deg, jd_center, window_hours=15.0, step_min=2.0):
    track = compute_target_track(ra_deg, dec_deg, lat_deg, lon_deg,
                                 jd_center - window_hours / 24, jd_center + window_hours / 24, step_min)
    i = int(np.argmax(track["alt_target"]))
    return track["jd"][i], track["alt_target"][i]


def find_altitude_crossing(ra_deg, dec_deg, lat_deg, lon_deg, jd_transit, threshold_deg, direction,
                           search_hours=15.0, step_min=2.0):
    sign = -1 if direction == "before" else 1
    track = compute_target_track(ra_deg, dec_deg, lat_deg, lon_deg,
                                 jd_transit, jd_transit + sign * search_hours / 24, step_min)
    a = track["alt_target"] - threshold_deg
    changes = np.where(a[:-1] * a[1:] <= 0)[0]
    if changes.size == 0:
        return None
    i = changes[0]
    return np.interp(0.0, [a[i], a[i + 1]], [track["jd"][i], track["jd"][i + 1]])


def find_rise_set(ra_deg, dec_deg, lat_deg, lon_deg, jd_obs, threshold_deg=0.0):
    track = compute_target_track(ra_deg, dec_deg, lat_deg, lon_deg, jd_obs - 0.75, jd_obs + 0.75, 2.0)
    a = track["alt_target"] - threshold_deg
    crossings = np.where(a[:-1] * a[1:] <= 0)[0]
    if len(crossings) < 2:
        return None, None, None
    times = [np.interp(0.0, [a[i], a[i + 1]], [track["jd"][i], track["jd"][i + 1]]) for i in crossings]
    transit, _ = find_transit(ra_deg, dec_deg, lat_deg, lon_deg, jd_obs)
    before = [t for t in times if t < transit]
    after = [t for t in times if t > transit]
    return (before[-1] if before else None, after[0] if after else None, transit)
