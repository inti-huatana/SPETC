"""
config_manager.py
Observatory presets, loaded from observatories.json (name, lon, lat, elev, mag_sky).
"""

import json
from pathlib import Path
from collections import namedtuple


ObservatoryPreset = namedtuple(
    "ObservatoryPreset",
    ["name", "lon", "lat", "elev", "mag_sky", "timezone", "utc_offset_h"])

_OBSERVATORIES_JSON = Path(__file__).parent / "observatories.json"


def load_observatories(path=None):
    """
    Load observatory presets from a JSON file: a list of objects with
    keys name, lon, lat, elev, mag_sky.

    Returns
    -------
    dict {name: ObservatoryPreset}
    """
    path = Path(path) if path else _OBSERVATORIES_JSON
    if not path.exists():
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    presets = {}
    for entry in data:
        lon = float(entry["lon"])
        presets[entry["name"]] = ObservatoryPreset(
            name=entry["name"],
            lon=lon,
            lat=float(entry["lat"]),
            elev=float(entry["elev"]),
            mag_sky=float(entry["mag_sky"]),
            # timezone and utc_offset_h let the GUI set the observation
            # date/time reference automatically when a site is selected.
            timezone=str(entry.get("timezone", "")),
            utc_offset_h=float(entry.get("utc_offset_h", round(lon / 15.0))),
        )
    return presets


def save_observatories(presets, path=None):
    """Persist a {name: ObservatoryPreset} mapping back to the JSON list."""
    path = Path(path) if path else _OBSERVATORIES_JSON
    ordered = sorted(presets.values(), key=lambda p: p.name.casefold())
    data = [{"name": p.name, "lon": p.lon, "lat": p.lat, "elev": p.elev,
             "mag_sky": p.mag_sky, "timezone": p.timezone,
             "utc_offset_h": p.utc_offset_h} for p in ordered]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


OBSERVATORY_PRESETS = load_observatories()
