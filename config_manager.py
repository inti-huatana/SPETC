"""
config_manager.py
Observatory presets, loaded from observatories.json (name, lon, lat, elev, mag_sky).
"""

import json
from pathlib import Path
from collections import namedtuple


ObservatoryPreset = namedtuple("ObservatoryPreset", ["name", "lon", "lat", "elev", "mag_sky"])

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
        presets[entry["name"]] = ObservatoryPreset(
            name=entry["name"],
            lon=float(entry["lon"]),
            lat=float(entry["lat"]),
            elev=float(entry["elev"]),
            mag_sky=float(entry["mag_sky"]),
        )
    return presets


OBSERVATORY_PRESETS = load_observatories()
