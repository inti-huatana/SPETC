"""SVO VOTable filter profiles stored locally below ``data``.

Each logical filter is represented by two XML files listed in
``filters.list``: ``<filter>_Vega`` and ``<filter>_AB``.  The supported
layouts are ``data/filters.list`` with profiles in ``data/filters/`` and the
older ``data/filters/filters.list`` layout.  The files contain
the transmission curve and the appropriate photometric zero point in Jy.
No part of the former monolithic ``resp_filters.dat`` catalogue is used.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from spectral_utils import make_curve, wavelength_unit
import astropy.units as u


AB_ZERO_POINT_JY = 3631.0


@dataclass(frozen=True)
class FilterProfile:
    """One filter transmission curve and one magnitude-system calibration."""

    label: str
    magnitude_system: str
    transmission: np.ndarray
    zero_point_jy: float
    detector_type: int
    metadata: dict
    source_path: Path

    @property
    def pivot_wavelength_aa(self):
        return float(self.metadata.get("WavelengthPivot", self.metadata.get("WavelengthRef", np.nan)))

    @property
    def effective_width_aa(self):
        return float(self.metadata.get("WidthEff", np.nan))

    @property
    def fwhm_aa(self):
        return float(self.metadata.get("FWHM", np.nan))


def filters_directory(data_directory):
    """Return the data root; both supported list layouts are resolved below."""
    return Path(data_directory)


def _list_path(root):
    root = Path(root)
    for candidate in (root / "filters.list", root / "filters" / "filters.list"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Filter list not found: expected {root / 'filters.list'} or {root / 'filters' / 'filters.list'}")


def _logical_name(entry):
    entry = Path(entry).name
    if entry.lower().endswith(".xml"):
        entry = entry[:-4]
    for suffix in ("_Vega", "_AB"):
        if entry.endswith(suffix):
            return entry[:-len(suffix)]
    return entry if entry.upper() == "BLANK" else None


def _resolve_profile_path(root, entry):
    """Resolve list entries for both historical and current data layouts."""
    root = Path(root)
    base = _list_path(root).parent
    entry_path = Path(entry)
    candidates = []
    for parent in (base, base / "filters", root, root / "filters"):
        candidates.append(parent / entry_path)
        if entry_path.suffix.lower() != ".xml":
            candidates.append(parent / f"{entry}.xml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates[:4])
    raise FileNotFoundError(f"Filter profile not found for {entry!r}; checked {searched} …")


def list_filter_labels(directory):
    """Return logical filter names from either supported local list layout."""
    directory = Path(directory)
    list_path = _list_path(directory)
    names = set()
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        name = _logical_name(entry)
        if name:
            names.add(name)
    # BLANK.xml is intentionally optional from filters.list: placing it in
    # data/filters makes the special unfiltered response available directly.
    if any(path.is_file() for path in (
        directory / "BLANK.xml", directory / "filters" / "BLANK.xml",
        list_path.parent / "BLANK.xml", list_path.parent / "filters" / "BLANK.xml",
    )):
        names.add("BLANK")
    if not names:
        raise ValueError(f"No *_Vega or *_AB profiles found in {list_path}")
    # The filter-free response is the natural spectroscopy default.  Keep it
    # first while retaining deterministic alphabetical order for real filters.
    return (["BLANK"] if "BLANK" in names else []) + sorted(
        (name for name in names if name != "BLANK"), key=str.casefold)


def _parse_votable(path, label, magnitude_system):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid filter VOTable {path}: {exc}") from exc
    metadata = {}
    for param in root.iter():
        if param.tag.rsplit("}", 1)[-1] != "PARAM":
            continue
        name, value = param.attrib.get("name"), param.attrib.get("value")
        if name and value is not None:
            metadata[name] = value
    try:
        xml_system = metadata["MagSys"].strip().upper()
        zero_point = float(metadata["ZeroPoint"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path} does not define MagSys and ZeroPoint") from exc
    if xml_system != magnitude_system.upper():
        raise ValueError(f"{path} declares MagSys={xml_system}, not {magnitude_system}")
    unit = metadata.get("ZeroPointUnit", "Jy").strip().lower()
    if unit != "jy":
        raise ValueError(f"{path} has unsupported zero-point unit {unit!r}; expected Jy")
    rows = []
    for tr in root.iter():
        if tr.tag.rsplit("}", 1)[-1] != "TR":
            continue
        cells = [cell.text for cell in tr if cell.tag.rsplit("}", 1)[-1] == "TD"]
        if len(cells) >= 2:
            try:
                rows.append((float(cells[0]), float(cells[1])))
            except (TypeError, ValueError):
                continue
    if len(rows) < 2:
        raise ValueError(f"{path} contains no usable wavelength/transmission rows")
    wavelength_unit_name = metadata.get("WavelengthUnit", "Angstrom")
    try:
        curve = make_curve(rows, wavelength_unit_name=wavelength_unit_name,
                           name=f"SVO filter {label}", source=path).data
        detector_type = int(float(metadata.get("DetectorType", 1)))
    except ValueError as exc:
        raise ValueError(f"{path} has invalid wavelength or DetectorType metadata: {exc}") from exc
    if detector_type not in {0, 1}:
        raise ValueError(f"{path} has unsupported DetectorType={detector_type}; expected 0 or 1")
    if not np.all(np.isfinite(curve)) or zero_point <= 0:
        raise ValueError(f"{path} contains non-finite data or a non-positive zero point")
    numeric_metadata = {}
    for name, value in metadata.items():
        try:
            numeric_metadata[name] = float(value)
        except ValueError:
            numeric_metadata[name] = value
    for name in ("WavelengthPivot", "WavelengthRef", "WavelengthMean", "WavelengthEff", "WavelengthMin",
                 "WavelengthMax", "WavelengthCen", "WidthEff", "FWHM"):
        if name in numeric_metadata and isinstance(numeric_metadata[name], float):
            numeric_metadata[name] = float((numeric_metadata[name] * wavelength_unit(wavelength_unit_name)).to_value(u.AA))
    return FilterProfile(label, magnitude_system.upper(), curve, zero_point, detector_type, numeric_metadata, Path(path))


def load_filter_profile(directory, label, magnitude_system):
    """Load ``label`` calibrated in either ``Vega`` or ``AB`` magnitudes."""
    system = magnitude_system.strip().upper()
    if system not in {"VEGA", "AB"}:
        raise ValueError("Magnitude system must be Vega or AB")
    directory = Path(directory)
    if label not in set(list_filter_labels(directory)):
        raise KeyError(f"Unknown filter {label!r}")
    if label == "BLANK":
        for entry in ("filters/BLANK.xml", "BLANK.xml", "filters/BLANK"):
            try:
                profile = _parse_votable(_resolve_profile_path(directory, entry), label, "AB")
                # A truly filter-free magnitude has no Vega passband.  The
                # special profile is therefore defined on the AB 3631-Jy
                # scale for both selector states.
                return profile if system == "AB" else replace(profile, magnitude_system="VEGA", zero_point_jy=AB_ZERO_POINT_JY)
            except FileNotFoundError:
                continue
        raise FileNotFoundError("BLANK.xml not found in the data filters directory")
    entry = f"{label}_{'Vega' if system == 'VEGA' else 'AB'}"
    return _parse_votable(_resolve_profile_path(directory, entry), label, system)


def magnitude_to_ab(magnitude, profile):
    """Convert a magnitude in ``profile``'s system into its AB equivalent."""
    return float(magnitude) - 2.5 * np.log10(float(profile.zero_point_jy) / AB_ZERO_POINT_JY)


def generic_zenith_atmosphere_curve():
    """Broad-band zenith transmission from the supplied ING skyext table.

    It replaces the atmospheric column formerly embedded in resp_filters.dat.
    Values are intentionally only a smooth broad-band baseline until a
    site-specific extinction curve is added in a later physics revision.
    """
    wavelength_aa = np.array([3000.0, 3650.0, 4450.0, 5510.0, 6580.0, 8060.0, 10000.0,
                              12500.0, 16500.0, 22000.0, 26000.0])
    extinction_mag = np.array([0.550, 0.550, 0.250, 0.150, 0.090, 0.060, 0.050,
                               0.100, 0.110, 0.070, 0.070])
    return np.column_stack((wavelength_aa, 10.0 ** (-0.4 * extinction_mag)))
