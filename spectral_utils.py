"""Unit-aware spectral curves and coverage checks used throughout SPETC."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import astropy.units as u
from astropy.io import fits


_UNIT_ALIASES = {
    "a": u.AA, "aa": u.AA, "angstrom": u.AA, "angstroms": u.AA, "å": u.AA,
    "nm": u.nm, "nanometer": u.nm, "nanometre": u.nm,
    "um": u.um, "micron": u.um, "microns": u.um, "µm": u.um,
}


def wavelength_unit(value):
    """Return an Astropy wavelength unit from an explicit user/data label."""
    key = str(value).strip().lower().replace(" ", "")
    if key not in _UNIT_ALIASES:
        raise ValueError("Wavelength unit must be Angstrom, nm, or um.")
    return _UNIT_ALIASES[key]


def unit_label(unit):
    unit = u.Unit(unit)
    if unit == u.AA:
        return "Angstrom"
    if unit == u.nm:
        return "nm"
    if unit == u.um:
        return "um"
    return unit.to_string()


@dataclass(frozen=True)
class SpectralCurve:
    """A finite, strictly increasing numerical curve stored internally in Å."""

    wavelength_aa: np.ndarray
    values: np.ndarray
    name: str = "spectral curve"
    source: str = ""

    @property
    def data(self):
        return np.column_stack((self.wavelength_aa, self.values))

    @property
    def coverage_aa(self):
        return float(self.wavelength_aa[0]), float(self.wavelength_aa[-1])


def make_curve(rows, *, wavelength_unit_name="Angstrom", name="spectral curve", source=""):
    """Validate and convert an Nx2 curve to internal Angstrom wavelengths."""
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 2:
        raise ValueError(f"{name} must contain at least two numerical wavelength/value rows.")
    raw_wave, ordinate = values[:, 0], values[:, 1]
    if not np.all(np.isfinite(raw_wave)) or not np.all(np.isfinite(ordinate)):
        raise ValueError(f"{name} contains non-finite values.")
    wave = (raw_wave * wavelength_unit(wavelength_unit_name)).to_value(u.AA)
    order = np.argsort(wave)
    wave, ordinate = wave[order], ordinate[order]
    unique = np.r_[True, np.diff(wave) > 0]
    wave, ordinate = wave[unique], ordinate[unique]
    if len(wave) < 2 or np.any(wave <= 0):
        raise ValueError(f"{name} requires at least two distinct positive wavelengths.")
    return SpectralCurve(wave, ordinate, name=name, source=str(source))


def as_curve(curve, name="spectral curve", wavelength_unit_name="Angstrom"):
    if isinstance(curve, SpectralCurve):
        return curve
    return make_curve(curve, wavelength_unit_name=wavelength_unit_name, name=name)


def require_coverage(target_wavelength_aa, curve, name=None):
    """Require a curve to fully cover a calculation interval; never silently zero-fill."""
    target = np.asarray(target_wavelength_aa, dtype=float)
    finite = target[np.isfinite(target)]
    if finite.size == 0:
        raise ValueError("Calculation wavelength grid is empty.")
    item = as_curve(curve, name or "spectral curve")
    lo, hi = float(finite.min()), float(finite.max())
    clo, chi = item.coverage_aa
    if lo < clo or hi > chi:
        raise ValueError(
            f"{name or item.name} covers {clo:.0f}–{chi:.0f} Å, but the requested calculation needs "
            f"{lo:.0f}–{hi:.0f} Å. Select compatible data or wavelength limits.")
    return item


def interpolate_checked(target_wavelength_aa, curve, name=None, *, clip=None):
    item = require_coverage(target_wavelength_aa, curve, name)
    result = np.interp(np.asarray(target_wavelength_aa, dtype=float), item.wavelength_aa, item.values)
    if clip is not None:
        result = np.clip(result, *clip)
    return result


def interpolate_zero_filled(target_wavelength_aa, curve, name="spectral curve", *, clip=None):
    """Interpolate a display-only curve, treating absent coverage as zero.

    Scientific ETC integrations must use :func:`interpolate_checked`.  This
    helper exists for diagnostic response plots, where showing the unavailable
    portion as zero is more useful than hiding the plot.
    """
    item = as_curve(curve, name=name)
    result = np.interp(np.asarray(target_wavelength_aa, dtype=float), item.wavelength_aa, item.values,
                       left=0.0, right=0.0)
    if clip is not None:
        result = np.clip(result, *clip)
    return result


def interpolate_edge_extended(target_wavelength_aa, curve, name="spectral curve", *, clip=None):
    """Interpolate, extending the first/last value outside the curve coverage.

    Appropriate for smooth physical quantities that certainly do not vanish
    outside a file's tabulated range — above all an atmospheric transmission
    curve, where zero-filling would model a perfectly opaque atmosphere.
    """
    item = as_curve(curve, name=name)
    result = np.interp(np.asarray(target_wavelength_aa, dtype=float), item.wavelength_aa, item.values)
    if clip is not None:
        result = np.clip(result, *clip)
    return result


def load_two_column_curve(path, *, wavelength_unit_name="Angstrom", name="spectral curve"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = re.split(r"[\s,;]+", line)
            if len(tokens) < 2:
                continue
            try:
                rows.append((float(tokens[0]), float(tokens[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numerical wavelength/value rows found in {path}")
    return make_curve(rows, wavelength_unit_name=wavelength_unit_name, name=name, source=path)


def _fits_column(data, names):
    lookup = {str(item).strip().lower(): item for item in data.names or ()}
    for candidate in names:
        if candidate in lookup:
            return np.asarray(data[lookup[candidate]], dtype=float), lookup[candidate]
    return None, None


def load_fits_transmission_curve(path):
    """Read an atmospheric FITS transmission curve in a documented convention.

    Direct transmission columns are accepted as fractions.  The local Paranal
    product is an ``EXTCOEFF_TABLE``: its ``EXTINCTION`` column is the
    extinction coefficient in mag per airmass, converted here to a zenith
    transmission.  The caller applies the requested airmass exactly once.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Earth atmospheric transmission FITS file not found: {path}")
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            if getattr(data, "names", None):
                wave, wave_name = _fits_column(
                    data, ("waveair", "wavelength", "wave", "lambda", "lam", "wl"))
                transmission, _ = _fits_column(
                    data, ("transmission", "trans", "throughput", "atmospheric_transmission", "atmtrans"))
                if wave is not None and transmission is not None:
                    unit = hdu.columns[wave_name].unit or hdu.header.get("TUNIT1") or "Angstrom"
                    return make_curve(np.column_stack((wave, transmission)), wavelength_unit_name=unit,
                                      name="earth atmospheric transmission", source=path)
                extinction, _ = _fits_column(data, ("extinction", "extcoeff", "extinction_coefficient"))
                if wave is not None and extinction is not None:
                    if not np.all(np.isfinite(extinction)):
                        raise ValueError("Atmospheric extinction coefficient contains non-finite values.")
                    unit = hdu.columns[wave_name].unit or hdu.header.get("TUNIT1") or "Angstrom"
                    zenith_transmission = np.power(10.0, -0.4 * extinction)
                    return make_curve(np.column_stack((wave, zenith_transmission)), wavelength_unit_name=unit,
                                      name="atmospheric extinction coefficient (converted to zenith transmission)",
                                      source=path)
            # A few local tools write a 2×N or N×2 image rather than a table.
            # Accept it explicitly; CUNIT1 specifies the wavelength column.
            image = np.asarray(data)
            if image.ndim == 2 and 2 in image.shape:
                rows = image.T if image.shape[0] == 2 else image
                if rows.shape[1] == 2:
                    unit = hdu.header.get("CUNIT1", "Angstrom")
                    return make_curve(rows, wavelength_unit_name=unit,
                                      name="earth atmospheric transmission", source=path)
    raise ValueError(
        f"{path} has no usable atmospheric-transmission table or two-column image. "
        "Accepted wavelength names: waveair/wavelength/wave/lambda/lam/wl; transmission names: "
        "transmission/trans/throughput/atmospheric_transmission; or an EXTINCTION coefficient in mag per airmass.")
