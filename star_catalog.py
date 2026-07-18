"""
star_catalog.py
interpola.db reader and automatic star spectrum loader.

interpola.db record format, from the Fortran read statement:

    type db_skel
         integer :: nrow
         real(kind=8) :: mv0, bv0
         character(len=10) :: name
         character(len=5)  :: spt
         character(len=28) :: filename
    end type
    read(1,'(i5,1x,2(f7.3,1x),a10,1x,a5,1x,a28)') db(i)

Fixed-width layout (0-indexed character positions):
    cols  0- 4  (5 chars)  : nrow   - number of data rows in the spectrum file
    col      5             : space
    cols  6-12  (7 chars)  : mv0    - reference (tabulated) magnitude of the spectrum
    col     13             : space
    cols 14-20  (7 chars)  : bv0    - B-V color
    col     21             : space
    cols 22-31 (10 chars)  : name   - star name
    col     32             : space
    cols 33-37  (5 chars)  : spt    - spectral type
    col     38             : space
    cols 39-66 (28 chars)  : filename - path to the 2-column ASCII spectrum file

This module parses interpola.db with that exact fixed-width layout, and
falls back to plain whitespace splitting (6 tokens) if a line does not
match it -- some interpola.db variants in circulation are written with a
simple free-format WRITE(*,*) instead of the formatted READ. Either way,
star spectrum selection happens by choosing an entry from this catalog:
no file dialog is ever opened for spectrum files. The corresponding
ASCII spectrum is located automatically from the 'filename' field,
resolved relative to the data directory (subdirectories are supported,
e.g. "bpgs/bpgs_92.ascii").

Flux normalization
------------------
The files are returned without changing their calibrated flux scale.  Their
``mv0`` catalogue field records the visual magnitude represented by the file:
zero for a BPGS visual-zero template, or a non-zero value for a spectrum held
at another visual normalization.  The physical engine applies the original
Fortran factor ``10**(0.4*mv0)`` and then rescales to the target's entered
reference magnitude.  Keeping this metadata with the spectrum is essential.

Spectral-type reference tables (ts_ms_*, ts_ps_*) are ported from apc.f90
for main-sequence and giant-branch stars, and are provided as a
supplementary lookup (typical Mv, B-V for a given spectral type) to help
"search by characteristics" even when browsing the catalog.
"""

from pathlib import Path
from collections import namedtuple

import numpy as np


StarRecord = namedtuple("StarRecord", ["nrow", "mv0", "bv0", "name", "spt", "filename"])


# ---------------------------------------------------------------------------
# Spectral type reference tables, ported from apc.f90 (module apc)
# Main sequence: SK82 par 4.1, pag 18, table 13
# Giants (RGB):  Gray 1992, appendix B, tab B.2 pag 432
# ---------------------------------------------------------------------------

_TS_MS_T = ["O3", "O4", "O5", "O6", "O7", "O8", "O9", "B0", "B1", "B2", "B3", "B5", "B7", "B8", "B9",
            "A0", "A1", "A2", "A3", "A5", "A7", "A8", "F0", "F2", "F5", "F8", "G0", "G2", "G5", "G8",
            "K0", "K1", "K2", "K3", "K4", "K5", "K7", "M0", "M1", "M2", "M3", "M4", "M5"]
_TS_MS_M = [-6.00, -5.90, -5.70, -5.50, -5.20, -4.90, -4.50, -4.00, -3.20, -2.45, -1.60,
            -1.20, -0.60, -0.25, 0.20, 0.65, 1.00, 1.30, 1.50, 1.95, 2.20, 2.40,
            2.70, 3.50, 3.60, 4.00, 4.40, 4.70, 5.10, 5.50, 5.90, 6.10, 6.40,
            6.65, 7.00, 7.35, 8.10, 8.80, 9.30, 9.90, 10.40, 11.30, 12.30]
_TS_MS_C = [-0.33, -0.33, -0.33, -0.33, -0.32, -0.32, -0.31, -0.30, -0.26, -0.24, -0.20,
            -0.17, -0.13, -0.11, -0.07, -0.02, 0.01, 0.05, 0.08, 0.15, 0.20, 0.25,
            0.30, 0.35, 0.44, 0.52, 0.58, 0.63, 0.68, 0.74, 0.81, 0.86, 0.91,
            0.96, 1.06, 1.15, 1.33, 1.40, 1.46, 1.49, 1.51, 1.54, 1.64]

_TS_PS_T = ["F0", "F2", "F5", "F6", "F7", "F8", "F9", "G0", "G1", "G2", "G3", "G4", "G5",
            "G6", "G7", "G8", "G9", "K0", "K1", "K2", "K3", "K4", "K5", "M0", "M1", "M2"]
_TS_PS_M = [1.2, 1.3, 1.4, 1.4, 1.4, 1.3, 1.3, 1.2, 1.1, 1.1, 1.0, 0.9, 0.8,
            0.8, 0.7, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.1, 0.0, -0.2, -0.3, -0.4]
_TS_PS_C = [0.300, 0.354, 0.430, 0.463, 0.500, 0.545, 0.595, 0.650, 0.710, 0.766, 0.816,
            0.859, 0.893, 0.918, 0.934, 0.954, 0.979, 1.015, 1.092, 1.159, 1.240, 1.385,
            1.485, 1.583, 1.600, 1.650]

SPECTRAL_TYPE_MS = {t: {"Mv": m, "B-V": c} for t, m, c in zip(_TS_MS_T, _TS_MS_M, _TS_MS_C)}
SPECTRAL_TYPE_GIANT = {t: {"Mv": m, "B-V": c} for t, m, c in zip(_TS_PS_T, _TS_PS_M, _TS_PS_C)}


def spectral_type_reference(spt, luminosity_class="V"):
    """
    Look up typical Mv, B-V for a spectral type, e.g. spectral_type_reference("G2").

    Parameters
    ----------
    spt : str
        Spectral type, e.g. "G2", "K5"
    luminosity_class : {"V", "III"}
        "V" = main sequence (default), "III" = giant branch

    Returns
    -------
    dict {"Mv": float, "B-V": float} or None if not tabulated
    """
    table = SPECTRAL_TYPE_MS if luminosity_class.upper() == "V" else SPECTRAL_TYPE_GIANT
    return table.get(spt.strip())


# ---------------------------------------------------------------------------
# interpola.db.csv parsing
# ---------------------------------------------------------------------------
#
# Format (header + comma-separated rows, fields whitespace-padded):
#
#   nrow ,Vmag   ,B-V     ,name       ,spt   ,filename
#    8839,  0.026,   0.000, Vega      ,   A0V, bpgs/alpha_lyr_stis_004.ascii
#    1467,-26.750,   0.630, Sun       ,   G2V, bpgs/sun_reference_stis_001.ascii

def parse_interpola_db(path):
    """
    Parse interpola.db.csv into a list of StarRecord.

    Columns: nrow, Vmag (-> mv0), B-V (-> bv0), name, spt, filename.
    The header line (first non-empty line) is skipped.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    list of StarRecord
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"interpola.db.csv not found at {path}")

    records = []
    with open(path, "r") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    if not lines:
        return records

    for line in lines[1:]:  # skip header
        fields = [tok.strip() for tok in line.split(",")]
        if len(fields) < 6:
            continue
        try:
            nrow = int(fields[0])
            mv0 = float(fields[1])
            bv0 = float(fields[2])
            name = fields[3]
            spt = fields[4]
            filename = fields[5]
        except ValueError:
            continue
        records.append(StarRecord(nrow, mv0, bv0, name, spt, filename))

    return records


# ---------------------------------------------------------------------------
# Search / selection
# ---------------------------------------------------------------------------

def search_stars(catalog, name_query=None, spt_prefix=None, bv_range=None, mv_range=None):
    """
    Filter a star catalog by name substring and/or physical characteristics.

    Parameters
    ----------
    catalog : list of StarRecord
    name_query : str, optional
        Case-insensitive substring match against star name.
    spt_prefix : str, optional
        Spectral type prefix, e.g. "G" matches G0..G9, "K5" matches only K5*.
    bv_range : (float, float), optional
        Inclusive B-V range.
    mv_range : (float, float), optional
        Inclusive reference-magnitude range.

    Returns
    -------
    list of StarRecord matching all given criteria.
    """
    results = catalog

    if name_query:
        q = name_query.strip().lower()
        results = [r for r in results if q in r.name.lower()]

    if spt_prefix:
        p = spt_prefix.strip().upper()
        results = [r for r in results if r.spt.upper().startswith(p)]

    if bv_range:
        lo, hi = bv_range
        results = [r for r in results if lo <= r.bv0 <= hi]

    if mv_range:
        lo, hi = mv_range
        results = [r for r in results if lo <= r.mv0 <= hi]

    return results


def find_star_by_name(catalog, name):
    """Exact (case-insensitive) name match. Returns StarRecord or None."""
    name_lower = name.strip().lower()
    for r in catalog:
        if r.name.lower() == name_lower:
            return r
    return None


# ---------------------------------------------------------------------------
# Automatic spectrum loading (no file dialog)
# ---------------------------------------------------------------------------

# Physical constants (erg*s, cm/s) - match apc.f90 / ephemeris.py
H_PLANCK = 6.62606876e-27
C_LIGHT = 2.99792458e10


# FITS spectral tables: accepted column names and wavelength units.  This is
# the STScI convention shared by CALSPEC, the solar-system surface-brightness
# atlas, the galactic emission-line atlas, the transient (SN/kilonova)
# templates and the CLOUDY planetary-nebula grids: a binary table with
# WAVELENGTH [Angstrom] and FLUX [erg/s/cm^2/A] columns.
_FITS_WAVELENGTH_COLUMNS = ("wavelength", "wave", "lambda", "wavelength_air", "wl")
_FITS_FLUX_COLUMNS = ("flux", "flam", "surf_bright", "surface_brightness", "sb")
_FITS_WAVELENGTH_UNITS = {
    "": 1.0, "angstrom": 1.0, "angstroms": 1.0, "a": 1.0,
    "nm": 10.0, "nanometer": 10.0, "nanometers": 10.0,
    "micron": 1.0e4, "microns": 1.0e4, "um": 1.0e4, "micrometer": 1.0e4, "micrometers": 1.0e4,
}


def load_fits_spectrum(path):
    """Read a CALSPEC/STScI-style FITS spectral table as (wavelength_A, flux) rows.

    Finds the first table HDU with a recognisable wavelength and flux column
    (WAVELENGTH/FLUX and common variants, case-insensitive), converts the
    wavelength unit from TUNIT when it is nm or micron, and returns an Nx2
    float array.  Flux units are returned as stored (CALSPEC: FLAM =
    erg/s/cm^2/A; the solar-system atlas: erg/s/cm^2/A/arcsec^2) - the ETC
    rescales every template to the user-entered magnitude, so only the
    spectral shape and the relative calibration matter.
    """
    from astropy.io import fits
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None or not getattr(data, "names", None):
                continue
            lookup = {str(name).strip().lower(): name for name in data.names}
            wave_key = next((k for k in _FITS_WAVELENGTH_COLUMNS if k in lookup), None)
            flux_key = next((k for k in _FITS_FLUX_COLUMNS if k in lookup), None)
            if wave_key is None or flux_key is None:
                continue
            wavelength = np.asarray(data[lookup[wave_key]], dtype=float).ravel()
            flux = np.asarray(data[lookup[flux_key]], dtype=float).ravel()
            unit = ""
            for index, name in enumerate(data.names, start=1):
                if name == lookup[wave_key]:
                    unit = str(hdu.header.get(f"TUNIT{index}", "")).strip().lower()
            scale = _FITS_WAVELENGTH_UNITS.get(unit)
            if scale is None:
                raise ValueError(f"{path}: unsupported wavelength unit {unit!r} in TUNIT.")
            return np.column_stack((wavelength * scale, flux))
    raise ValueError(
        f"{path} contains no table HDU with recognisable wavelength/flux columns "
        f"(accepted: {', '.join(_FITS_WAVELENGTH_COLUMNS)} / {', '.join(_FITS_FLUX_COLUMNS)}).")


def load_star_spectrum(record, data_dir):
    """
    Load the calibrated spectrum for a catalogue entry.

    The returned values retain the file's flux scale.  ``record.mv0`` is
    consumed later by the ETC to convert the distribution to visual zero and
    then to the user-entered reference magnitude.

    Two-column ASCII and CALSPEC/STScI-style FITS tables are supported; the
    format is chosen by the file extension (.fits/.fit).

    No file dialog is used: the spectrum file is located automatically
    from record.filename under data_dir (subdirectories supported).
    """
    data_dir = Path(data_dir)
    spec_path = data_dir / record.filename

    if not spec_path.exists():
        raise FileNotFoundError(
            f"Spectrum file for '{record.name}' not found: {spec_path}\n"
            f"(referenced by interpola.db as '{record.filename}')"
        )

    if spec_path.suffix.lower() in (".fits", ".fit"):
        raw = load_fits_spectrum(spec_path)
        good = (np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1])
                & (raw[:, 0] > 0) & (raw[:, 1] > 0))
        rows = raw[good]
        if rows.size == 0:
            raise ValueError(f"No valid (wavelength, flux>0) rows found in {spec_path}")
        spectrum = np.asarray(rows, dtype=float)
    else:
        rows = []
        with open(spec_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                wl = float(tokens[0])
                flux = float(tokens[1])
                if np.isfinite(wl) and np.isfinite(flux) and wl > 0 and flux > 0:
                    rows.append((wl, flux))

        if not rows:
            raise ValueError(f"No valid (wavelength, flux>0) rows found in {spec_path}")

        spectrum = np.asarray(rows, dtype=float)
    spectrum = spectrum[np.argsort(spectrum[:, 0])]
    keep = np.r_[True, np.diff(spectrum[:, 0]) > 0]
    spectrum = spectrum[keep]
    if len(spectrum) < 2:
        raise ValueError(f"Spectrum for '{record.name}' has fewer than two distinct wavelengths: {spec_path}")
    if record.nrow > 0 and abs(len(spectrum) - record.nrow) > 1:
        # Header/comment rows are common, hence this is intentionally only a diagnostic.
        print(f"Warning: {record.name} declares {record.nrow} rows but {len(spectrum)} usable rows were loaded.")
    return spectrum
