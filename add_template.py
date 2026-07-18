"""Import template spectra (CALSPEC FITS, STScI atlases, ASCII) into SPETC.

Usage:
    python3 add_template.py <file-or-directory> [name] [spectral_type]

Accepts a single spectrum or a directory (all ``*.fits``/``*.fit``/
``*.ascii``/``*.dat``/``*.txt`` inside).  For each spectrum the tool:

1. reads it (CALSPEC/STScI FITS binary tables with WAVELENGTH/FLUX columns,
   TUNIT-aware, or two-column ASCII);
2. copies it under ``data/imported/`` (unless it already lives below the
   data directory);
3. computes the synthetic Vega V magnitude of the file against the shipped
   Bessell.V profile - this is the catalogue ``mv0``, the magnitude the
   file *represents*, which the ETC uses to restore the visual-zero scale
   before rescaling to the user-entered target magnitude.  B-V is computed
   the same way from Bessell.B.  Response-weighted band coverage is
   reported; below 99% the value is flagged approximate (use the V
   reference filter for such templates, so the exact same-response path is
   taken and no truncated synthetic integral enters the calibration);
4. appends the row to ``data/interpola.db.csv``.

Because the ETC always rescales templates to the entered magnitude, files
with arbitrary or surface-brightness flux units (the transient templates,
the solar-system atlas) are perfectly usable: only the spectral shape and
the relative calibration matter, and ``mv0`` makes the normalization
self-consistent.
"""

import re
import shutil
import sys
from pathlib import Path

import numpy as np

import filter_catalog as fcat
from etc_physics import covered_response_fraction, synthetic_magnitude
from star_catalog import load_fits_spectrum, parse_interpola_db

DATA_DIR = Path(__file__).resolve().parent / "data"
IMPORT_SUBDIR = "imported"
SPECTRUM_SUFFIXES = (".fits", ".fit", ".ascii", ".dat", ".txt")


# CALSPEC filename suffixes: instrument/source tokens followed by a 3-digit
# version, e.g. alpha_lyr_stis_011, agk_81d266_stisnic_007, sun_mod_001.
_CALSPEC_INSTRUMENT_TOKENS = {
    "stis", "nic", "stisnic", "stiswfc", "stiswfcnic", "fos", "iue", "oke",
    "mod", "model", "wfirc", "wfc3", "uvis", "cohen", "reference",
}


def _clean_display_name(stem):
    """Strip CALSPEC instrument+version suffixes for the catalogue name."""
    tokens = stem.split("_")
    if tokens and re.fullmatch(r"\d{3}", tokens[-1]):
        tokens = tokens[:-1]
        while tokens and tokens[-1].lower() in _CALSPEC_INSTRUMENT_TOKENS:
            tokens = tokens[:-1]
    return "_".join(tokens) or stem


def _read_spectrum(path):
    path = Path(path)
    if path.suffix.lower() in (".fits", ".fit"):
        raw = load_fits_spectrum(path)
    else:
        raw = np.loadtxt(path, usecols=(0, 1), comments="#", ndmin=2)
    good = (np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1])
            & (raw[:, 0] > 0) & (raw[:, 1] > 0))
    spectrum = raw[good]
    if len(spectrum) < 2:
        raise ValueError(f"{path}: fewer than two valid (wavelength, flux>0) rows.")
    spectrum = spectrum[np.argsort(spectrum[:, 0])]
    keep = np.r_[True, np.diff(spectrum[:, 0]) > 0]
    return spectrum[keep]


def _band_magnitude(spectrum, profile):
    """Synthetic magnitude and response-weighted coverage in one profile."""
    coverage = covered_response_fraction(spectrum[:, 0], profile.transmission[:, 0],
                                         profile.transmission[:, 1], profile.detector_type)
    if coverage <= 0.0:
        return np.nan, coverage
    magnitude = synthetic_magnitude(spectrum, profile.transmission,
                                    profile.zero_point_jy, profile.detector_type)
    return magnitude, coverage


def import_template(path, name=None, spectral_type="", data_dir=DATA_DIR):
    """Import one spectrum; returns the catalogue row fields and diagnostics."""
    path = Path(path).resolve()
    data_dir = Path(data_dir).resolve()
    spectrum = _read_spectrum(path)
    v_profile = fcat.load_filter_profile(data_dir, "Bessell.V", "Vega")
    b_profile = fcat.load_filter_profile(data_dir, "Bessell.B", "Vega")
    v_mag, v_coverage = _band_magnitude(spectrum, v_profile)
    b_mag, b_coverage = _band_magnitude(spectrum, b_profile)
    if not np.isfinite(v_mag):
        raise ValueError(
            f"{path.name}: no overlap with the Bessell.V response - the catalogue mv0 "
            "cannot be derived. Supply a template that reaches the V band, or extend it.")
    bv_colour = (b_mag - v_mag) if np.isfinite(b_mag) and b_coverage >= 0.99 else 0.0
    if data_dir in path.parents:
        relative = path.relative_to(data_dir)
    else:
        destination = data_dir / IMPORT_SUBDIR / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        relative = destination.relative_to(data_dir)
    display_name = (name or _clean_display_name(path.stem))[:24].replace(",", " ")
    catalog_path = data_dir / "interpola.db.csv"
    existing = {record.name.lower() for record in parse_interpola_db(catalog_path)}
    if display_name.lower() in existing:
        raise ValueError(f"A catalogue entry named {display_name!r} already exists.")
    row = (f"{len(spectrum):5d},{v_mag:7.3f},{bv_colour:7.3f}, {display_name:<10s},"
           f"{spectral_type:>6s}, {relative.as_posix()}")
    with catalog_path.open("a", encoding="utf-8") as stream:
        stream.write(row + "\n")
    return {"name": display_name, "mv0": v_mag, "bv0": bv_colour,
            "v_coverage": v_coverage, "b_coverage": b_coverage,
            "rows": len(spectrum), "filename": relative.as_posix(),
            "wavelength_range_aa": (float(spectrum[0, 0]), float(spectrum[-1, 0]))}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    target = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else None
    spectral_type = sys.argv[3] if len(sys.argv) > 3 else ""
    files = ([target] if target.is_file()
             else sorted(p for p in target.iterdir() if p.suffix.lower() in SPECTRUM_SUFFIXES))
    if not files:
        raise SystemExit(f"No spectra found in {target}")
    if len(files) > 1 and name:
        raise SystemExit("A custom name is only valid for a single file.")
    for path in files:
        try:
            info = import_template(path, name if len(files) == 1 else None, spectral_type)
        except (OSError, ValueError) as exc:
            print(f"SKIP {path.name}: {exc}")
            continue
        lo, hi = info["wavelength_range_aa"]
        note = ""
        if info["v_coverage"] < 0.99:
            note = (f"  [mv0 approximate: only {100 * info['v_coverage']:.0f}% of the V response "
                    "covered - use the V reference filter for this template]")
        print(f"OK   {info['name']:<12s} {lo:8.0f}-{hi:8.0f} A  {info['rows']:6d} rows  "
              f"mv0={info['mv0']:+.3f}  B-V={info['bv0']:+.3f}  -> {info['filename']}{note}")


if __name__ == "__main__":
    main()
