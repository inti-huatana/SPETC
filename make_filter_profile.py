"""Build calibrated SPETC/SVO-style filter profiles from a transmission curve.

Amateur filters (Astrodon, Astronomik, Baader, Optolong, Chroma RGB and
narrowband sets) are not in the SVO Filter Profile Service, but a measured
or digitized transmission curve is all that is needed: the zero points are
*computed*, not measured.  This tool applies the same synthetic convention
SVO itself uses:

* AB zero point: 3631 Jy at every wavelength by definition (Oke & Gunn).
* Vega zero point: the photon-weighted mean flux density of the CALSPEC
  Vega spectrum (alpha_lyr_stis, Bohlin) through the filter,
  ZP_nu = int F_lambda T lambda dlam / int (c/lambda^2) T lambda dlam,
  i.e. the flux a magnitude-zero star delivers in this band.  The CALSPEC
  Vega spectrum shipped in ``data/bpgs`` is the calibration source, so the
  produced profiles are on exactly the same scale as the shipped
  Johnson-Cousins/Sloan SVO files.

Usage:
    python3 make_filter_profile.py <name> <transmission_file> [unit] [out_dir]

``unit`` is Angstrom (default), nm or um; ``out_dir`` defaults to
``data/filters``.  Writes ``<name>_Vega.xml`` and ``<name>_AB.xml`` and
prints the line to add to ``data/filters.list``.  Transmission may be 0-1
or percent (auto-detected when the maximum exceeds 1.5).
"""

import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.constants import c

from spectral_utils import load_two_column_curve

VEGA_SPECTRUM = Path(__file__).resolve().parent / "data" / "bpgs" / "alpha_lyr_stis_004.ascii"
AB_ZERO_POINT_JY = 3631.0

_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.1">
<RESOURCE>
<TABLE utype="photdm:PhotometryFilter.transmissionCurve.spectrum">
<PARAM name="filterID" value="SPETC/{name}" datatype="char" arraysize="*"/>
<PARAM name="Description" value="{description}" datatype="char" arraysize="*"/>
<PARAM name="DetectorType" value="{detector_type}" datatype="char" arraysize="*"/>
<PARAM name="WavelengthUnit" value="Angstrom" datatype="char" arraysize="*"/>
<PARAM name="WavelengthPivot" value="{pivot:.4f}" datatype="double" unit="Angstrom"/>
<PARAM name="WavelengthMin" value="{wmin:.4f}" datatype="double" unit="Angstrom"/>
<PARAM name="WavelengthMax" value="{wmax:.4f}" datatype="double" unit="Angstrom"/>
<PARAM name="WidthEff" value="{width_eff:.4f}" datatype="double" unit="Angstrom"/>
<PARAM name="FWHM" value="{fwhm:.4f}" datatype="double" unit="Angstrom"/>
<PARAM name="MagSys" value="{mag_sys}" datatype="char" arraysize="*"/>
<PARAM name="ZeroPoint" value="{zero_point:.4f}" datatype="double" unit="Jy"/>
<PARAM name="ZeroPointUnit" value="Jy" datatype="char" arraysize="*"/>
<DATA>
<TABLEDATA>
{rows}
</TABLEDATA>
</DATA>
</TABLE>
</RESOURCE>
</VOTABLE>
"""


def _trapz(y, x):
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return integrate(y, x)


def synthetic_vega_zero_point_jy(wavelength_aa, transmission, vega_spectrum_path=VEGA_SPECTRUM):
    """Photon-weighted mean F_nu of the CALSPEC Vega spectrum in the band [Jy]."""
    vega = load_two_column_curve(vega_spectrum_path, name="CALSPEC Vega spectrum")
    lo, hi = wavelength_aa.min(), wavelength_aa.max()
    if lo < vega.wavelength_aa[0] or hi > vega.wavelength_aa[-1]:
        raise ValueError(
            f"The filter ({lo:.0f}-{hi:.0f} Å) exceeds the CALSPEC Vega spectrum coverage "
            f"({vega.wavelength_aa[0]:.0f}-{vega.wavelength_aa[-1]:.0f} Å); a Vega zero point "
            "cannot be computed synthetically for this band.")
    grid = np.linspace(lo, hi, 8192)
    response = np.interp(grid, wavelength_aa, transmission, left=0.0, right=0.0)
    f_lambda = np.interp(grid, vega.wavelength_aa, vega.values)  # erg/s/cm2/A
    numerator = _trapz(f_lambda * response * grid, grid)         # photon weighting
    c_aa_s = c.to_value(u.AA / u.s)
    denominator = _trapz((c_aa_s / grid**2) * response * grid, grid)
    mean_f_nu = (numerator / denominator) * u.erg / (u.s * u.cm**2 * u.Hz)
    return float(mean_f_nu.to_value(u.Jy))


def _band_metadata(wavelength_aa, transmission):
    pivot = np.sqrt(_trapz(transmission * wavelength_aa, wavelength_aa)
                    / _trapz(transmission / wavelength_aa, wavelength_aa))
    width_eff = _trapz(transmission, wavelength_aa) / transmission.max()
    half = 0.5 * transmission.max()
    above = np.where(transmission >= half)[0]
    fwhm = wavelength_aa[above[-1]] - wavelength_aa[above[0]] if above.size >= 2 else width_eff
    return pivot, width_eff, fwhm


def build_filter_profiles(name, transmission_path, wavelength_unit="Angstrom",
                          out_dir="data/filters", detector_type=1, description=""):
    """Write ``<name>_Vega.xml`` and ``<name>_AB.xml`` beside the shipped filters."""
    curve = load_two_column_curve(transmission_path, wavelength_unit_name=wavelength_unit,
                                  name=f"{name} transmission")
    wavelength = curve.wavelength_aa
    transmission = np.clip(np.asarray(curve.values, dtype=float), 0.0, None)
    if transmission.max() > 1.5:  # percent input
        transmission = transmission / 100.0
    transmission = np.clip(transmission, 0.0, 1.0)
    if not np.any(transmission > 0):
        raise ValueError("The transmission curve has no positive values.")
    pivot, width_eff, fwhm = _band_metadata(wavelength, transmission)
    vega_zp = synthetic_vega_zero_point_jy(wavelength, transmission)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"<TR><TD>{w:.4f}</TD><TD>{t:.6f}</TD></TR>"
                     for w, t in zip(wavelength, transmission))
    written = []
    for mag_sys, zero_point in (("Vega", vega_zp), ("AB", AB_ZERO_POINT_JY)):
        payload = _XML_TEMPLATE.format(
            name=name, description=description or f"{name} (user transmission, synthetic zero point)",
            detector_type=int(detector_type), pivot=pivot, wmin=wavelength.min(),
            wmax=wavelength.max(), width_eff=width_eff, fwhm=fwhm, mag_sys=mag_sys,
            zero_point=zero_point, rows=rows)
        path = out_dir / f"{name}_{mag_sys}.xml"
        path.write_text(payload, encoding="utf-8")
        written.append(path)
    return {"vega_zero_point_jy": vega_zp, "pivot_aa": pivot, "fwhm_aa": fwhm,
            "width_eff_aa": width_eff, "files": written}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    name, transmission_path = sys.argv[1], sys.argv[2]
    unit = sys.argv[3] if len(sys.argv) > 3 else "Angstrom"
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "data/filters"
    result = build_filter_profiles(name, transmission_path, unit, out_dir)
    print(f"{name}: pivot {result['pivot_aa']:.1f} A, FWHM {result['fwhm_aa']:.1f} A, "
          f"Vega ZP {result['vega_zero_point_jy']:.1f} Jy (AB ZP {AB_ZERO_POINT_JY:.0f} Jy)")
    for path in result["files"]:
        print(f"wrote {path}")
    print(f"Add to data/filters.list:\n  filters/{name}_Vega\n  filters/{name}_AB")


if __name__ == "__main__":
    main()
