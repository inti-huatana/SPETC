"""Build calibrated SVO-style filter profiles from a transmission curve.

Amateur filters (Astrodon, Astronomik, Baader, Optolong, Chroma RGB and
narrowband sets) are not in the SVO Filter Profile Service, but a measured
or digitized transmission curve is all that is needed: the zero points are
*computed*, not measured, following the same synthetic convention SVO uses:

* AB zero point: 3631 Jy at every wavelength by definition (Oke & Gunn).
* Vega zero point: the photon-weighted mean flux density of the CALSPEC
  Vega spectrum (alpha_lyr_stis, Bohlin) through the filter.

The output is a full SVO-format VOTable, with the complete photdm
utype/UCD annotation and every characterising quantity measured from the
curve using the SVO definitions (WavelengthMean/Eff/Min/Max/Cen/Pivot/
Peak/Phot, WidthEff, FWHM, Fsun from the CALSPEC solar spectrum), so the
files are drop-in interchangeable with profiles downloaded from SVO.

Usage:
    python3 make_filter_profile.py <name> <transmission_file> [unit] [out_dir]

``name`` is conventionally ``System.Band`` (e.g. ``Astrodon.G``);
``unit`` is Angstrom (default), nm or um; ``out_dir`` defaults to
``data/filters``.  Writes ``<name>_Vega.xml`` and ``<name>_AB.xml`` and
prints the lines to add to ``data/filters.list``.  Transmission may be 0-1
or percent (auto-detected when the maximum exceeds 1.5).
"""

import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.constants import c

from spectral_utils import load_two_column_curve

_DATA_DIR = Path(__file__).resolve().parent / "data"
VEGA_SPECTRUM = _DATA_DIR / "bpgs" / "alpha_lyr_stis_004.ascii"
SUN_SPECTRUM = _DATA_DIR / "bpgs" / "sun_reference_stis_001.ascii"
AB_ZERO_POINT_JY = 3631.0


def _trapz(y, x):
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return integrate(y, x)


def _interpolated_reference(path, grid_aa, name):
    reference = load_two_column_curve(path, name=name)
    if grid_aa.min() < reference.wavelength_aa[0] or grid_aa.max() > reference.wavelength_aa[-1]:
        return None
    return np.interp(grid_aa, reference.wavelength_aa, reference.values)


def synthetic_vega_zero_point_jy(wavelength_aa, transmission, vega_spectrum_path=VEGA_SPECTRUM):
    """Photon-weighted mean F_nu of the CALSPEC Vega spectrum in the band [Jy]."""
    grid = np.linspace(wavelength_aa.min(), wavelength_aa.max(), 8192)
    response = np.interp(grid, wavelength_aa, transmission, left=0.0, right=0.0)
    f_lambda = _interpolated_reference(vega_spectrum_path, grid, "CALSPEC Vega spectrum")
    if f_lambda is None:
        raise ValueError(
            f"The filter ({wavelength_aa.min():.0f}-{wavelength_aa.max():.0f} Å) exceeds the "
            "CALSPEC Vega spectrum coverage; a Vega zero point cannot be computed synthetically.")
    numerator = _trapz(f_lambda * response * grid, grid)
    c_aa_s = c.to_value(u.AA / u.s)
    denominator = _trapz((c_aa_s / grid**2) * response * grid, grid)
    mean_f_nu = (numerator / denominator) * u.erg / (u.s * u.cm**2 * u.Hz)
    return float(mean_f_nu.to_value(u.Jy))


def measure_band_quantities(wavelength_aa, transmission):
    """All SVO characterising quantities, using the SVO definitions."""
    wavelength = np.asarray(wavelength_aa, dtype=float)
    band = np.asarray(transmission, dtype=float)
    grid = np.linspace(wavelength.min(), wavelength.max(), 8192)
    response = np.interp(grid, wavelength, band, left=0.0, right=0.0)
    peak = response.max()
    integral_t = _trapz(response, grid)
    quantities = {
        "WavelengthMean": _trapz(grid * response, grid) / integral_t,
        "WavelengthPivot": np.sqrt(_trapz(grid * response, grid) / _trapz(response / grid, grid)),
        # Peak and Min/Max follow SVO literally: tabulated lambda values, not
        # interpolated crossings ("the first/last lambda value with a
        # transmission at least 1% of maximum transmission").
        "WavelengthPeak": wavelength[int(np.argmax(band))],
        "WidthEff": integral_t / peak,
    }
    above_1pc = np.where(band >= 0.01 * band.max())[0]
    quantities["WavelengthMin"] = wavelength[above_1pc[0]]
    quantities["WavelengthMax"] = wavelength[above_1pc[-1]]
    # FWHM and its central wavelength from the half-maximum crossings.
    above_half = np.where(response >= 0.5 * peak)[0]
    half_lo, half_hi = grid[above_half[0]], grid[above_half[-1]]
    quantities["FWHM"] = half_hi - half_lo
    quantities["WavelengthCen"] = 0.5 * (half_lo + half_hi)
    # Vega-weighted quantities.
    vega = _interpolated_reference(VEGA_SPECTRUM, grid, "CALSPEC Vega spectrum")
    if vega is not None:
        vega_weight = response * vega
        quantities["WavelengthEff"] = _trapz(grid * vega_weight, grid) / _trapz(vega_weight, grid)
        quantities["WavelengthPhot"] = (_trapz(grid**2 * vega_weight, grid)
                                        / _trapz(grid * vega_weight, grid))
    # Mean solar flux density through the band (photon-weighted, CALSPEC Sun).
    sun = _interpolated_reference(SUN_SPECTRUM, grid, "CALSPEC solar spectrum")
    if sun is not None:
        quantities["Fsun"] = (_trapz(sun * response * grid, grid)
                              / _trapz(response * grid, grid))
    quantities["WavelengthRef"] = quantities["WavelengthPivot"]
    return quantities


_WAVELENGTH_PARAMS = (
    ("WavelengthRef", "em.wl;meta.main", "photdm:PhotometryFilter.spectralLocation.value",
     "Reference wavelength. Defined as the same as the pivot wavelength."),
    ("WavelengthMean", "em.wl", "",
     "Mean wavelength. Defined as integ[x*filter(x) dx]/integ[filter(x) dx]"),
    ("WavelengthEff", "em.wl.effective", "",
     "Effective wavelength. Defined as integ[x*filter(x)*vega(x) dx]/integ[filter(x)*vega(x) dx]"),
    ("WavelengthMin", "em.wl;stat.min", "photdm:PhotometryFilter.bandwidth.start.value",
     "Minimum filter wavelength. Defined as the first lambda value with a transmission at least 1% of maximum transmission"),
    ("WavelengthMax", "em.wl;stat.max", "photdm:PhotometryFilter.bandwidth.stop.value",
     "Maximum filter wavelength. Defined as the last lambda value with a transmission at least 1% of maximum transmission"),
    ("WidthEff", "instr.bandwidth", "photdm:PhotometryFilter.bandwidth.extent.value",
     "Effective width. Defined as integ[filter(x) dx]/max(filter(x)). Equivalent to the horizontal size of a rectangle with height equal to maximum transmission and with the same area that the one covered by the filter transmission curve."),
    ("WavelengthCen", "em.wl", "",
     "Central wavelength. Defined as the central wavelength between the two points defining FWMH"),
    ("WavelengthPivot", "em.wl", "",
     "Pivot wavelength. Defined as sqrt{integ[x*filter(x) dx]/integ[filter(x) dx/x]}"),
    ("WavelengthPeak", "em.wl", "",
     "Peak wavelength. Defined as the lambda value with larger transmission"),
    ("WavelengthPhot", "em.wl", "",
     "Photon distribution based effective wavelength. Defined as integ[x^2*filter(x)*vega(x) dx]/integ[x*filter(x)*vega(x) dx]"),
    ("FWHM", "instr.bandwidth", "",
     "Full width at half maximum. Defined as the difference between the two wavelengths for which filter transmission is half maximum"),
)


def _param(name, value, *, unit=None, ucd=None, utype=None, datatype="double",
           arraysize=None, description=None):
    attributes = [f'name="{name}"', f'value="{value}"']
    if unit:
        attributes.append(f'unit="{unit}"')
    if ucd:
        attributes.append(f'ucd="{ucd}"')
    if utype:
        attributes.append(f'utype="{utype}"')
    attributes.append(f'datatype="{datatype}"')
    if arraysize:
        attributes.append(f'arraysize="{arraysize}"')
    if description:
        return (f'    <PARAM {" ".join(attributes)} >\n'
                f'       <DESCRIPTION>{description}</DESCRIPTION>\n'
                f'    </PARAM>')
    return f'    <PARAM {" ".join(attributes)}/>'


def _votable(name, mag_sys, zero_point_jy, detector_type, quantities, wavelength_aa,
             transmission, description, phot_system, band, facility, components,
             profile_reference, calibration_reference):
    lines = ['<?xml version="1.0"?>',
             '<VOTABLE version="1.1" xsi:schemaLocation="http://www.ivoa.net/xml/VOTable/v1.1" '
             'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
             '  <INFO name="QUERY_STATUS" value="OK"/>',
             '  <RESOURCE type="results">',
             '    <TABLE utype="photdm:PhotometryFilter.transmissionCurve.spectrum">',
             _param("FilterProfileService", "SPETC/make_filter_profile", ucd="meta.ref.ivorn",
                    utype="PhotometryFilter.fpsIdentifier", datatype="char", arraysize="*"),
             _param("filterID", f"SPETC/{name}", ucd="meta.ref.ivoid",
                    utype="photdm:PhotometryFilter.identifier", datatype="char", arraysize="*"),
             _param("WavelengthUnit", "Angstrom", ucd="meta.unit",
                    utype="PhotometryFilter.SpectralAxis.unit", datatype="char", arraysize="*"),
             _param("WavelengthUCD", "em.wl", ucd="meta.ucd",
                    utype="PhotometryFilter.SpectralAxis.UCD", datatype="char", arraysize="*"),
             _param("Description", description, ucd="meta.note",
                    utype="photdm:PhotometryFilter.description", datatype="char", arraysize="*"),
             _param("PhotSystem", phot_system, utype="photdm:PhotometricSystem.description",
                    datatype="char", arraysize="*", description="Photometric system"),
             _param("DetectorType", str(int(detector_type)), ucd="meta.code",
                    utype="photdm:PhotometricSystem.detectorType", datatype="char", arraysize="*",
                    description="Detector type. 0:Energy counter, 1:Photon counter."),
             _param("Band", band, ucd="instr.bandpass",
                    utype="photdm:PhotometryFilter.bandName", datatype="char", arraysize="*"),
             _param("Facility", facility, ucd="instr.obsty", datatype="char", arraysize="*",
                    description="Observational facility"),
             _param("ProfileReference", profile_reference, datatype="char", arraysize="*"),
             _param("CalibrationReference", calibration_reference, datatype="char", arraysize="*"),
             _param("components", components, datatype="char", arraysize="*",
                    description="Transmission components")]
    for key, ucd, utype, definition in _WAVELENGTH_PARAMS:
        if key in quantities:
            lines.append(_param(key, f"{quantities[key]:.9f}", unit="Angstrom", ucd=ucd,
                                utype=utype or None, description=definition))
    if "Fsun" in quantities:
        lines.append(_param("Fsun", f"{quantities['Fsun']:.9f}", unit="erg/cm2/s/A",
                            ucd="phot.flux.density",
                            description="Sun flux. Mean solar flux density through the band, "
                                        "computed from the CALSPEC solar reference spectrum."))
    lines += [
        _param("PhotCalID", f"SPETC/{name}/{mag_sys}", ucd="meta.id",
               utype="photdm:PhotCal.identifier", datatype="char", arraysize="*"),
        _param("MagSys", mag_sys, ucd="meta.code",
               utype="photdm:PhotCal.MagnitudeSystem.type", datatype="char", arraysize="*"),
        _param("ZeroPoint", f"{zero_point_jy:.4f}", unit="Jy", ucd="phot.flux.density",
               utype="photdm:PhotCal.zeroPoint.flux.value"),
        _param("ZeroPointUnit", "Jy", ucd="meta.unit",
               utype="photdm:PhotCal.ZeroPoint.flux.unitexpression", datatype="char", arraysize="*"),
        _param("ZeroPointType", "Pogson", ucd="meta.code",
               utype="photdm:PhotCal.ZeroPoint.type", datatype="char", arraysize="*"),
        '      <FIELD name="Wavelength" utype="spec:Data.SpectralAxis.Value" ucd="em.wl" '
        'unit="Angstrom" datatype="double"/>',
        '      <FIELD name="Transmission" utype="spec:Data.FluxAxis.Value" ucd="phys.transmission" '
        'unit="" datatype="double"/>',
        '      <DATA>',
        '        <TABLEDATA>']
    for wavelength, value in zip(wavelength_aa, transmission):
        lines += ['          <TR>',
                  f'            <TD>{wavelength:.4f}</TD>',
                  f'            <TD>{value:.10f}</TD>',
                  '          </TR>']
    lines += ['        </TABLEDATA>', '      </DATA>', '    </TABLE>',
              '  </RESOURCE>', '</VOTABLE>', '']
    return "\n".join(lines)


def build_filter_profiles(name, transmission_path, wavelength_unit="Angstrom",
                          out_dir="data/filters", detector_type=1, description="",
                          phot_system=None, facility=None, components="Filter",
                          profile_reference="", calibration_reference=""):
    """Write SVO-format ``<name>_Vega.xml`` and ``<name>_AB.xml`` profiles."""
    curve = load_two_column_curve(transmission_path, wavelength_unit_name=wavelength_unit,
                                  name=f"{name} transmission")
    wavelength = curve.wavelength_aa
    transmission = np.clip(np.asarray(curve.values, dtype=float), 0.0, None)
    if transmission.max() > 1.5:  # percent input
        transmission = transmission / 100.0
    transmission = np.clip(transmission, 0.0, 1.0)
    if not np.any(transmission > 0):
        raise ValueError("The transmission curve has no positive values.")
    quantities = measure_band_quantities(wavelength, transmission)
    vega_zp = synthetic_vega_zero_point_jy(wavelength, transmission)
    system, _, band = name.rpartition(".")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        detector_type=detector_type, quantities=quantities, wavelength_aa=wavelength,
        transmission=transmission,
        description=description or f"{name} (user transmission, synthetic zero point)",
        phot_system=phot_system or (system or "SPETC"), band=band or name,
        facility=facility or (system or "SPETC"), components=components,
        profile_reference=profile_reference or f"User-supplied transmission: {Path(transmission_path).name}",
        calibration_reference=calibration_reference
        or "Synthetic zero point from the CALSPEC Vega spectrum (Bohlin), SVO convention")
    written = []
    for mag_sys, zero_point in (("Vega", vega_zp), ("AB", AB_ZERO_POINT_JY)):
        path = out_dir / f"{name}_{mag_sys}.xml"
        path.write_text(_votable(name, mag_sys, zero_point, **common), encoding="utf-8")
        written.append(path)
    return {"vega_zero_point_jy": vega_zp, "quantities": quantities, "files": written}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    name, transmission_path = sys.argv[1], sys.argv[2]
    unit = sys.argv[3] if len(sys.argv) > 3 else "Angstrom"
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "data/filters"
    result = build_filter_profiles(name, transmission_path, unit, out_dir)
    quantities = result["quantities"]
    print(f"{name}: pivot {quantities['WavelengthPivot']:.1f} A, FWHM {quantities['FWHM']:.1f} A, "
          f"Weff {quantities['WidthEff']:.1f} A, Vega ZP {result['vega_zero_point_jy']:.1f} Jy "
          f"(AB ZP {AB_ZERO_POINT_JY:.0f} Jy)")
    if "Fsun" in quantities:
        print(f"Fsun {quantities['Fsun']:.3f} erg/cm2/s/A")
    for path in result["files"]:
        print(f"wrote {path}")
    print(f"Add to data/filters.list:\n  filters/{name}_Vega\n  filters/{name}_AB")


if __name__ == "__main__":
    main()
