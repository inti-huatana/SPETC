"""Focused v9 safeguards; run with ``python tests/test_spectral_pipeline.py``."""

from pathlib import Path
import tempfile

import numpy as np
from astropy.io import fits

from detector import Detector, load_qe_curve
from etc_physics import synthetic_magnitude
from spectral_utils import interpolate_checked, interpolate_zero_filled, load_fits_transmission_curve
from spectroscopy import SpectroscopyETC


def test_explicit_nm_qe_conversion():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as stream:
        stream.write("400 0.5\n700 0.8\n")
        path = Path(stream.name)
    try:
        curve = load_qe_curve(path, "nm")
        assert np.allclose(curve[:, 0], [4000.0, 7000.0])
    finally:
        path.unlink(missing_ok=True)


def test_svo_energy_and_photon_semantics_differ_for_coloured_sed():
    wavelength = np.linspace(3500.0, 9500.0, 4000)
    template = np.column_stack((wavelength, (wavelength / 5500.0) ** -4 * 1e-9))
    band = np.column_stack((wavelength, np.ones_like(wavelength)))
    energy_mag = synthetic_magnitude(template, band, 3631.0, detector_type=0)
    photon_mag = synthetic_magnitude(template, band, 3631.0, detector_type=1)
    assert abs(energy_mag - photon_mag) > 0.01


def test_spectroscopy_rejects_template_outside_requested_range():
    wavelength = np.linspace(5000.0, 6000.0, 100)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-9)))
    band = np.column_stack((wavelength, np.ones_like(wavelength)))
    qe = np.array([[4500.0, 0.8], [7000.0, 0.8]])
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    detector = Detector(10.0, 2.0, 10000.0, 16)
    atmosphere = {"airmass": 1.0, "seeing_arcsec": 1.0, "transmission_curve": None}
    sky = {"sky_mag": 20.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True}
    try:
        SpectroscopyETC(telescope, detector, atmosphere, sky).compute_spectroscopy(
            template, 1000.0, 1.0, 10.0, (7000.0, 8000.0), 10.0, qe, band, visual_band=band)
    except ValueError as exc:
        assert "template spectrum covers" in str(exc)
    else:
        raise AssertionError("out-of-coverage spectrum must not silently return zero counts")


def test_spectroscopic_observing_filter_applies_to_source_and_sky():
    wavelength = np.linspace(5000.0, 6000.0, 100)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-9)))
    reference_band = np.column_stack((wavelength, np.ones_like(wavelength)))
    blocking_filter = np.column_stack((wavelength, np.zeros_like(wavelength)))
    qe = np.array([[4500.0, 0.8], [7000.0, 0.8]])
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    detector = Detector(10.0, 2.0, 10000.0, 16)
    atmosphere = {"airmass": 1.0, "seeing_arcsec": 1.0, "transmission_curve": None}
    sky = {"sky_mag": 20.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True}
    result = SpectroscopyETC(telescope, detector, atmosphere, sky).compute_spectroscopy(
        template, 1000.0, 1.0, 10.0, (5200.0, 5800.0), 10.0, qe, reference_band,
        observing_filter=blocking_filter)
    assert np.allclose(result["photons_source_es"], 0.0)
    assert np.allclose(result["photons_sky_es"], 0.0)


def test_fits_atmosphere_wavelength_unit_is_honoured():
    """FITS table units are converted to the common internal Angstrom grid."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "earth_atmosphere.fits"
        columns = [
            fits.Column(name="WAVE", format="D", unit="nm", array=np.array([350.0, 550.0, 900.0])),
            fits.Column(name="TRANSMISSION", format="D", array=np.array([0.6, 0.8, 0.7])),
        ]
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
        curve = load_fits_transmission_curve(path)
    assert np.allclose(curve.wavelength_aa, [3500.0, 5500.0, 9000.0])


def test_two_column_fits_image_is_accepted():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "earth_atmosphere_image.fits"
        header = fits.Header({"CUNIT1": "nm"})
        fits.PrimaryHDU(np.array([[350.0, 550.0, 900.0], [0.6, 0.8, 0.7]]), header=header).writeto(path)
        curve = load_fits_transmission_curve(path)
    assert np.allclose(curve.wavelength_aa, [3500.0, 5500.0, 9000.0])


def test_display_interpolation_zero_fills_missing_coverage():
    curve = np.array([[5000.0, 0.4], [6000.0, 0.8]])
    values = interpolate_zero_filled(np.array([4500.0, 5000.0, 5500.0, 6500.0]), curve)
    assert np.allclose(values, [0.0, 0.4, 0.6, 0.0])


def test_paranal_extinction_coefficient_is_converted_once_to_zenith_transmission():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "paranal_extinction.fits"
        columns = [
            fits.Column(name="WAVEAIR", format="D", unit="angstrom", array=np.array([3000.0, 5500.0, 10500.0])),
            fits.Column(name="EXTINCTION", format="D", array=np.array([0.10, 0.20, 0.30])),
        ]
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
        curve = load_fits_transmission_curve(path)
    assert np.allclose(curve.values, 10.0 ** (-0.4 * np.array([0.10, 0.20, 0.30])))
    try:
        interpolate_checked(np.array([10501.0]), curve, "Paranal atmosphere")
    except ValueError as exc:
        assert "covers" in str(exc)
    else:
        raise AssertionError("Paranal atmospheric coverage must be a hard limit")


if __name__ == "__main__":
    test_explicit_nm_qe_conversion()
    test_svo_energy_and_photon_semantics_differ_for_coloured_sed()
    test_spectroscopy_rejects_template_outside_requested_range()
    test_spectroscopic_observing_filter_applies_to_source_and_sky()
    test_fits_atmosphere_wavelength_unit_is_honoured()
    test_two_column_fits_image_is_accepted()
    test_display_interpolation_zero_fills_missing_coverage()
    test_paranal_extinction_coefficient_is_converted_once_to_zenith_transmission()
    print("PASS: v9 spectral pipeline safeguards")
