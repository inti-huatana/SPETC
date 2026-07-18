"""Focused v9/v10 safeguards; run with ``python tests/test_spectral_pipeline.py``."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from astropy.io import fits

from detector import Detector, load_qe_curve
from ephemeris import parallactic_angle_deg, pickering_airmass
from etc_physics import psf_encircled_energy, psf_slit_throughput, synthetic_magnitude
from observing_conditions import (differential_refraction_arcsec, digitization_noise_e,
                                  scintillation_fractional_rms)
from sky_brightness import moonlight_scattering_function, sky_brightness_total, van_rhijn_factor
from spectral_utils import interpolate_checked, interpolate_zero_filled, load_fits_transmission_curve
from spectroscopy import SpectroscopyETC, grating_dispersion_aa_per_pixel


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


def test_krisciunas_schaefer_scattering_function():
    """f(rho) must follow the published K&S (1991) parenthesization."""
    rho = 90.0
    expected = 10.0 ** 5.36 * (1.06 + np.cos(np.radians(rho)) ** 2) + 10.0 ** (6.15 - rho / 40.0)
    assert np.isclose(moonlight_scattering_function(rho), expected)
    # The pre-fix misparenthesized form exceeded the truth by >1e3 at 30 deg.
    assert moonlight_scattering_function(30.0) < 1e6


def test_full_moon_brightens_v_sky_by_several_magnitudes():
    dark = sky_brightness_total(2026, 7, 18, 0, include_moon=False,
                                ecliptic_lat_deg=90, galactic_lat_deg=90)
    bright = sky_brightness_total(2026, 7, 18, 0, include_moon=True, lunar_phase_deg=0,
                                  moon_separation_deg=30, airmass_moon=1.2, airmass_target=1.1,
                                  ecliptic_lat_deg=90, galactic_lat_deg=90)
    brightening = dark[2] - bright[2]
    assert 3.0 < brightening < 7.0, brightening


def test_sky_is_brighter_on_ecliptic_and_galactic_plane():
    high = sky_brightness_total(2026, 7, 18, 0, include_moon=False,
                                ecliptic_lat_deg=90, galactic_lat_deg=90)
    low = sky_brightness_total(2026, 7, 18, 0, include_moon=False,
                               ecliptic_lat_deg=0, galactic_lat_deg=0)
    assert low[2] < high[2]
    # Zodiacal light must be symmetric about the ecliptic.
    south = sky_brightness_total(2026, 7, 18, 0, include_moon=False,
                                 ecliptic_lat_deg=-30, galactic_lat_deg=90)
    north = sky_brightness_total(2026, 7, 18, 0, include_moon=False,
                                 ecliptic_lat_deg=30, galactic_lat_deg=90)
    assert np.allclose(south, north)


def test_van_rhijn_is_sublinear_and_bounded():
    assert np.isclose(float(van_rhijn_factor(0.0)), 1.0)
    assert 5.0 < float(van_rhijn_factor(90.0)) < 7.0


def test_pickering_airmass_matches_sec_z_high_and_beats_it_low():
    assert np.isclose(pickering_airmass(90.0), 1.0, atol=1e-6)
    assert abs(pickering_airmass(60.0) - 1.0 / np.sin(np.radians(60.0))) < 0.01
    assert pickering_airmass(5.0) < 1.0 / np.sin(np.radians(5.0))  # sec z overestimates


def test_parallactic_angle_zero_on_meridian():
    assert np.isclose(parallactic_angle_deg(0.0, 20.0, 45.0), 0.0)
    assert parallactic_angle_deg(15.0, 20.0, 45.0) > 0.0  # west of meridian: q > 0


def test_moffat_has_stronger_wings_than_gaussian():
    ee_gauss = psf_encircled_energy(0.4, 0.8, "gaussian")
    ee_moffat = psf_encircled_energy(0.4, 0.8, "moffat", 2.5)
    assert ee_moffat < ee_gauss
    # Slit coupling decreases when the PSF is decentred by dispersion.
    centred = psf_slit_throughput(1.0, 1.0, "moffat", 2.5)
    offset = psf_slit_throughput(1.0, 1.0, "moffat", 2.5, offset_arcsec=0.5)
    assert offset < centred


def test_filippenko_dispersion_scale():
    """4000-5500 A differential refraction at z=45 deg is ~1 arcsec."""
    value = differential_refraction_arcsec(4000.0, 5500.0, np.sqrt(2.0))
    assert 0.8 < abs(value) < 1.3, value


def test_scintillation_and_digitization_orders_of_magnitude():
    frac = scintillation_fractional_rms(358.0, 1.5, 1366.0, 60.0)
    assert 5e-4 < frac < 5e-3, frac
    assert np.isclose(digitization_noise_e(2.5), 2.5 / np.sqrt(12.0))


def test_star_analyser_dispersion():
    """SA100 at 42 mm on 4.63 um pixels disperses ~11 A/pixel."""
    dispersion = grating_dispersion_aa_per_pixel(100.0, 42.0, 4.63)
    assert 10.0 < dispersion < 12.0, dispersion
    assert grating_dispersion_aa_per_pixel(200.0, 42.0, 4.63) == dispersion / 2.0


def test_extended_source_photometry_uniform_disc():
    from photometry import PhotometryETC
    wavelength = np.linspace(4000.0, 7000.0, 200)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-12)))
    band = np.column_stack((wavelength, np.ones_like(wavelength)))
    qe = np.column_stack((wavelength, np.full_like(wavelength, 0.8)))
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    detector = Detector(10.0, 2.0, 80000.0, 16)
    atmosphere = {"airmass": 1.0, "seeing_arcsec": 1.0, "transmission_curve": None}
    sky = {"sky_mag": 20.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True,
           "aperture_radius_arcsec": 2.0}
    etc = PhotometryETC(telescope, detector, atmosphere, sky)
    extended = etc.compute_photometry_single(template, band, qe, 12.0, 60.0,
                                             source_geometry="extended", source_area_arcsec2=400.0)
    point = etc.compute_photometry_single(template, band, qe, 12.0, 60.0)
    # The aperture (12.6 arcsec^2) captures only ~3% of a 400 arcsec^2 source.
    assert extended["photons_source_es"] < point["photons_source_es"]
    expected_fraction = np.pi * 4.0 / 400.0
    assert np.isclose(extended["photons_source_es"] / point["photons_source_es"] *
                      psf_encircled_energy(2.0, 1.0), expected_fraction, rtol=1e-6)
    # A uniform source cannot have a PSF-concentrated peak pixel.
    assert extended["peak_e_unclipped"] < point["peak_e_unclipped"]


if __name__ == "__main__":
    test_explicit_nm_qe_conversion()
    test_svo_energy_and_photon_semantics_differ_for_coloured_sed()
    test_spectroscopy_rejects_template_outside_requested_range()
    test_spectroscopic_observing_filter_applies_to_source_and_sky()
    test_fits_atmosphere_wavelength_unit_is_honoured()
    test_two_column_fits_image_is_accepted()
    test_display_interpolation_zero_fills_missing_coverage()
    test_paranal_extinction_coefficient_is_converted_once_to_zenith_transmission()
    test_krisciunas_schaefer_scattering_function()
    test_full_moon_brightens_v_sky_by_several_magnitudes()
    test_sky_is_brighter_on_ecliptic_and_galactic_plane()
    test_van_rhijn_is_sublinear_and_bounded()
    test_pickering_airmass_matches_sec_z_high_and_beats_it_low()
    test_parallactic_angle_zero_on_meridian()
    test_moffat_has_stronger_wings_than_gaussian()
    test_filippenko_dispersion_scale()
    test_scintillation_and_digitization_orders_of_magnitude()
    test_star_analyser_dispersion()
    test_extended_source_photometry_uniform_disc()
    print("PASS: v9 spectral pipeline safeguards + v10 physics regressions")
