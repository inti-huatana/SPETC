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


def test_spectroscopy_zero_fills_template_outside_coverage():
    """v10.1 contract: observing-side truncation zero-fills instead of raising."""
    wavelength = np.linspace(5000.0, 6000.0, 100)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-9)))
    band = np.column_stack((wavelength, np.ones_like(wavelength)))
    qe = np.array([[4500.0, 0.8], [8500.0, 0.8]])
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    detector = Detector(10.0, 2.0, 10000.0, 16)
    atmosphere = {"airmass": 1.0, "seeing_arcsec": 1.0, "transmission_curve": None}
    sky = {"sky_mag": 20.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True}
    result = SpectroscopyETC(telescope, detector, atmosphere, sky).compute_spectroscopy(
        template, 1000.0, 1.0, 10.0, (5500.0, 8000.0), 10.0, qe, band, visual_band=band)
    covered = result["wavelength_aa"] <= 6000.0
    assert np.all(result.loc[covered, "photons_source_es"].to_numpy()[:-1] > 0.0)
    assert np.allclose(result.loc[~covered, "photons_source_es"], 0.0)
    assert np.allclose(result.loc[~covered, "snr"], 0.0)


def test_calibration_band_coverage_is_still_strict():
    """Zero-fill must not reach the calibration integral: a partially covered
    reference band would silently rescale the whole spectrum (x1.9 for half
    coverage)."""
    from etc_physics import calibrated_template_magnitude
    wavelength = np.linspace(4000.0, 7000.0, 600)
    ref_band = np.column_stack((wavelength, ((wavelength >= 5000) & (wavelength <= 6000)).astype(float)))
    vis_band = np.column_stack((wavelength, ((wavelength >= 6200) & (wavelength <= 6800)).astype(float)))
    half = wavelength[wavelength >= 5500.0]
    template_half = np.column_stack((half, np.full_like(half, 1e-12)))
    try:
        calibrated_template_magnitude(template_half, 10.0, ref_band, 3631.0, 0.0, vis_band, 3631.0)
    except ValueError as exc:
        assert "calibration" in str(exc) or "covers only" in str(exc)
    else:
        raise AssertionError("partially covered reference band must raise, not silently rescale")


def test_atmosphere_extends_edges_instead_of_going_opaque():
    from etc_physics import atmospheric_transmission
    import astropy.units as u
    curve = np.column_stack((np.linspace(4000.0, 7000.0, 50), np.full(50, 0.8)))
    wave = np.array([3500.0, 5500.0, 8000.0]) * u.AA
    transmission = atmospheric_transmission(wave, {"airmass": 1.0, "transmission_curve": curve})
    assert np.allclose(transmission, 0.8), transmission


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


def test_slit_spectrograph_resolving_power_lhires_like():
    """LHIRES III-like: 2400 l/mm Littrow, 200/200 mm, 25 um slit on a 2 m FL
    telescope -> R of order 2e4 at H-alpha, slit-limited."""
    from spectroscopy import slit_spectrograph_resolving_power
    result = slit_spectrograph_resolving_power(6563.0, 2400.0, 200.0, 200.0, 2000.0,
                                               2.58, 3.0)  # 25 um at 2 m FL = 2.58"
    assert 10000.0 < result["resolving_power"] < 30000.0, result
    assert result["limited_by"] == "slit"
    low_res = slit_spectrograph_resolving_power(6563.0, 300.0, 130.0, 130.0, 2000.0, 2.58, 3.0)
    assert low_res["resolving_power"] < result["resolving_power"]


def test_gain_table_loader():
    from detector import load_gain_table
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as stream:
        stream.write("# setting  e/ADU  RN  FWC\n0 3.6 3.5 50000\n100, 1.0, 1.5, 20000\n")
        path = Path(stream.name)
    try:
        rows = load_gain_table(path)
        assert len(rows) == 2 and rows[0]["gain_setting"] == 0.0
        assert rows[1] == {"gain_setting": 100.0, "gain_e_adu": 1.0,
                           "read_noise_e": 1.5, "full_well_e": 20000.0}
    finally:
        path.unlink(missing_ok=True)


def test_stack_planner_closed_form():
    from solvers import plan_stack, exposure_time_for_snr
    plan = plan_stack(100.0, 50.0, 20.0, 1.0, 5.0, 20.0, max_unsaturated_exptime_s=60.0)
    assert plan["limited_by"] == "saturation" and plan["sub_exposure_s"] == 60.0
    assert plan["achieved_snr"] >= 100.0
    stacked = plan["n_frames"] * plan["sub_exposure_s"]
    snr_check = 50.0 * stacked / np.sqrt((50.0 + 20.0 + 1.0) * stacked
                                         + plan["n_frames"] * 20.0 * 25.0)
    assert np.isclose(snr_check, plan["achieved_snr"])
    assert plan["read_noise_penalty_percent"] >= 0.0
    ideal = exposure_time_for_snr(100.0, 50.0, 20.0, 1.0, 5.0, 20.0)
    assert stacked >= ideal


def test_sigma_ew_column_present_and_scales_inversely_with_snr():
    wavelength = np.linspace(5000.0, 6000.0, 100)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-9)))
    band = np.column_stack((wavelength, np.ones_like(wavelength)))
    qe = np.array([[4500.0, 0.8], [7000.0, 0.8]])
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    detector = Detector(10.0, 2.0, 1e9, 32)
    atmosphere = {"airmass": 1.0, "seeing_arcsec": 1.0, "transmission_curve": None}
    sky = {"sky_mag": 20.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True}
    etc = SpectroscopyETC(telescope, detector, atmosphere, sky)
    short = etc.compute_spectroscopy(template, 1000.0, 1.0, 10.0, (5200.0, 5800.0), 10.0, qe, band, visual_band=band)
    long = etc.compute_spectroscopy(template, 1000.0, 1.0, 40.0, (5200.0, 5800.0), 10.0, qe, band, visual_band=band)
    assert "sigma_ew_mangstrom" in short.columns
    assert float(long["sigma_ew_mangstrom"].median()) < float(short["sigma_ew_mangstrom"].median())


def test_target_snr_solver_includes_scintillation():
    """The solved exposure must reproduce the requested S/N even when
    scintillation dominates (bright star): its variance is linear in t and
    enters the closed form as an extra rate."""
    from photometry import PhotometryETC
    from observing_conditions import scintillation_variance_rate_e2_s
    from solvers import exposure_time_for_snr
    wavelength = np.linspace(4000.0, 7000.0, 300)
    template = np.column_stack((wavelength, np.full_like(wavelength, 3.6e-9)))
    band = np.column_stack((wavelength, ((wavelength >= 5000) & (wavelength <= 6000)).astype(float)))
    qe = np.column_stack((wavelength, np.full_like(wavelength, 0.9)))
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    detector = Detector(4.63, 1.0, 1e9, 32, read_noise_e=1.5)
    atmosphere = {"airmass": 1.3, "seeing_arcsec": 2.5, "elevation_m": 1366.0, "transmission_curve": None}
    sky = {"sky_mag": 20.5, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True,
           "aperture_radius_arcsec": 4.0}
    etc = PhotometryETC(telescope, detector, atmosphere, sky)
    probe = etc.compute_photometry_single(template, band, qe, 6.0, 1.0)
    scint_rate = scintillation_variance_rate_e2_s(probe["source_rate_per_s"], 358.0, 1.3, 1366.0)
    target = 500.0
    t_exp = exposure_time_for_snr(target, probe["source_rate_per_s"], probe["sky_rate_per_s"],
                                  detector.dark_current_e_s_pix * probe["n_pixels"],
                                  detector.read_noise_e, probe["n_pixels"],
                                  extra_variance_rate_e2_s=scint_rate)
    achieved = etc.compute_photometry_single(template, band, qe, 6.0, t_exp)["snr"]
    assert abs(achieved / target - 1.0) < 0.02, (t_exp, achieved)
    # Without the scintillation term the solver would badly undershoot.
    naive = exposure_time_for_snr(target, probe["source_rate_per_s"], probe["sky_rate_per_s"],
                                  detector.dark_current_e_s_pix * probe["n_pixels"],
                                  detector.read_noise_e, probe["n_pixels"])
    assert t_exp > 2.0 * naive


def test_filter_profile_builder_matches_svo_scale():
    """Synthetic Vega zero point of the shipped Bessell.V profile must land
    within ~1% of the SVO-declared value."""
    from filter_catalog import load_filter_profile
    from make_filter_profile import synthetic_vega_zero_point_jy
    profile = load_filter_profile(Path(__file__).resolve().parent.parent / "data", "Bessell.V", "Vega")
    zp = synthetic_vega_zero_point_jy(profile.transmission[:, 0], profile.transmission[:, 1])
    assert abs(zp / profile.zero_point_jy - 1.0) < 0.02, zp


if __name__ == "__main__":
    test_explicit_nm_qe_conversion()
    test_svo_energy_and_photon_semantics_differ_for_coloured_sed()
    test_spectroscopy_zero_fills_template_outside_coverage()
    test_calibration_band_coverage_is_still_strict()
    test_atmosphere_extends_edges_instead_of_going_opaque()
    test_slit_spectrograph_resolving_power_lhires_like()
    test_gain_table_loader()
    test_stack_planner_closed_form()
    test_sigma_ew_column_present_and_scales_inversely_with_snr()
    test_target_snr_solver_includes_scintillation()
    test_filter_profile_builder_matches_svo_scale()
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
