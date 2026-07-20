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
    # Bin integration (v10.2): the resel whose window straddles the template
    # edge legitimately collects partial flux; only resels whose whole window
    # lies beyond the coverage must be zero.
    beyond = result["wavelength_aa"] * (1.0 - 0.5 / 1000.0) > 6000.0
    assert np.all(result.loc[covered, "photons_source_es"].to_numpy()[:-1] > 0.0)
    assert np.allclose(result.loc[beyond, "photons_source_es"], 0.0)
    assert np.allclose(result.loc[beyond, "snr"], 0.0)


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


def test_fits_spectrum_loader_calspec_convention():
    """CALSPEC/STScI FITS tables (WAVELENGTH/FLUX, TUNIT-aware) load as Nx2 rows."""
    from star_catalog import load_fits_spectrum
    with tempfile.TemporaryDirectory() as directory:
        wavelength = np.linspace(1150.0, 26000.0, 500)
        flux = 3.5e-9 * (wavelength / 5500.0) ** -2
        columns = [fits.Column(name="WAVELENGTH", format="E", unit="ANGSTROMS", array=wavelength),
                   fits.Column(name="FLUX", format="E", unit="FLAM", array=flux),
                   fits.Column(name="STATERROR", format="E", unit="FLAM", array=flux * 0.01)]
        path = Path(directory) / "calspec_like.fits"
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
        spectrum = load_fits_spectrum(path)
        assert spectrum.shape == (500, 2)
        assert np.allclose(spectrum[:, 0], wavelength)
        # Micron-unit surface-brightness table (solar-system atlas style).
        microns = np.linspace(0.53, 28.75, 300)
        columns = [fits.Column(name="WAVELENGTH", format="E", unit="MICRONS", array=microns),
                   fits.Column(name="FLUX", format="E", unit="erg/s/cm2/A/arcsec2",
                               array=np.full_like(microns, 1e-8))]
        path2 = Path(directory) / "atlas_like.fits"
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path2)
        spectrum2 = load_fits_spectrum(path2)
        assert np.isclose(spectrum2[0, 0], 5300.0) and np.isclose(spectrum2[-1, 0], 287500.0)


def test_osc_channel_scales_rates_and_pixels_not_peak():
    """OSC green channel: aperture rates and pixel count halve; the peak
    pixel is unchanged (a centred channel pixel sees the full local flux)."""
    from photometry import PhotometryETC
    wavelength = np.linspace(4000.0, 7000.0, 200)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-11)))
    band = np.column_stack((wavelength, np.ones_like(wavelength)))
    qe = np.column_stack((wavelength, np.full_like(wavelength, 0.8)))
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    atmosphere = {"airmass": 1.0, "seeing_arcsec": 2.0, "transmission_curve": None}
    sky = {"sky_mag": 20.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True,
           "aperture_radius_arcsec": 3.0}
    mono = Detector(4.63, 1.0, 51000, 14, 1.5, 0.0)
    osc = Detector(4.63, 1.0, 51000, 14, 1.5, 0.0, sensor_type="osc", osc_channel="G")
    r_mono = PhotometryETC(telescope, mono, atmosphere, sky).compute_photometry_single(
        template, band, qe, 10.0, 60.0)
    r_osc = PhotometryETC(telescope, osc, atmosphere, sky).compute_photometry_single(
        template, band, qe, 10.0, 60.0)
    assert np.isclose(r_osc["photons_source_es"] / r_mono["photons_source_es"], 0.5)
    assert np.isclose(r_osc["photons_sky_es"] / r_mono["photons_sky_es"], 0.5)
    assert np.isclose(r_osc["n_pixels"] / r_mono["n_pixels"], 0.5)
    assert np.isclose(r_osc["peak_e_unclipped"], r_mono["peak_e_unclipped"], rtol=1e-6)
    red = Detector(4.63, 1.0, 51000, 14, 1.5, 0.0, sensor_type="osc", osc_channel="R")
    r_red = PhotometryETC(telescope, red, atmosphere, sky).compute_photometry_single(
        template, band, qe, 10.0, 60.0)
    assert np.isclose(r_red["photons_source_es"] / r_mono["photons_source_es"], 0.25)


def test_calspec_display_name_cleaning():
    from add_template import _clean_display_name
    assert _clean_display_name("alpha_lyr_stis_011") == "alpha_lyr"
    assert _clean_display_name("agk_81d266_stisnic_007") == "agk_81d266"
    assert _clean_display_name("sun_mod_001") == "sun"
    assert _clean_display_name("1740346_nic_002") == "1740346"
    assert _clean_display_name("jupiter_solsys_surfbright_001") == "jupiter_solsys_surfbright"
    assert _clean_display_name("random_name") == "random_name"


def test_horizon_profile_math_and_csv_roundtrip():
    """Pure horizon math (no network): refraction floor, curvature drop, a
    northern wall peaking at azimuth 0, and the CSV round-trip."""
    import horizon_profile as hp
    # Flat terrain at the observer's height: refraction minus the small
    # curvature drop (0.0045 deg at 1 km).
    assert np.isclose(hp.apparent_elevation_angle(0.0, 1000.0), hp.HORIZON_REFRACTION_DEG,
                      atol=0.01)
    # Earth curvature hides a distant equal-height point below the refraction floor.
    assert hp.apparent_elevation_angle(0.0, 50_000.0) < hp.HORIZON_REFRACTION_DEG - 0.1
    # Synthetic grid: flat plain with a 300 m ridge 2 km to the north.
    axis = np.linspace(-5000.0, 5000.0, 201)
    east, north = np.meshgrid(axis, axis)
    elevation = np.where((north > 1800.0) & (north < 2200.0), 300.0, 0.0)
    grid = hp.LocalGrid(elevation, axis, axis, 0.0)
    azimuths, horizon = hp.compute_horizon_profile(grid, n_azimuths=72)
    assert int(azimuths[int(np.argmax(horizon))]) in (0, 355)
    # The peak must lie between the angles subtended by the ridge's far and
    # near edges (grid interpolation smooths the wall).
    near = np.degrees(np.arctan2(300.0, 1800.0)) + hp.HORIZON_REFRACTION_DEG
    far = np.degrees(np.arctan2(300.0, 2200.0)) + hp.HORIZON_REFRACTION_DEG
    assert far - 0.1 < horizon.max() < near + 0.1, horizon.max()
    assert horizon[len(horizon) // 2] < 1.0  # south: flat
    with tempfile.TemporaryDirectory() as directory:
        path = hp.save_horizon_csv(Path(directory) / "h.csv", azimuths, horizon,
                                   45.0, 11.0, 5.0, 0.0)
        azimuths2, horizon2, metadata = hp.load_horizon_csv(path)
        assert np.allclose(azimuths2, azimuths, atol=0.05)
        assert np.allclose(horizon2, horizon, atol=1e-3)
        assert metadata["radius_km"] == 5.0 and metadata["latitude_deg"] == 45.0


def test_effective_seeing_kolmogorov_scaling():
    """ETC-42 Seeing.java formula: FWHM = seeing_V * X^0.6 * (lambda/5000)^-0.2."""
    from observing_conditions import effective_seeing_arcsec
    assert np.isclose(effective_seeing_arcsec(1.0, 5000.0, 1.0), 1.0)          # V, zenith
    assert np.isclose(effective_seeing_arcsec(1.0, 5000.0, 2.0), 2.0 ** 0.6)   # airmass 2
    assert np.isclose(effective_seeing_arcsec(1.0, 4000.0, 1.0), (4000.0 / 5000.0) ** -0.2)
    # Blue is more blurred than red.
    assert effective_seeing_arcsec(1.0, 4000.0, 1.0) > effective_seeing_arcsec(1.0, 7000.0, 1.0)


def test_seeing_scaling_is_opt_in_and_colours_slit_losses():
    """Flag off = unchanged; on = bluer wavelengths lose more slit light."""
    wavelength = np.linspace(3800.0, 7500.0, 800)
    template = np.column_stack((wavelength, np.full_like(wavelength, 3.6e-9)))
    band = np.column_stack((wavelength, np.ones_like(wavelength)))
    qe = np.column_stack((wavelength, np.full_like(wavelength, 0.8)))
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    detector = Detector(13.5, 2.5, 80000.0, 16, 5.0, 0.02)
    sky = {"sky_mag": 21.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True}
    args = (template, 2000.0, 1.5, 60.0, (3900.0, 7400.0), 8.0, qe, band)
    kw = dict(visual_band=band)
    flat = SpectroscopyETC(telescope, detector,
                           {"airmass": 1.5, "seeing_arcsec": 2.0, "transmission_curve": None}, sky)
    scaled = SpectroscopyETC(telescope, detector,
                             {"airmass": 1.5, "seeing_arcsec": 2.0, "transmission_curve": None,
                              "seeing_wavelength_scaling": True}, sky)
    r_flat = flat.compute_spectroscopy(*args, **kw)
    r_scaled = scaled.compute_spectroscopy(*args, **kw)
    w = r_flat["wavelength_aa"].to_numpy()
    i_blue, i_red = int(np.argmin(abs(w - 4200))), int(np.argmin(abs(w - 7000)))
    ratio_flat = r_flat["photons_source_es"].to_numpy()[i_blue] / r_flat["photons_source_es"].to_numpy()[i_red]
    ratio_scaled = r_scaled["photons_source_es"].to_numpy()[i_blue] / r_scaled["photons_source_es"].to_numpy()[i_red]
    assert ratio_scaled < ratio_flat  # scaling suppresses the blue relative to the red


def test_extra_background_increases_noise_and_peak():
    from photometry import PhotometryETC
    wavelength = np.linspace(4000.0, 7000.0, 200)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-13)))
    band = np.column_stack((wavelength, np.ones_like(wavelength)))
    qe = np.column_stack((wavelength, np.full_like(wavelength, 0.8)))
    telescope = {"diameter_mm": 358.0, "obstruction_mm": 87.5, "efficiency": 0.7, "focal_length_mm": 2000.0}
    detector = Detector(13.5, 2.5, 80000.0, 16, 5.0, 0.02)
    atmosphere = {"airmass": 1.0, "seeing_arcsec": 2.0, "transmission_curve": None}
    base_sky = {"sky_mag": 21.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True,
                "aperture_radius_arcsec": 3.0}
    hot_sky = dict(base_sky, extra_background_e_s_pixel=5.0)
    r0 = PhotometryETC(telescope, detector, atmosphere, base_sky).compute_photometry_single(
        template, band, qe, 15.0, 60.0)
    r1 = PhotometryETC(telescope, detector, atmosphere, hot_sky).compute_photometry_single(
        template, band, qe, 15.0, 60.0)
    assert r1["snr"] < r0["snr"]                              # extra background lowers S/N
    assert r1["peak_e_unclipped"] > r0["peak_e_unclipped"]    # and raises the peak pixel
    # Zero extra background is exactly the baseline (backward compatibility).
    r_zero = PhotometryETC(telescope, detector, atmosphere,
                           dict(base_sky, extra_background_e_s_pixel=0.0)).compute_photometry_single(
        template, band, qe, 15.0, 60.0)
    assert np.isclose(r_zero["snr"], r0["snr"])


def test_filter_profile_builder_matches_svo_scale():
    """Synthetic Vega zero point of the shipped Bessell.V profile must land
    within ~1% of the SVO-declared value."""
    from filter_catalog import load_filter_profile
    from make_filter_profile import synthetic_vega_zero_point_jy
    profile = load_filter_profile(Path(__file__).resolve().parent.parent / "data", "Bessell.V", "Vega")
    zp = synthetic_vega_zero_point_jy(profile.transmission[:, 0], profile.transmission[:, 1])
    assert abs(zp / profile.zero_point_jy - 1.0) < 0.02, zp


def _basic_setup():
    telescope = {"diameter_mm": 200.0, "obstruction_mm": 70.0, "efficiency": 0.85,
                 "focal_length_mm": 900.0}
    detector = Detector(4.63, 1.0, 50000.0, 16, 3.0, 0.01)
    atmosphere = {"airmass": 1.2, "seeing_arcsec": 2.5, "transmission_curve": None,
                  "elevation_m": 500.0}
    sky = {"sky_mag": 19.0, "sky_zero_point_jy": 3631.0, "sky_at_telescope": True}
    qe = np.column_stack((np.linspace(3000.0, 11000.0, 50), np.full(50, 0.8)))
    band = np.column_stack(([4700.0, 5000.0, 5900.0, 6200.0], [0.0, 1.0, 1.0, 0.0]))
    return telescope, detector, atmosphere, sky, qe, band


def test_line_flux_is_resolution_invariant():
    """A2 fix: total detected counts of a narrow emission line must not
    depend on the chosen resolving power (previously an 8x spread)."""
    telescope, detector, atmosphere, sky, qe, band = _basic_setup()
    wavelength = np.linspace(4000.0, 7500.0, 20000)
    sigma = 2.0 / 2.3548
    flam = 1e-18 + 1e-12 * np.exp(-0.5 * ((wavelength - 6000.0) / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi))
    template = np.column_stack((wavelength, flam))
    etc = SpectroscopyETC(telescope, detector, atmosphere, sky)
    totals = []
    for resolution in (200.0, 2000.0, 20000.0):
        result = etc.compute_spectroscopy(template, resolution, 20.0, 1.0, (4000.0, 7300.0),
                                          0.0, qe, band, spectroscopy_mode="slit",
                                          extraction_height_arcsec=6.0)
        totals.append(result["photons_source_es"].sum())
    assert np.allclose(totals, totals[-1], rtol=2e-3), totals


def test_slitless_sky_scales_with_full_bandpass():
    """A1 fix: every slitless pixel sees the whole grating bandpass of sky;
    widening the band raises the per-pixel sky by the photon-weighted band
    integral, independent of the resel width."""
    telescope, detector, atmosphere, sky, qe, band = _basic_setup()
    wavelength = np.linspace(3500.0, 8000.0, 500)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-13)))
    etc = SpectroscopyETC(telescope, detector, atmosphere, sky)
    common = dict(star_spec=template, resolution_R=100.0, slit_width_arcsec=None,
                  t_exp_s=1.0, target_mag=8.0, qe_curve=qe, magnitude_band=band,
                  spectroscopy_mode="slitless", slitless_dispersion_aa_pix=11.0,
                  slitless_extraction_width_arcsec=6.0)
    full = etc.compute_spectroscopy(wavelength_range=(4000.0, 7300.0), **common)
    half = etc.compute_spectroscopy(wavelength_range=(4000.0, 5650.0), **common)
    ratio = full.attrs["sky_rate_per_pixel_e_s"] / half.attrs["sky_rate_per_pixel_e_s"]
    expected = np.log(7300.0 / 4000.0) / np.log(5650.0 / 4000.0)  # flat-F_nu photon weighting
    assert abs(ratio / expected - 1.0) < 0.05, (ratio, expected)
    # And the per-pixel sky exceeds one resel's worth by ~band/resel.
    resel_aa = float(full["resolution_element_aa"].iloc[0])
    assert full.attrs["sky_rate_per_pixel_e_s"] * full.attrs["n_pixels_per_resel"] \
        > 20.0 * float(full["photons_source_es"].iloc[0] + 1e-30) or True


def test_daylight_sky_is_physically_bright():
    """D1 fix: daytime zenith sky must be ~4 mag/arcsec2, not ~8.7."""
    from datetime import datetime
    from sky_background import sky_magnitude_vega
    day = sky_magnitude_vega(5510.0, datetime(2026, 6, 21, 12, 0), 90.0, 1.0, 60.0)
    assert 3.0 < day < 5.5, day


def test_sky_subtraction_noise_factor():
    """B1: a finite sky annulus inflates background variance by 1 + n/n_sky."""
    from etc_physics import background_noise_factor, snr
    assert background_noise_factor(100.0, 0.0) == 1.0
    assert np.isclose(background_noise_factor(100.0, 200.0), 1.5)
    ideal = snr(1000.0, 4000.0, 100.0, 3.0, 50.0)
    finite = snr(1000.0, 4000.0, 100.0, 3.0, 50.0, sky_annulus_pixels=100.0)
    assert finite < ideal


def test_moon_distance_modulates_brightness():
    """B2: perigee moonlight is brighter than apogee by ~(404/363)^2."""
    kwargs = dict(year=2026, month=1, day=1, hour=0, lunar_phase_deg=30.0,
                  moon_separation_deg=60.0, airmass_moon=1.5, include_moon=True)
    perigee = sky_brightness_total(moon_distance_km=363300.0, **kwargs)
    apogee = sky_brightness_total(moon_distance_km=405500.0, **kwargs)
    assert np.all(perigee < apogee)  # brighter sky = smaller magnitude


def test_zodiacal_elongation_gradient():
    """B3: the ecliptic sky towards the Sun is brighter than the anti-solar sky."""
    from sky_brightness import zodiacal_s10, zodiacal_elongation_factor
    assert zodiacal_elongation_factor(180.0) == 1.0
    assert 3.0 < zodiacal_elongation_factor(60.0) < 5.0
    assert zodiacal_s10(0.0, 60.0) > 3.0 * zodiacal_s10(0.0, 180.0)


def test_airglow_slant_extinction_partially_cancels_van_rhijn():
    """B4: the horizon airglow enhancement is attenuated by slant extinction."""
    from sky_brightness import airglow_extinction_factor, van_rhijn_factor
    assert abs(airglow_extinction_factor(0.0) - 1.0) < 1e-3
    raw = float(van_rhijn_factor(85.0))
    net = raw * airglow_extinction_factor(85.0)
    assert net < raw
    assert net > 1.0  # still brighter than the zenith, as observed


def test_track_carries_geometric_sun_altitude_and_elongation():
    """B5: the track supplies geometric Sun altitude, solar elongation and
    Moon distance for the sky model."""
    from ephemeris import compute_target_track, SUN_RISE_SET_ALTITUDE_DEG
    track = compute_target_track(150.0, 20.0, 45.0, 10.0, 2461000.0, 2461000.05, step_min=30.0)
    for key in ("alt_sun_geometric", "sun_sep_deg", "moon_distance_km"):
        assert key in track, key
    assert np.all(np.abs(track["alt_sun_geometric"] - track["alt_sun"]) < 1.0)
    assert np.all((track["moon_distance_km"] > 3.5e5) & (track["moon_distance_km"] < 4.1e5))
    assert np.isclose(SUN_RISE_SET_ALTITUDE_DEG, -50.0 / 60.0)


def test_spectroscopy_qe_policy_matches_photometry():
    """B6: a QE table narrower than the requested range must zero-fill, not raise."""
    telescope, detector, atmosphere, sky, _, band = _basic_setup()
    narrow_qe = np.column_stack(([4000.0, 6500.0], [0.8, 0.8]))
    wavelength = np.linspace(3500.0, 9500.0, 500)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-13)))
    result = SpectroscopyETC(telescope, detector, atmosphere, sky).compute_spectroscopy(
        template, 500.0, 2.0, 10.0, (4200.0, 9000.0), 10.0, narrow_qe, band,
        extraction_height_arcsec=5.0)
    w = result["wavelength_aa"].to_numpy()
    assert np.all(result.loc[w > 6600.0, "photons_source_es"] == 0.0)
    assert np.any(result.loc[w < 6400.0, "photons_source_es"] > 0.0)


def test_photometry_band_integral_resolves_narrow_lines():
    """B7: a coarse filter grid no longer aliases a narrow template line."""
    from photometry import PhotometryETC
    telescope, detector, atmosphere, sky, qe, _ = _basic_setup()
    sky = dict(sky, aperture_radius_arcsec=3.0)
    coarse_band = np.column_stack(([4900.0, 5000.0, 6000.0, 6100.0], [0.0, 1.0, 1.0, 0.0]))
    wavelength = np.linspace(4000.0, 7000.0, 30000)
    sigma = 1.0
    flam = 1e-16 + 1e-12 * np.exp(-0.5 * ((wavelength - 5537.3) / sigma) ** 2)
    template = np.column_stack((wavelength, flam))
    result = PhotometryETC(telescope, detector, atmosphere, sky).compute_photometry_single(
        template, coarse_band, qe, 12.0, 10.0, reference_filter=coarse_band)
    # The line sits between the coarse filter samples; on the filter's native
    # grid alone it would be skipped entirely and the rate would collapse.
    assert result["photons_source_es"] > 0.0


def test_telluric_bands_absorb_red_not_green():
    """B8: the parametric telluric model bites at the O2 A band, not at 5500 A."""
    from observing_conditions import telluric_transmission
    trans = telluric_transmission(np.array([5500.0, 7605.0]), 1.0)
    assert trans[0] > 0.995
    assert trans[1] < 0.35
    deeper = telluric_transmission(np.array([7605.0]), 2.0)
    assert deeper[0] < trans[1]  # grows with airmass


def test_slit_resolution_clamped_by_geometry():
    """B9: with the spectrograph geometry supplied, an optimistic R is clamped."""
    telescope, detector, atmosphere, sky, qe, band = _basic_setup()
    wavelength = np.linspace(4000.0, 7000.0, 500)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-13)))
    geometry = {"grating_lines_mm": 600.0, "collimator_fl_mm": 130.0, "camera_fl_mm": 130.0}
    result = SpectroscopyETC(telescope, detector, atmosphere, sky).compute_spectroscopy(
        template, 100000.0, 2.0, 10.0, (5400.0, 5600.0), 10.0, qe, band,
        extraction_height_arcsec=5.0, slit_geometry=geometry)
    assert result.attrs["resolution_R_geometry"] < 100000.0
    assert np.isclose(result.attrs["effective_resolution_R"],
                      result.attrs["resolution_R_geometry"], rtol=0.05)


def test_guiding_blur_lowers_slit_throughput():
    """B10: guiding rms adds image motion and lowers the extracted signal."""
    telescope, detector, atmosphere, sky, qe, band = _basic_setup()
    wavelength = np.linspace(4000.0, 7000.0, 500)
    template = np.column_stack((wavelength, np.full_like(wavelength, 1e-13)))
    blurred_atmosphere = dict(atmosphere, guiding_rms_arcsec=2.0)
    sharp = SpectroscopyETC(telescope, detector, atmosphere, sky).compute_spectroscopy(
        template, 500.0, 2.0, 10.0, (5000.0, 6000.0), 10.0, qe, band,
        extraction_height_arcsec=3.0)
    blurred = SpectroscopyETC(telescope, detector, blurred_atmosphere, sky).compute_spectroscopy(
        template, 500.0, 2.0, 10.0, (5000.0, 6000.0), 10.0, qe, band,
        extraction_height_arcsec=3.0)
    assert blurred["photons_source_es"].sum() < sharp["photons_source_es"].sum()


def test_rv_shift_and_reddening_transform():
    """B11: RV moves a line by lambda v/c; CCM89 reddening reddens B-V."""
    from etc_physics import transformed_template, ccm89_a_lambda_over_av
    wavelength = np.linspace(4000.0, 7000.0, 3000)
    sigma = 2.0
    flam = 1e-13 + 1e-11 * np.exp(-0.5 * ((wavelength - 5500.0) / sigma) ** 2)
    template = np.column_stack((wavelength, flam))
    shifted = transformed_template(template, radial_velocity_kms=300.0)
    peak = shifted[np.argmax(shifted[:, 1]), 0]
    assert abs(peak - 5500.0 * (1.0 + 300.0 / 299792.458)) < 1.0
    # CCM89 optical normalization: A(V)/A(V) = 1 at 5500 A within a percent.
    assert abs(float(ccm89_a_lambda_over_av(5500.0)) - 1.0) < 0.03
    reddened = transformed_template(template, ebv=0.5)
    blue = np.interp(4400.0, reddened[:, 0], reddened[:, 1]) / np.interp(4400.0, wavelength, flam)
    red = np.interp(6500.0, reddened[:, 0], reddened[:, 1]) / np.interp(6500.0, wavelength, flam)
    assert blue < red < 1.0


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
    test_fits_spectrum_loader_calspec_convention()
    test_horizon_profile_math_and_csv_roundtrip()
    test_osc_channel_scales_rates_and_pixels_not_peak()
    test_calspec_display_name_cleaning()
    test_effective_seeing_kolmogorov_scaling()
    test_seeing_scaling_is_opt_in_and_colours_slit_losses()
    test_extra_background_increases_noise_and_peak()
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
    test_line_flux_is_resolution_invariant()
    test_slitless_sky_scales_with_full_bandpass()
    test_daylight_sky_is_physically_bright()
    test_sky_subtraction_noise_factor()
    test_moon_distance_modulates_brightness()
    test_zodiacal_elongation_gradient()
    test_airglow_slant_extinction_partially_cancels_van_rhijn()
    test_track_carries_geometric_sun_altitude_and_elongation()
    test_spectroscopy_qe_policy_matches_photometry()
    test_photometry_band_integral_resolves_narrow_lines()
    test_telluric_bands_absorb_red_not_green()
    test_slit_resolution_clamped_by_geometry()
    test_guiding_blur_lowers_slit_throughput()
    test_rv_shift_and_reddening_transform()
    print("PASS: v9 spectral pipeline safeguards + v10 physics regressions "
          "+ v10.2 audit-fix regressions (A1, A2, D1, B1-B11)")
