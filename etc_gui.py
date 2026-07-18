#!/usr/bin/env python3
"""Generic astronomical ETC: one manually selected observing time per run."""

import json
import sys
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astroquery.simbad import Simbad

from ephemeris import compute_target_track
from photometry import PhotometryETC
from spectroscopy import SpectroscopyETC
from detector import Detector, load_qe_curve, load_transmission_curve
from spectral_utils import load_fits_transmission_curve, interpolate_zero_filled
from solvers import exposure_time_for_snr
from config_manager import OBSERVATORY_PRESETS
import filter_catalog as fcat
import star_catalog as scat
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib_plots import show_photometry_plot, show_spectroscopy_plot
from sky_background import sky_magnitude_vega
from sky_brightness import sky_brightness_total, BAND_WAVELENGTH_NM, BAND_VEGA_ZEROPOINT_JY
from etc_physics import synthetic_magnitude, magnitude_f_lambda


CONFIG_FILE = Path(__file__).with_name("etc_user_config.json")
SITES_FILE = Path(__file__).with_name("etc_sites.json")
PROFILE_SCHEMA_VERSION = 1


class ETCGUI(tk.Tk):
    """The instrument configuration is automatically read/written as JSON."""

    def __init__(self):
        super().__init__()
        self.title("SPETC v10.0 - Spectro-Photometry Exposure Time Calculator")
        self.geometry("1850x980")
        self.minsize(1200, 720)
        self._saved = self._load_config()
        self._vars = {}
        self.active_profile_path = self._saved.get("active_instrument_profile", "")
        self._site_records = self._load_site_records()
        self.star_catalog = []
        self.star_spec = None
        self.selected_star = None
        self.star_id_map = {}
        self.filter_resp_data = None
        self.qe_curve = None
        self.qe_source_path = None
        self.throughput_curve = None
        self.throughput_source_path = None
        self.slit_resolution_curve = None
        self.slit_resolution_source_path = None
        self.result_df = None
        self.time_series_df = None
        self.plot_window = None
        self.earth_atmosphere_curve = None
        self.results_window = None
        self.tree = None
        self.time_tree = None
        self.info_text = None
        self._build_ui()
        self.after(0, self._update_observatory_status)
        self._reload_catalog()
        if self.active_profile_path:
            self._load_instrument_profile_path(Path(self.active_profile_path), report_errors=False)
        self.protocol("WM_DELETE_WINDOW", self._on_exit)
        self.after(100, self._save_config)

    def _load_config(self):
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _var(self, key, default):
        value = str(self._saved.get(key, default))
        var = tk.StringVar(value=value)
        self._vars[key] = var
        return var

    def _save_config(self):
        data = {key: var.get() for key, var in self._vars.items()}
        data["schema_version"] = 9
        data["active_instrument_profile"] = str(self.active_profile_path)
        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
        except OSError as exc:
            self.status_var.set(f"Configuration not saved: {exc}")

    def _build_ui(self):
        """Five operational columns; numerical results live in a Toplevel."""
        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=5, pady=5)
        col1 = self._new_scroll_column(panes, 355)
        col2 = self._new_scroll_column(panes, 345)
        col3 = self._new_scroll_column(panes, 305)
        col4 = self._new_scroll_column(panes, 355)
        col5 = self._new_scroll_column(panes, 220)
        self._build_observation_column(col1)
        self._build_instrument_column(col2)
        self._build_mode_column(col3)
        self._build_data_column(col4)
        self._build_actions_column(col5)
        self.status_var = tk.StringVar(value="Ready")
        #ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", padx=5, pady=(0, 5))
        self.status_label = tk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            bg="#e6e6e6",
        )
        self.status_label.pack(fill="x", padx=5, pady=(0, 5))
        
        self.status_var.trace_add("write", lambda *_: self._update_status_colour())
    
    def _update_status_colour(self):
        text = self.status_var.get().strip().lower()
    
        if text.startswith("calculating"):
            colour = "#00d9e8"      # cyan
        elif text.startswith("error"):
            colour = "#e53935"      # red
        else:
            colour = "#e6e6e6"      # default grey
    
        self.status_label.configure(bg=colour)

    def _new_scroll_column(self, panes, width):
        """Create one independently scrollable column in the main window."""
        outer = ttk.Frame(panes, width=width)
        panes.add(outer, weight=1)
        canvas = tk.Canvas(outer, highlightthickness=0, width=width)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        holder = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=holder, anchor="nw")
        holder.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return holder

    def _section(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill="x", padx=5, pady=4)
        frame.columnconfigure(1, weight=1)
        return frame

    @staticmethod
    def _entry(frame, row, text, variable):
        ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
        ttk.Entry(frame, textvariable=variable, width=20).grid(row=row, column=1, sticky="ew", pady=1)

    def _build_observation_column(self, holder):
        f = self._section(holder, "OBSERVATION DATE")
        self.date_var = self._var("date", datetime.now().strftime("%Y-%m-%d"))
        self.time_var = self._var("time", "22:00")
        self.time_ref_var = self._var("time_reference", "local")
        self.utc_offset_var = self._var("utc_offset_h", "1")
        self.timezone_var = self._var("timezone", "Europe/Rome")
        self.timezone_source_var = self._var("timezone_source", "iana")
        if "timezone_source" not in self._saved:
            self.timezone_source_var.set("iana" if self.timezone_var.get().strip() else "offset")
        self._entry(f, 0, "Date (YYYY-MM-DD):", self.date_var)
        self._entry(f, 1, "Time (HH:MM):", self.time_var)
        time_frame = ttk.Frame(f)
        time_frame.grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(time_frame, text="Local", variable=self.time_ref_var, value="local").pack(side="left")
        ttk.Radiobutton(time_frame, text="UT", variable=self.time_ref_var, value="UT").pack(side="left", padx=8)
        source_frame = ttk.Frame(f)
        source_frame.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(source_frame, text="Local-time source:").pack(side="left")
        ttk.Radiobutton(source_frame, text="IANA timezone", variable=self.timezone_source_var, value="iana",
                        command=self._update_timezone_source_state).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(source_frame, text="UTC offset", variable=self.timezone_source_var, value="offset",
                        command=self._update_timezone_source_state).pack(side="left", padx=(6, 0))
        ttk.Label(f, text="IANA timezone:").grid(row=4, column=0, sticky="w")
        self.timezone_combo = ttk.Combobox(f, textvariable=self.timezone_var, state="readonly", width=25,
                                           values=tuple(sorted(available_timezones())))
        self.timezone_combo.grid(row=4, column=1, sticky="ew", pady=1)
        ttk.Label(f, text="UTC offset (h):").grid(row=5, column=0, sticky="w")
        self.utc_offset_entry = ttk.Entry(f, textvariable=self.utc_offset_var, width=20)
        self.utc_offset_entry.grid(row=5, column=1, sticky="ew", pady=1)
        self._update_timezone_source_state()
        ttk.Label(f, text="Manual time slider:").grid(row=6, column=0, sticky="w")
        self.time_slider = ttk.Scale(f, from_=0, to=1439, orient="horizontal", command=self._slider_changed)
        self.time_slider.grid(row=6, column=1, sticky="ew")
        self._sync_slider_from_time()
        self.time_var.trace_add("write", lambda *_: self._sync_slider_from_time())

        f = self._section(holder, "OBSERVATORY")
        self.obs_var = self._var("observatory", "Asiago")
        self.lat_var = self._var("latitude_deg", "45.8486")
        self.lon_var = self._var("longitude_deg", "11.5696")
        self.elev_var = self._var("elevation_m", "1366")
        ttk.Label(f, text="Saved site:").grid(row=0, column=0, sticky="w")
        self.site_combo = ttk.Combobox(f, textvariable=self.obs_var, state="readonly", values=self._site_names())
        self.site_combo.grid(row=0, column=1, sticky="ew")
        self.site_combo.bind("<<ComboboxSelected>>", self._on_obs_changed)
        self._entry(f, 1, "Latitude (deg):", self.lat_var)
        self._entry(f, 2, "Longitude, East + (deg):", self.lon_var)
        self._entry(f, 3, "Elevation (m):", self.elev_var)
        ttk.Button(f, text="Save current site…", command=self._save_current_site).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 1))
        ttk.Button(f, text="Import site file…", command=self._import_site_file).grid(row=5, column=0, columnspan=2, sticky="ew", pady=1)

        f = self._section(holder, "TARGET (ICRS/J2000)")
        self.target_name_var = self._var("target_name", "")
        self.ra_var = self._var("ra", "279.2347")
        self.dec_var = self._var("dec", "38.7837")
        self.coord_format_var = self._var("coordinate_format", "decimal")
        self.mag_var = self._var("target_magnitude", str(self._saved.get("target_magnitude_ab", "15.0")))
        self.mag_system_var = self._var("magnitude_system", "Vega")
        legacy_band = str(self._saved.get("magnitude_band", "Bessell.V"))
        self.reference_band_var = self._var("magnitude_reference_band", legacy_band)
        self.band_var = self._var("observing_band", legacy_band)
        self._entry(f, 0, "Target name (SIMBAD, optional):", self.target_name_var)
        ttk.Button(f, text="Resolve name", command=self._resolve_target_name).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(2, 1))
        self.target_resolution_status = tk.StringVar(value="Enter a name and press Resolve name to query SIMBAD.")
        ttk.Label(f, textvariable=self.target_resolution_status, foreground="gray35", wraplength=320,
                  justify="left").grid(row=2, column=0, columnspan=2, sticky="w")
        self._entry(f, 3, "RA (deg or hh:mm:ss):", self.ra_var)
        self._entry(f, 4, "Dec (deg or dd:mm:ss):", self.dec_var)
        fmt = ttk.Frame(f)
        fmt.grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(fmt, text="Decimal", variable=self.coord_format_var, value="decimal").pack(side="left")
        ttk.Radiobutton(fmt, text="Sexagesimal", variable=self.coord_format_var, value="sexagesimal").pack(side="left", padx=8)
        self._entry(f, 6, "Magnitude:", self.mag_var)
        magsys = ttk.Frame(f)
        magsys.grid(row=7, column=0, columnspan=2, sticky="w")
        ttk.Label(magsys, text="Magnitude system:").pack(side="left")
        ttk.Radiobutton(magsys, text="Vega (default)", variable=self.mag_system_var, value="Vega").pack(side="left", padx=(6, 0))
        ttk.Radiobutton(magsys, text="AB", variable=self.mag_system_var, value="AB").pack(side="left", padx=(6, 0))
        ttk.Label(f, text="Reference magnitude filter:").grid(row=8, column=0, sticky="w")
        self.reference_filter_combo = ttk.Combobox(f, textvariable=self.reference_band_var, state="readonly", values=())
        self.reference_filter_combo.grid(row=8, column=1, sticky="ew")
        #self.reference_filter_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_template_display())
        self.reference_filter_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: (self._update_template_display(),
                            self._update_combined_response_preview()),
        )
        ttk.Label(f, text="The observing filter is independent.", foreground="gray35", wraplength=320, justify="left").grid(row=9, column=0, columnspan=2, sticky="w", pady=(4, 0))
#        ttk.Label(f, text="The template flux is scaled from its stored Vmag to this measurement. The observing filter is independent.", foreground="gray35", wraplength=320, justify="left").grid(row=9, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_instrument_column(self, holder):
        f = self._section(holder, "TELESCOPE AND DETECTOR")
        self.diam_var = self._var("diameter_mm", "358")
        self.obstruct_var = self._var("obstruction_mm", "87.5")
        self.eff_var = self._var("throughput_excluding_qe", "0.70")
        self.focal_var = self._var("focal_length_mm", "2000")
        self.pixel_var = self._var("pixel_size_um", "13.5")
        self.gain_var = self._var("gain_e_adu", "2.5")
        self.fullwell_var = self._var("full_well_e", "80000")
        self.bitdepth_var = self._var("bit_depth", "16")
        self.readnoise_var = self._var("read_noise_e", "5.0")
        self.dark_var = self._var("dark_current_e_s_pix", "0.0")
        for row, label, var in [(0, "Primary diameter (mm):", self.diam_var), (1, "Obstruction diameter (mm):", self.obstruct_var),
                                (2, "Optics throughput, no QE:", self.eff_var), (3, "Focal length (mm):", self.focal_var),
                                (4, "Pixel size (um):", self.pixel_var), (5, "Gain (e-/ADU):", self.gain_var),
                                (6, "Full well (e-):", self.fullwell_var), (7, "ADC bits:", self.bitdepth_var),
                                (8, "Read noise (e- rms/pix):", self.readnoise_var), (9, "Dark current (e-/s/pix):", self.dark_var)]:
            self._entry(f, row, label, var)
        self.throughput_path_var = self._var("throughput_curve_path", "")
        self.throughput_unit_var = self._var("throughput_wavelength_unit", "Angstrom")
        self.qe_unit_var = self._var("qe_wavelength_unit", "Angstrom")
        ttk.Label(f, text="Calibrated optics response:").grid(row=10, column=0, sticky="w")
        curve_row = ttk.Frame(f); curve_row.grid(row=10, column=1, sticky="ew")
        curve_row.columnconfigure(0, weight=1)
        ttk.Entry(curve_row, textvariable=self.throughput_path_var, width=14).grid(row=0, column=0, sticky="ew")
        ttk.Button(curve_row, text="Browse…", command=self._choose_throughput_curve).grid(row=0, column=1, padx=(3, 0))
        ttk.Label(f, text="Response wavelength unit:").grid(row=11, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.throughput_unit_var, state="readonly", values=("Angstrom", "nm", "um"), width=10).grid(row=11, column=1, sticky="ew")
        self.slit_resolution_path_var = self._var("slit_resolution_curve_path", "")
        ttk.Label(f, text="Calibrated slit R (width):").grid(row=12, column=0, sticky="w")
        resolution_row = ttk.Frame(f); resolution_row.grid(row=12, column=1, sticky="ew")
        resolution_row.columnconfigure(0, weight=1)
        ttk.Entry(resolution_row, textvariable=self.slit_resolution_path_var, width=14).grid(row=0, column=0, sticky="ew")
        ttk.Button(resolution_row, text="Browse…", command=self._choose_slit_resolution_curve).grid(row=0, column=1, padx=(3, 0))
        ttk.Label(f, text="QE data wavelength unit:").grid(row=13, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.qe_unit_var, state="readonly", values=("Angstrom", "nm", "um"), width=10).grid(row=13, column=1, sticky="ew")
#        ttk.Label(f, text="Optics response: wavelength, throughput excluding QE. qe.dat is loaded separately.\n"
#                          "Slit R(width): slit width [arcsec], measured resolving power R.",
#                  foreground="gray35", wraplength=300, justify="left").grid(row=14, column=0, columnspan=2, sticky="w", pady=(3, 0))

        f = self._section(holder, "ATMOSPHERE / SKY")
        self.seeing_var = self._var("seeing_arcsec", "0.8")
        self.sky_var = self._var("sky_ab_mag_arcsec2", "21.7")
        self.sky_model_var = self._var("sky_model", "ing")
        # Keep existing local configurations usable after the visible model
        # name changed from its implementation provenance to ING.
        if self.sky_model_var.get() == "fortran":
            self.sky_model_var.set("ING")
        self._entry(f, 0, "Seeing FWHM (arcsec):", self.seeing_var)
        ttk.Label(f, text="Sky background:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.sky_model_var, state="readonly", values=("ing", "fixed_ab")).grid(row=1, column=1, sticky="ew")
        self._entry(f, 2, "Fixed sky (AB mag/arcsec2):", self.sky_var)
        self.psf_model_var = self._var("psf_model", "gaussian")
        self.moffat_beta_var = self._var("moffat_beta", "2.5")
        ttk.Label(f, text="PSF model:").grid(row=3, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.psf_model_var, state="readonly",
                     values=("gaussian", "moffat")).grid(row=3, column=1, sticky="ew")
        self._entry(f, 4, "Moffat beta (>1):", self.moffat_beta_var)
        ttk.Label(f, text="ING: pos-dep sky model; fixed_ab: observed sky. Moffat real seeing wings; 2.5<= beta <=4.7",
                  foreground="gray35", wraplength=300, justify="left").grid(row=5, column=0, columnspan=2, sticky="w")
#        ttk.Label(f, text="ING: position-dependent night/twilight/day model (van Rhijn airglow,\n"
#                          "zodiacal, starlight, Krisciunas-Schaefer Moon); fixed_ab: manual observed sky.\n"
#                          "Moffat reproduces real seeing wings; beta 2.5-4.7 (Gaussian limit).",
#                  foreground="gray35", wraplength=300, justify="left").grid(row=5, column=0, columnspan=2, sticky="w")

        self.response_preview_section = self._section(holder, "SYSTEM RESPONSE")
        self.response_preview_status = tk.StringVar(value="Select an observing filter and a stellar template.")
        ttk.Label(self.response_preview_section, textvariable=self.response_preview_status, foreground="blue",
                  wraplength=300, justify="left").grid(row=0, column=0, sticky="w")
        self.response_preview_figure = Figure(figsize=(3.2, 1.75), dpi=100)
        self.response_preview_axis = self.response_preview_figure.add_subplot(111)
        self.response_preview_canvas = FigureCanvasTkAgg(self.response_preview_figure, master=self.response_preview_section)
        self.response_preview_canvas.get_tk_widget().grid(row=1, column=0, sticky="ew", pady=(3, 0))
        self.response_preview_section.pack_forget()

        f = self._section(holder, "INSTRUMENT PROFILE")
        self.instrument_profile_section = f
        self.profile_status = tk.StringVar(value="Using current GUI values; QE loaded with data directory.")
        ttk.Label(f, textvariable=self.profile_status, foreground="blue", wraplength=300, justify="left").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(f, text="Save instrument profile…", command=self._save_instrument_profile).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 1))
        ttk.Button(f, text="Load instrument profile…", command=self._load_instrument_profile).grid(row=2, column=0, columnspan=2, sticky="ew", pady=1)
        self.eff_var.trace_add("write", lambda *_: self._update_combined_response_preview())
        self.throughput_unit_var.trace_add("write", lambda *_: self._on_throughput_unit_changed())
        self.qe_unit_var.trace_add("write", lambda *_: self._on_qe_unit_changed())

    def _build_mode_column(self, holder):
        f = self._section(holder, "CALCULATION")
        self.mode_var = self._var("mode", "photometry")
        ttk.Radiobutton(f, text="Photometry", variable=self.mode_var, value="photometry").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(f, text="Spectroscopy", variable=self.mode_var, value="spectroscopy").grid(row=1, column=0, columnspan=2, sticky="w")
        self.mode_var.trace_add("write", lambda *_: self._on_calculation_mode_changed())
        self.texp_var = self._var("exposure_time_s", "60")
        self._entry(f, 2, "Exposure time (s):", self.texp_var)
        self.target_snr_var = self._var("target_snr", "")
        self._entry(f, 3, "Target S/N (optional):", self.target_snr_var)
        ttk.Label(f, text="A target S/N overrides exposure time..", foreground="gray35", wraplength=300,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w")

        f = self._section(holder, "PHOTOMETRY")
        self.aperture_var = self._var("photometric_aperture_radius_arcsec", "1.0")
        self._entry(f, 0, "Aperture radius (arcsec):", self.aperture_var)
        self.source_geometry_var = self._var("source_geometry", "point")
        self.source_area_var = self._var("source_area_arcsec2", "100.0")
        ttk.Label(f, text="Source geometry:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.source_geometry_var, state="readonly",
                     values=("point", "extended")).grid(row=1, column=1, sticky="ew")
        self._entry(f, 2, "Extended source area (arcsec2):", self.source_area_var)
        #ttk.Label(f, text="Point: PSF aperture losses. Extended (galaxy/nebula/planet): the magnitude\n"
        #                  "stays the integrated magnitude spread uniformly over the stated area;\n"
        #                  "valid when the source is much larger than the seeing disc.",
        #          foreground="gray35", wraplength=280, justify="left").grid(row=3, column=0, columnspan=2, sticky="w")

        f = self._section(holder, "SPECTROSCOPY")
        self.spectroscopy_mode_var = self._var("spectroscopy_mode", "slit")
        ttk.Label(f, text="Mode:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.spectroscopy_mode_var, state="readonly",
                     values=("slit", "slitless")).grid(row=0, column=1, sticky="ew")
        self.resolution_var = self._var("resolution_R", "10000")
        self.slit_var = self._var("slit_width_arcsec", "1.0")
        self.extract_var = self._var("extraction_height_arcsec", "1.0")
        self.sampling_var = self._var("pixels_per_resolution_element", "2.0")
        self.slitless_width_var = self._var("slitless_extraction_width_arcsec", "1.0")
        self.slitless_dispersion_var = self._var("slitless_dispersion_aa_pix", "10.0")
        self.slitless_lsf_var = self._var("slitless_intrinsic_fwhm_pix", "1.0")
        self.grating_lines_var = self._var("slitless_grating_lines_mm", "0")
        self.grating_distance_var = self._var("slitless_grating_distance_mm", "42.0")
        self.grating_efficiency_var = self._var("slitless_grating_efficiency", "1.0")
        self.slit_orientation_var = self._var("slit_orientation", "parallactic")
        self.wlmin_var = self._var("wavelength_min_aa", "4000")
        self.wlmax_var = self._var("wavelength_max_aa", "10000")
        self.reference_wavelength_var = self._var("reference_wavelength_aa", "5500")
        for row, label, var in [(1, "Slit resolving power R:", self.resolution_var), (2, "Slit width (arcsec):", self.slit_var),
                                (3, "Extraction height (arcsec):", self.extract_var), (4, "Pixels / slit res. element:", self.sampling_var),
                                (5, "Slitless cross-disp extraction:", self.slitless_width_var),
                                (6, "Slitless dispersion (A/pix):", self.slitless_dispersion_var),
                                (7, "Slitless intrinsic FWHM (pix):", self.slitless_lsf_var),
                                (8, "Grating lines/mm (0=manual):", self.grating_lines_var),
                                (9, "Grating-sensor distance (mm):", self.grating_distance_var),
                                (10, "Grating efficiency (0-1):", self.grating_efficiency_var),
                                (11, "Wavelength minimum (A):", self.wlmin_var), (12, "Wavelength maximum (A):", self.wlmax_var),
                                (13, "S/N reference wavelength (A):", self.reference_wavelength_var)]:
            self._entry(f, row, label, var)
        ttk.Label(f, text="Slit orientation:").grid(row=14, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.slit_orientation_var, state="readonly",
                     values=("parallactic", "fixed")).grid(row=14, column=1, sticky="ew")
#        ttk.Label(f, text="Slitless: a Star Analyser 100/200 is described by its grooves/mm and\n"
#                          "grating-to-sensor distance, which set the dispersion; 0 lines/mm keeps the\n"
#                          "manual A/pix. 'fixed' slit orientation applies the worst-case Filippenko\n"
#                          "atmospheric-dispersion slit loss; 'parallactic' avoids it.",
#                  foreground="gray35", wraplength=280, justify="left").grid(row=15, column=0, columnspan=2, sticky="w")

    def _build_data_column(self, holder):
        f = self._section(holder, "FILTER SELECTOR")
        ttk.Label(f, text="Observing filter:").grid(row=0, column=0, sticky="w")
        self.filter_combo = ttk.Combobox(f, textvariable=self.band_var, state="readonly", values=())
        self.filter_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(1, 4))
        self.filter_combo.bind("<<ComboboxSelected>>", lambda _event: (self._update_filter_display(), self._update_combined_response_preview()))
        self.filter_display_var = tk.StringVar(value="Observing-filter response will appear after the data are loaded.")
        ttk.Label(f, textvariable=self.filter_display_var, foreground="blue", wraplength=310, justify="left").grid(row=2, column=0, columnspan=2, sticky="w")
        self.filter_canvas = tk.Canvas(f, height=145, background="white", highlightthickness=1, highlightbackground="#b5b5b5")
        self.filter_canvas.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.filter_canvas.bind("<Configure>", lambda _event: self._update_filter_display())
        self.mag_system_var.trace_add("write", lambda *_args: (self._update_filter_display(), self._update_combined_response_preview()))

        f = self._section(holder, "STELLAR TEMPLATE SELECTOR")
        self.data_dir_var = self._var("data_directory", "data")
        self._entry(f, 0, "Data directory:", self.data_dir_var)
        ttk.Button(f, text="Reload catalog", command=self._reload_catalog).grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        self.catalog_status = tk.StringVar(value="Catalog not loaded")
        ttk.Label(f, textvariable=self.catalog_status, foreground="blue", wraplength=420, justify="left").grid(row=2, column=0, columnspan=2, sticky="w")
        self.star_search_var = tk.StringVar()
        ttk.Label(f, text="Spectral type search:").grid(row=3, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.star_search_var).grid(row=3, column=1, sticky="ew")
        self.star_search_var.trace_add("write", lambda *_: self._refresh_star_list())
        self.star_tree = ttk.Treeview(f, columns=("name", "spt", "bv"), show="headings", height=7)
        self.star_tree.heading("name", text="Name")
        self.star_tree.heading("spt", text="SpT")
        self.star_tree.heading("bv", text="B-V")
        self.star_tree.column("name", width=130, anchor="center")
        self.star_tree.column("spt", width=65, anchor="center")
        self.star_tree.column("bv", width=60, anchor="center")

        self.star_tree.grid(row=4, column=0, columnspan=2, sticky="ew", pady=3)
        star_scroll = ttk.Scrollbar(f, orient="vertical", command=self.star_tree.yview)
        star_scroll.grid(row=4, column=2, sticky="ns", pady=3)
        self.star_tree.configure(yscrollcommand=star_scroll.set)
        self.star_tree.bind("<<TreeviewSelect>>", self._on_star_selected)
        self.star_status = tk.StringVar(value="Select one template")
        ttk.Label(f, textvariable=self.star_status, foreground="blue", wraplength=420, justify="left").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Button(f, text="Validate V and B−V", command=self._validate_selected_template).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(4, 1))
        self.template_display_var = tk.StringVar(value="Selected template spectrum will appear here.")
        ttk.Label(f, textvariable=self.template_display_var, foreground="blue", wraplength=310,
                  justify="left").grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.template_canvas = tk.Canvas(f, height=145, background="white", highlightthickness=1,
                                         highlightbackground="#b5b5b5")
        self.template_canvas.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.template_canvas.bind("<Configure>", lambda _event: self._update_template_display())

    def _build_actions_column(self, holder):
        f = self._section(holder, "RUN ETC")
#        ttk.Button(f, text="Run ETC", command=self._run_etc).grid(row=0, column=0, sticky="ew", pady=2)
#        ttk.Button(f, text="Exit", command=self._on_exit).grid(row=1, column=0, sticky="ew", pady=(14, 2))
#        tk.Button(
#            f, text="Run ETC", command=self._run_etc,
#            bg="#22a447", activebackground="#178134",
#            fg="white", activeforeground="white",
#        ).grid(row=0, column=0, sticky="ew", pady=2)

        self.run_etc_button = tk.Button(
            f,
            text="Run ETC",
            command=self._run_etc,
            bg="#22a447",
            activebackground="#178134",
            fg="white",
            activeforeground="white",
        )
        self.run_etc_button.grid(row=0, column=0, sticky="ew", pady=2)
            
        tk.Button(
            f, text="Exit", command=self._on_exit,
            bg="#c62828", activebackground="#8e1b1b",
            fg="white", activeforeground="white",
        ).grid(row=1, column=0, sticky="ew", pady=(14, 2))
    


        f = self._section(holder, "OBSERVATORY STATUS")
        self.observatory_status_var = tk.StringVar(value="Updating current observatory status...")
        ttk.Label(f, textvariable=self.observatory_status_var, justify="left", wraplength=220,
                  foreground="#1f4f82").grid(row=0, column=0, sticky="w")

        f = self._section(holder, "TIME CONVERTER")
        midnight = Time(datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00"), scale="utc").mjd
        self.time_convert_input_kind_var = tk.StringVar(value="MJD")
        self.time_convert_output_kind_var = tk.StringVar(value="ISO")
        self.time_convert_input_var = tk.StringVar(value=f"{midnight:.5f}")
        self.time_convert_output_var = tk.StringVar()
        ttk.Label(f, text="Input:").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.time_convert_input_var, width=18).grid(row=0, column=1, sticky="ew")
        ttk.Label(f, text="Input type:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.time_convert_input_kind_var, values=("ISO", "MJD", "JD"),
                     state="readonly", width=7).grid(row=1, column=1, sticky="w")
        ttk.Label(f, text="Output type:").grid(row=2, column=0, sticky="w")
        output_combo = ttk.Combobox(f, textvariable=self.time_convert_output_kind_var, values=("ISO", "MJD", "JD"),
                                    state="readonly", width=7)
        output_combo.grid(row=2, column=1, sticky="w")
        output_combo.bind("<<ComboboxSelected>>", lambda _event: self._convert_time_value())
        ttk.Button(f, text="Convert", command=self._convert_time_value).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(3, 1))
        ttk.Label(f, text="").grid(row=4, column=0, sticky="w")
        output = ttk.Entry(f, textvariable=self.time_convert_output_var, state="readonly", justify="left")
        output.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(3,0))
#        self.time_convert_status_var = tk.StringVar(value="UTC scale.")
#        ttk.Label(f, textvariable=self.time_convert_status_var, foreground="gray35", wraplength=220,
#                  justify="left").grid(row=5, column=0, columnspan=2, sticky="w", pady=(3, 0))
#        self._convert_time_value()

        f = self._section(holder, "GEOGRAPHIC COORDINATES")
        self.dms_input_var = tk.StringVar(value="45 24 28.1 N 11 52 24.1 E")
        self.dms_output_var = tk.StringVar()
        ttk.Label(f, text="DMS latitude, longitude:").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(f, textvariable=self.dms_input_var, width=28).grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(f, text="Convert to decimal degrees", command=self._convert_dms_coordinates).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(3, 1))
        ttk.Label(f, text="").grid(row=3, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.dms_output_var, state="readonly", width=20).grid(row=3, column=1, sticky="ew")
#       self.dms_status_var = tk.StringVar(value="Accepts N/S/E/W, degree symbols, or signed D M S values.")
#       ttk.Label(f, textvariable=self.dms_status_var, foreground="gray35", wraplength=220,
#                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w", pady=(3, 0))
#        self._convert_dms_coordinates()

        about = self._section(holder, "SPETC")
        ttk.Label(about, text="Spectro-Photometry\nExposure Time Calculator",
                  justify="left", wraplength=190, foreground="#1f4f82").pack(fill="x")
        ttk.Separator(about, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(about, text="Version 10.0\n2007-2026",
                  justify="left", wraplength=190).pack(fill="x")
        ttk.Separator(about, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(about, text="Mauro Barbieri\nmauro.barbieri@pm.me",
                  justify="left", wraplength=190).pack(fill="x")

    def _update_timezone_source_state(self):
        """Keep IANA-zone and fixed-offset local-time inputs mutually exclusive."""
        use_iana = self.timezone_source_var.get() == "iana"
        self.timezone_combo.configure(state="readonly" if use_iana else "disabled")
        self.utc_offset_entry.configure(state="disabled" if use_iana else "normal")

    def _timezone_for_display(self):
        if self.timezone_source_var.get() == "iana":
            return ZoneInfo(self.timezone_var.get().strip())
        return timezone(timedelta(hours=float(self.utc_offset_var.get())))

    def _local_timezone_label(self):
        return (self.timezone_var.get().strip() if self.timezone_source_var.get() == "iana"
                else f"UTC{float(self.utc_offset_var.get()):+g}")

    def _update_observatory_status(self):
        try:
            now = Time(datetime.now(timezone.utc), scale="utc")
            location = EarthLocation.from_geodetic(float(self.lon_var.get()) * u.deg,
                                                    float(self.lat_var.get()) * u.deg,
                                                    float(self.elev_var.get()) * u.m)
            local = now.to_datetime(timezone=self._timezone_for_display())
            lst = now.sidereal_time("apparent", longitude=location.lon).to_string(
                unit=u.hourangle, sep=":", precision=1, pad=True)
            sun_alt = get_sun(now).transform_to(AltAz(obstime=now, location=location,
                                                       pressure=0 * u.hPa)).alt.to_value(u.deg)
            self.observatory_status_var.set(
                f"Local: {local:%Y-%m-%d %H:%M:%S}\n"
                f"LST: {lst}\nMJD: {now.mjd:.5f}\nJD: {now.jd:.5f}\n"
                f"Sun altitude: {sun_alt:.1f} deg\nElevation: {float(self.elev_var.get()):.0f} m")
        except Exception as exc:
            self.observatory_status_var.set(f"Observatory status unavailable: {exc}")
        self.after(60000, self._update_observatory_status)

    def _convert_time_value(self):
        try:
            value, kind = self.time_convert_input_var.get().strip(), self.time_convert_input_kind_var.get()
            if kind == "ISO":
                parsed = Time(value.replace("Z", ""), format="isot", scale="utc")
            elif kind == "MJD":
                parsed = Time(float(value), format="mjd", scale="utc")
            else:
                parsed = Time(float(value), format="jd", scale="utc")
            output_kind = self.time_convert_output_kind_var.get()
            if output_kind == "ISO":
                parsed.precision = 2
                output = parsed.isot
            elif output_kind == "MJD":
                output = f"{parsed.mjd:.8f}"
            else:
                output = f"{parsed.jd:.8f}"
            self.time_convert_output_var.set(output)
            #self.time_convert_status_var.set("UTC scale")
        except Exception as exc:
            self.time_convert_output_var.set("")
            self.time_convert_status_var.set(f"Conversion failed: {exc}")

    @staticmethod
    def _parse_dms_value(numbers, direction=None):
        if not numbers or len(numbers) > 3:
            raise ValueError("each coordinate needs decimal degrees or D M S")
        first = float(numbers[0])
        magnitude = abs(first) + (float(numbers[1]) / 60.0 if len(numbers) > 1 else 0.0) + \
                    (float(numbers[2]) / 3600.0 if len(numbers) > 2 else 0.0)
        if direction:
            return -magnitude if direction.upper() in {"S", "W"} else magnitude
        return -magnitude if first < 0 else magnitude

    @classmethod
    def _parse_dms_pair(cls, value):
        import re
        tokens = re.findall(r"[NSEWnsew]|[+-]?\d+(?:\.\d+)?", value)
        groups, current = [], []
        for token in tokens:
            if token.upper() in {"N", "S", "E", "W"}:
                groups.append((current, token.upper())); current = []
            else:
                if current and token[0] in "+-":
                    groups.append((current, None)); current = []
                current.append(token)
        if current:
            groups.append((current, None))
        if len(groups) == 1 and len(groups[0][0]) in {2, 4, 6} and groups[0][1] is None:
            values = groups[0][0]
            if len(values) == 2:
                groups = [([values[0]], None), ([values[1]], None)]
            else:
                split = len(values) // 2
                groups = [(values[:split], None), (values[split:], None)]
        if len(groups) != 2:
            raise ValueError("provide latitude followed by longitude")
        parsed = [(cls._parse_dms_value(numbers, direction), direction) for numbers, direction in groups]
        lat_item = next((item for item in parsed if item[1] in {"N", "S"}), parsed[0])
        lon_item = next((item for item in parsed if item[1] in {"E", "W"}), parsed[1])
        lat, lon = lat_item[0], lon_item[0]
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            raise ValueError("latitude must be within +/-90 and longitude within +/-180 degrees")
        return lat, lon

    def _convert_dms_coordinates(self):
        try:
            lat, lon = self._parse_dms_pair(self.dms_input_var.get())
            self.dms_output_var.set(f"LAT {lat:.4f} LON {lon:.4f}")
            #self.dms_status_var.set("East long>0")
        except Exception as exc:
            self.dms_output_var.set("")
            self.dms_status_var.set(f"Conversion failed: {exc}")

    def _load_site_records(self):
        """Load defaults plus the persistent local site list."""
        records = {
            name: {"name": name, "lat": obs.lat, "lon": obs.lon, "elev": obs.elev,
                   "utc_offset_h": round(obs.lon / 15), "timezone": "Europe/Rome",
                   "sky_ab_mag_arcsec2": obs.mag_sky}
            for name, obs in OBSERVATORY_PRESETS.items()
        }
        try:
            with SITES_FILE.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            entries = payload.get("sites", []) if isinstance(payload, dict) else payload
            for item in entries:
                name = str(item["name"]).strip()
                records[name] = {
                    "name": name, "lat": float(item["lat"]), "lon": float(item["lon"]),
                    "elev": float(item.get("elev", 0.0)),
                    "utc_offset_h": float(item.get("utc_offset_h", round(float(item["lon"]) / 15))),
                    "timezone": str(item.get("timezone", "")),
                    "sky_ab_mag_arcsec2": float(item.get("sky_ab_mag_arcsec2", 21.5)),
                }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
        return records

    def _site_names(self):
        return sorted(self._site_records, key=str.casefold)

    def _write_site_records(self):
        payload = {"schema_version": 1, "sites": [self._site_records[name] for name in self._site_names()]}
        with SITES_FILE.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)

    def _refresh_site_selector(self):
        if hasattr(self, "site_combo"):
            self.site_combo.configure(values=self._site_names())

    def _on_obs_changed(self, *_):
        site = self._site_records.get(self.obs_var.get())
        if site:
            self.lat_var.set(f"{site['lat']:.5f}")
            self.lon_var.set(f"{site['lon']:.5f}")
            self.elev_var.set(f"{site['elev']:.0f}")
            self.utc_offset_var.set(str(site["utc_offset_h"]))
            if site.get("timezone"):
                self.timezone_var.set(str(site["timezone"]))
                self.timezone_source_var.set("iana")
                self._update_timezone_source_state()
            self.sky_var.set(f"{site['sky_ab_mag_arcsec2']:.2f}")

    def _save_current_site(self):
        name = simpledialog.askstring("Save current site", "Site name:", initialvalue=self.obs_var.get())
        if not name:
            return
        try:
            name = name.strip()
            if not name:
                raise ValueError("Site name cannot be empty.")
            self._site_records[name] = {
                "name": name, "lat": float(self.lat_var.get()), "lon": float(self.lon_var.get()),
                "elev": float(self.elev_var.get()), "utc_offset_h": float(self.utc_offset_var.get()),
                "timezone": self.timezone_var.get().strip(),
                "sky_ab_mag_arcsec2": float(self.sky_var.get()),
            }
            self._write_site_records()
            self.obs_var.set(name)
            self._refresh_site_selector()
            self.status_var.set(f"Site {name!r} saved to {SITES_FILE.name}.")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save site", str(exc))

    def _import_site_file(self):
        path = filedialog.askopenfilename(title="Import site JSON", filetypes=[("JSON", "*.json"), ("All files", "*")])
        if not path:
            return
        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            entries = payload.get("sites", []) if isinstance(payload, dict) else payload
            imported = 0
            for item in entries:
                name = str(item["name"]).strip()
                self._site_records[name] = {
                    "name": name, "lat": float(item["lat"]), "lon": float(item["lon"]),
                    "elev": float(item.get("elev", 0.0)),
                    "utc_offset_h": float(item.get("utc_offset_h", round(float(item["lon"]) / 15))),
                    "timezone": str(item.get("timezone", "")),
                    "sky_ab_mag_arcsec2": float(item.get("sky_ab_mag_arcsec2", 21.5)),
                }
                imported += 1
            if not imported:
                raise ValueError("No valid site entries found.")
            self._write_site_records()
            self._refresh_site_selector()
            self.status_var.set(f"Imported {imported} site(s) into {SITES_FILE.name}.")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            messagebox.showerror("Import site", f"Could not import site file:\n{exc}")

    @staticmethod
    def _instrument_profile_keys():
        return (
            "diameter_mm", "obstruction_mm", "throughput_excluding_qe", "focal_length_mm",
            "pixel_size_um", "gain_e_adu", "full_well_e", "bit_depth", "read_noise_e",
            "dark_current_e_s_pix", "seeing_arcsec", "sky_model", "sky_ab_mag_arcsec2",
            "photometric_aperture_radius_arcsec", "resolution_R", "slit_width_arcsec",
            "extraction_height_arcsec", "pixels_per_resolution_element", "wavelength_min_aa",
            "wavelength_max_aa", "reference_wavelength_aa", "spectroscopy_mode",
            "slitless_extraction_width_arcsec", "slitless_dispersion_aa_pix", "slitless_intrinsic_fwhm_pix",
            "qe_wavelength_unit", "throughput_wavelength_unit",
        )

    def _choose_throughput_curve(self):
        path = filedialog.askopenfilename(title="Select instrument transmission curve",
                                          filetypes=[("Data files", "*.dat *.txt *.csv"), ("All files", "*")])
        if not path:
            return
        try:
            self.throughput_curve = load_transmission_curve(path, self.throughput_unit_var.get())
            self.throughput_source_path = Path(path).resolve()
            self.throughput_path_var.set(str(self.throughput_source_path))
            self.profile_status.set(f"Instrument transmission loaded: {self.throughput_source_path.name}.")
            self.status_var.set("Instrument transmission curve loaded."); self._update_combined_response_preview()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Instrument transmission", str(exc))

    def _on_qe_unit_changed(self):
        """Reinterpret the QE file when the user explicitly changes its unit."""
        path = self.qe_source_path
        if path is not None and Path(path).is_file():
            try:
                self.qe_curve = load_qe_curve(path, self.qe_unit_var.get())
            except (OSError, ValueError) as exc:
                self.qe_curve = None
                self.catalog_status.set(f"QE: {exc}")
        self._update_combined_response_preview()

    def _on_throughput_unit_changed(self):
        """Reinterpret the selected optics-response file in its declared unit."""
        path = self.throughput_source_path
        if path is None and self.throughput_path_var.get().strip():
            path = Path(self.throughput_path_var.get()).expanduser()
        if path is not None and Path(path).is_file():
            try:
                self.throughput_curve = load_transmission_curve(path, self.throughput_unit_var.get())
                self.throughput_source_path = Path(path).resolve()
            except (OSError, ValueError) as exc:
                self.throughput_curve = None
                self.catalog_status.set(f"Instrument transmission: {exc}")
        self._update_combined_response_preview()

    def _choose_slit_resolution_curve(self):
        path = filedialog.askopenfilename(title="Select slit width / resolving-power curve",
                                          filetypes=[("Data files", "*.dat *.txt *.csv"), ("All files", "*")])
        if not path:
            return
        try:
            curve = load_transmission_curve(path)
            curve = curve[np.argsort(curve[:, 0])]
            if np.any(curve[:, 0] <= 0) or np.any(curve[:, 1] <= 0):
                raise ValueError("Slit resolution curve requires positive width and R values.")
            self.slit_resolution_curve = curve
            self.slit_resolution_source_path = Path(path).resolve()
            self.slit_resolution_path_var.set(str(self.slit_resolution_source_path))
            self.status_var.set("Calibrated slit-resolution curve loaded.")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Slit resolution curve", str(exc))

    def _save_instrument_profile(self):
        if self.qe_source_path is None or not Path(self.qe_source_path).is_file():
            messagebox.showerror("Save instrument profile", "Load a valid QE curve before saving a profile.")
            return
        path = filedialog.asksaveasfilename(title="Save instrument profile", defaultextension=".json",
                                            initialfile="instrument_profile.json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            profile_path = Path(path)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            qe_destination = profile_path.parent / f"{profile_path.stem}_qe.dat"
            source = Path(self.qe_source_path).resolve()
            if source != qe_destination.resolve():
                shutil.copy2(source, qe_destination)
            payload = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "profile_name": profile_path.stem,
                "qe_file": qe_destination.name,
                "instrument": {key: self._vars[key].get() for key in self._instrument_profile_keys()},
            }
            if self.throughput_source_path is not None and Path(self.throughput_source_path).is_file():
                throughput_destination = profile_path.parent / f"{profile_path.stem}_throughput.dat"
                source = Path(self.throughput_source_path).resolve()
                if source != throughput_destination.resolve():
                    shutil.copy2(source, throughput_destination)
                payload["throughput_file"] = throughput_destination.name
            if self.slit_resolution_source_path is not None and Path(self.slit_resolution_source_path).is_file():
                resolution_destination = profile_path.parent / f"{profile_path.stem}_slit_resolution.dat"
                source = Path(self.slit_resolution_source_path).resolve()
                if source != resolution_destination.resolve():
                    shutil.copy2(source, resolution_destination)
                payload["slit_resolution_file"] = resolution_destination.name
            with profile_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
            self.active_profile_path = str(profile_path.resolve())
            self._save_config()
            curve_names = [payload[name] for name in ("throughput_file", "slit_resolution_file") if name in payload]
            curve_note = " and " + ", ".join(curve_names) if curve_names else ""
            self.profile_status.set(f"Saved {profile_path.name} with associated {qe_destination.name}{curve_note}.")
            self.status_var.set("Instrument profile saved.")
        except (OSError, KeyError) as exc:
            messagebox.showerror("Save instrument profile", str(exc))

    def _load_instrument_profile(self):
        path = filedialog.askopenfilename(title="Load instrument profile", filetypes=[("JSON", "*.json"), ("All files", "*")])
        if not path:
            return
        self._load_instrument_profile_path(Path(path), report_errors=True)

    def _load_instrument_profile_path(self, profile_path, report_errors):
        try:
            profile_path = Path(profile_path).expanduser().resolve()
            with profile_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            values = payload["instrument"]
            for key in self._instrument_profile_keys():
                if key in values:
                    self._vars[key].set(str(values[key]))
            qe_path = profile_path.parent / payload["qe_file"]
            self.qe_curve = load_qe_curve(qe_path, self.qe_unit_var.get())
            self.qe_source_path = qe_path
            throughput_name = payload.get("throughput_file")
            if throughput_name:
                throughput_path = profile_path.parent / throughput_name
                self.throughput_curve = load_transmission_curve(throughput_path, self.throughput_unit_var.get())
                self.throughput_source_path = throughput_path
                self.throughput_path_var.set(str(throughput_path))
            else:
                self.throughput_curve = None
                self.throughput_source_path = None
                self.throughput_path_var.set("")
            resolution_name = payload.get("slit_resolution_file")
            if resolution_name:
                resolution_path = profile_path.parent / resolution_name
                curve = load_transmission_curve(resolution_path)
                curve = curve[np.argsort(curve[:, 0])]
                if np.any(curve[:, 0] <= 0) or np.any(curve[:, 1] <= 0):
                    raise ValueError("Slit resolution curve requires positive width and R values.")
                self.slit_resolution_curve = curve
                self.slit_resolution_source_path = resolution_path
                self.slit_resolution_path_var.set(str(resolution_path))
            else:
                self.slit_resolution_curve = None
                self.slit_resolution_source_path = None
                self.slit_resolution_path_var.set("")
            self.active_profile_path = str(profile_path)
            self.profile_status.set(f"Loaded {profile_path.name}; QE: {qe_path.name}.")
            self.status_var.set("Instrument profile loaded.")
            self._save_config()
            return True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            if report_errors:
                messagebox.showerror("Load instrument profile", f"Could not load profile:\n{exc}")
            elif hasattr(self, "profile_status"):
                self.profile_status.set(f"Saved profile could not be restored: {exc}")
            return False

    def _slider_changed(self, value):
        minute = int(round(float(value)))
        candidate = f"{minute // 60:02d}:{minute % 60:02d}"
        if self.time_var.get() != candidate:
            self.time_var.set(candidate)

    def _sync_slider_from_time(self):
        try:
            hour, minute = map(int, self.time_var.get().split(":"))
            value = min(max(hour * 60 + minute, 0), 1439)
            if abs(self.time_slider.get() - value) > 0.5:
                self.time_slider.set(value)
        except (ValueError, AttributeError):
            pass

    def _reload_catalog(self):
        base = self._data_directory()
        messages = []
        try:
            self.star_catalog = scat.parse_interpola_db(base / "interpola.db.csv")
            messages.append(f"{len(self.star_catalog)} templates")
        except Exception as exc:
            self.star_catalog = []; messages.append(f"catalog: {exc}")
        try:
            self.filter_resp_data = fcat.filters_directory(base)
            labels = fcat.list_filter_labels(self.filter_resp_data)
            self.filter_combo.configure(values=labels)
            self.reference_filter_combo.configure(values=labels)
            if self.mode_var.get() == "spectroscopy" and "BLANK" in labels:
                self.band_var.set("BLANK")
            elif self.band_var.get() not in labels:
                self.band_var.set("Bessell.V" if "Bessell.V" in labels else labels[0])
            if self.reference_band_var.get() not in labels:
                self.reference_band_var.set("Bessell.V" if "Bessell.V" in labels else labels[0])
            messages.append(f"{len(labels)} XML filters")
        except Exception as exc:
            self.filter_resp_data = None; messages.append(f"filters: {exc}")
        try:
            self.qe_source_path = base / "qe.dat"
            self.qe_curve = load_qe_curve(self.qe_source_path, self.qe_unit_var.get())
            messages.append("QE loaded")
        except Exception as exc:
            self.qe_curve = None; self.qe_source_path = None; messages.append(f"QE: {exc}")
        try:
            configured = Path(self.throughput_path_var.get()).expanduser() if self.throughput_path_var.get().strip() else base / "throughput.dat"
            if configured.is_file():
                self.throughput_curve = load_transmission_curve(configured, self.throughput_unit_var.get())
                self.throughput_source_path = configured.resolve()
                self.throughput_path_var.set(str(self.throughput_source_path))
                messages.append("instrument transmission loaded")
            else:
                self.throughput_curve = None; self.throughput_source_path = None
        except Exception as exc:
            self.throughput_curve = None; self.throughput_source_path = None; messages.append(f"transmission: {exc}")
        try:
            configured = Path(self.slit_resolution_path_var.get()).expanduser() if self.slit_resolution_path_var.get().strip() else base / "slit_resolution.dat"
            if configured.is_file():
                curve = load_transmission_curve(configured)
                curve = curve[np.argsort(curve[:, 0])]
                if np.any(curve[:, 0] <= 0) or np.any(curve[:, 1] <= 0):
                    raise ValueError("Slit resolution curve requires positive width and R values.")
                self.slit_resolution_curve = curve
                self.slit_resolution_source_path = configured.resolve()
                self.slit_resolution_path_var.set(str(self.slit_resolution_source_path))
                messages.append("slit-resolution curve loaded")
            else:
                self.slit_resolution_curve = None; self.slit_resolution_source_path = None
        except Exception as exc:
            self.slit_resolution_curve = None; self.slit_resolution_source_path = None; messages.append(f"slit R: {exc}")
        try:
            atmosphere_path = base / "earth_atmospheric_transmission.fits"
            if atmosphere_path.is_file():
                curve = load_fits_transmission_curve(atmosphere_path)
                self.earth_atmosphere_curve = curve.data
                messages.append(f"{curve.name} loaded ({curve.coverage_aa[0]:.0f}-{curve.coverage_aa[1]:.0f} Å)")
            else:
                self.earth_atmosphere_curve = None
                messages.append("earth atmosphere FITS not found; broad-band fallback")
        except Exception as exc:
            self.earth_atmosphere_curve = None; messages.append(f"earth atmosphere: {exc}")
        self.catalog_status.set(" | ".join(messages)); self._refresh_star_list(); self._update_filter_display(); self._update_template_display(); self._update_combined_response_preview()
        if hasattr(self, "profile_status") and self.qe_source_path is not None:
            self.profile_status.set(f"QE loaded from {self.qe_source_path}.")

    def _on_calculation_mode_changed(self):
        """Use the filter-free response by default for spectroscopy."""
        if self.mode_var.get() == "spectroscopy" and self.filter_resp_data is not None:
            try:
                if "BLANK" in fcat.list_filter_labels(self.filter_resp_data):
                    self.band_var.set("BLANK")
            except (OSError, ValueError):
                pass
        if hasattr(self, "filter_canvas"):
            self._update_filter_display()
            self._update_combined_response_preview()

    def _update_filter_display(self):
        """Draw a compact response curve; it is a display only, not new physics."""
        if not hasattr(self, "filter_canvas"):
            return
        canvas = self.filter_canvas
        canvas.delete("all")
        if self.filter_resp_data is None:
            return
        try:
            profile = fcat.load_filter_profile(self.filter_resp_data, self.band_var.get(), self.mag_system_var.get())
            wave = profile.transmission[:, 0]
            trans = np.clip(profile.transmission[:, 1], 0.0, None)
            centre = profile.pivot_wavelength_aa
            width_eff = profile.effective_width_aa
            fwhm = profile.fwhm_aa
            self.filter_display_var.set(
                f"Pivot: {centre:.0f} Å | width: {width_eff:.0f} Å | FWHM: {fwhm:.0f} Å\n"
                f"{profile.magnitude_system} zero point: {profile.zero_point_jy:.4g} Jy")
            width = max(canvas.winfo_width(), 2)
            height = max(canvas.winfo_height(), 2)
            left, right, top, bottom = 34, width - 8, 8, height - 22
            grid_colour = "#dddddd"
            for fraction in np.linspace(0.0, 1.0, 6):
                x_grid = left + fraction * (right - left)
                y_grid = top + fraction * (bottom - top)
                canvas.create_line(x_grid, top, x_grid, bottom, fill=grid_colour)
                canvas.create_line(left, y_grid, right, y_grid, fill=grid_colour)
            canvas.create_line(left, top, left, bottom, right, bottom, fill="#666666")
            nonzero = trans > 0
            if not np.any(nonzero):
                return
            xmin, xmax = float(wave[nonzero].min()), float(wave[nonzero].max())
            if xmax <= xmin:
                xmax = xmin + 1.0
            ymax = max(float(trans.max()), 1.0)
            x = left + (wave - xmin) / (xmax - xmin) * (right - left)
            y = bottom - trans / ymax * (bottom - top)
            points = [point for pair in zip(x, y) for point in pair]
            canvas.create_line(*points, fill="#1f77b4", width=2)
            for fraction in np.linspace(0.0, 1.0, 26):
                x_tick = left + fraction * (right - left)
                tick_height = 5 if round(fraction * 25) % 5 == 0 else 3
                canvas.create_line(x_tick, bottom, x_tick, bottom + tick_height, fill="#666666")
            canvas.create_text(left, height - 10, text=f"{xmin:.0f}", anchor="w", fill="#555555", font=("TkDefaultFont", 8))
            canvas.create_text(right, height - 10, text=f"{xmax:.0f} Å", anchor="e", fill="#555555", font=("TkDefaultFont", 8))
            canvas.create_text(4, top, text=f"{ymax:.2g}", anchor="nw", fill="#555555", font=("TkDefaultFont", 8))
        except (OSError, KeyError, ValueError):
            self.filter_display_var.set("Selected filter response is unavailable.")

    def _update_template_display(self):
        """Draw the catalogue flux distribution of the selected template."""
        if not hasattr(self, "template_canvas"):
            return
        canvas = self.template_canvas
        canvas.delete("all")
        if self.star_spec is None:
            return
        wave = np.asarray(self.star_spec[:, 0], dtype=float)
        flux = np.asarray(self.star_spec[:, 1], dtype=float)
        valid = np.isfinite(wave) & np.isfinite(flux) & (flux > 0)
        if valid.sum() < 2:
            self.template_display_var.set("The selected template has no positive flux samples.")
            return
        wave, flux = wave[valid], flux[valid]
        log_flux = np.log10(flux)
        xmin, xmax = float(wave.min()), float(wave.max())
        ymin, ymax = float(log_flux.min()), float(log_flux.max())
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        width, height = max(canvas.winfo_width(), 2), max(canvas.winfo_height(), 2)
        left, right, top, bottom = 38, width - 8, 8, height - 22
        for fraction in np.linspace(0.0, 1.0, 6):
            x_grid = left + fraction * (right - left)
            y_grid = top + fraction * (bottom - top)
            canvas.create_line(x_grid, top, x_grid, bottom, fill="#dddddd")
            canvas.create_line(left, y_grid, right, y_grid, fill="#dddddd")
        canvas.create_line(left, top, left, bottom, right, bottom, fill="#666666")
        x = left + (wave - xmin) / (xmax - xmin) * (right - left)
        y = bottom - (log_flux - ymin) / (ymax - ymin) * (bottom - top)
        canvas.create_line(*[point for pair in zip(x, y) for point in pair], fill="#8b3a3a", width=1.5)
        for fraction in np.linspace(0.0, 1.0, 26):
            x_tick = left + fraction * (right - left)
            tick_height = 5 if round(fraction * 25) % 5 == 0 else 3
            canvas.create_line(x_tick, bottom, x_tick, bottom + tick_height, fill="#666666")
        self.template_display_var.set(
            f"Template Fλ distribution: {xmin:.0f}–{xmax:.0f} Å; log flux span {ymin:.2f} to {ymax:.2f}.")
        canvas.create_text(left, height - 10, text=f"{xmin:.0f}", anchor="w", fill="#555555", font=("TkDefaultFont", 8))
        canvas.create_text(right, height - 10, text=f"{xmax:.0f} Å", anchor="e", fill="#555555", font=("TkDefaultFont", 8))
        canvas.create_text(3, top, text="log Fλ", anchor="nw", fill="#555555", font=("TkDefaultFont", 8))


    def _set_etc_ready(self, ready, message=""):
        self.etc_ready = bool(ready)
    
        if hasattr(self, "run_etc_button"):
            self.run_etc_button.configure(
                state="normal" if self.etc_ready else "disabled"
            )
    
        if message:
            self.response_preview_status.set(message)


    def _update_combined_response_preview(self):
        """Show the display-only system response, zero-filled outside coverage."""
        if not hasattr(self, "response_preview_section"):
            return
        ready = self.filter_resp_data is not None and self.star_spec is not None
        if not ready:
            self._set_etc_ready(
                False,
                "Select an observing filter and a stellar template."
            )
            self.response_preview_section.pack_forget()
            return
        try:
            profile = fcat.load_filter_profile(self.filter_resp_data, self.band_var.get(), self.mag_system_var.get())
            wave = np.asarray(profile.transmission[:, 0], dtype=float)
            response = np.clip(np.asarray(profile.transmission[:, 1], dtype=float), 0.0, 1.0)
            positive = response > 0
            wave = wave[positive]; response = response[positive]
            if wave.size < 2:
                raise ValueError("Observing filter has no positive response samples.")
            source = interpolate_zero_filled(wave, self.star_spec, "template spectrum")
            qe = (interpolate_zero_filled(wave, self.qe_curve, "QE curve", clip=(0.0, 1.0))
                  if self.qe_curve is not None else np.zeros_like(wave))
            earth = (interpolate_zero_filled(wave, self.earth_atmosphere_curve,
                                             "earth atmospheric transmission", clip=(0.0, 1.0))
                     if self.earth_atmosphere_curve is not None else np.zeros_like(wave))
            throughput = (interpolate_zero_filled(wave, self.throughput_curve,
                                                  "instrument throughput curve", clip=(0.0, 1.0))
                          if self.throughput_curve is not None else np.ones_like(wave))
            detected = source * response * qe * throughput * earth * max(float(self.eff_var.get()), 0.0) * wave

            # The reference filter defines the user-provided magnitude.
            # It must overlap the template with positive flux.
            reference_profile = fcat.load_filter_profile(
                self.filter_resp_data,
                self.reference_band_var.get(),
                self.mag_system_var.get(),
            )
            ref_wave = np.asarray(reference_profile.transmission[:, 0], dtype=float)
            ref_response = np.clip(
                np.asarray(reference_profile.transmission[:, 1], dtype=float),
                0.0, 1.0,
            )
            ref_source = interpolate_zero_filled(
                ref_wave,
                self.star_spec,
                "template spectrum",
            )
            
            reference_signal = np.trapezoid(
                np.clip(ref_source, 0.0, None) * ref_response,
                ref_wave,
            )
            
            detected_signal = np.trapezoid(
                np.clip(detected, 0.0, None),
                wave,
            )
            
            if not np.isfinite(reference_signal) or reference_signal <= 0.0:
                raise ValueError(
                    "The template has no positive flux in the selected reference "
                    "magnitude filter. Select another template or reference filter."
                )
            
            if not np.isfinite(detected_signal) or detected_signal <= 0.0:
                raise ValueError(
                    "The selected template, observing filter, QE, atmosphere and "
                    "throughput produce zero detected target flux."
                )
        
            maximum = float(np.nanmax(detected))
            normalized = detected / maximum if np.isfinite(maximum) and maximum > 0 else np.zeros_like(detected)
            ax = self.response_preview_axis
            ax.clear()
            ax.plot(wave, normalized, color="#6a3d9a", linewidth=1.5)
            ax.set_xlim(float(wave.min()), float(wave.max()))
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel("Wavelength [Å]", fontsize=8)
            ax.set_ylabel("Norm. response", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.minorticks_on()
            ax.grid(True, which="major", color="#d0d0d0", linewidth=0.6)
            ax.grid(True, which="minor", color="#ededed", linewidth=0.4)
            self.response_preview_figure.tight_layout(pad=0.6)
            self.response_preview_canvas.draw_idle()
            self.response_preview_status.set(
                "Template × filter × QE × optics × zenith Earth atmosphere. Missing coverage is zero.")
            if not self.response_preview_section.winfo_manager():
                self.response_preview_section.pack(fill="x", padx=5, pady=4, before=self.instrument_profile_section)
            self._set_etc_ready(True)
#        except Exception as exc:
#            self.response_preview_status.set(f"System response unavailable: {exc}")
#            if not self.response_preview_section.winfo_manager():
#                self.response_preview_section.pack(fill="x", padx=5, pady=4, before=self.instrument_profile_section)
        except Exception as exc:
            self._set_etc_ready(False, f"Cannot run ETC: {exc}")
            self.response_preview_axis.clear()
            self.response_preview_canvas.draw_idle()
        
            if not self.response_preview_section.winfo_manager():
                self.response_preview_section.pack(
                    fill="x", padx=5, pady=4,
                    before=self.instrument_profile_section,
                )
    


    def _refresh_star_list(self):
        if not hasattr(self, "star_tree"): return
        self.star_tree.delete(*self.star_tree.get_children()); self.star_id_map = {}
        query = self.star_search_var.get().strip() or None
        for rec in scat.search_stars(self.star_catalog, spt_prefix=query):
            item = self.star_tree.insert("", "end", values=(rec.name, rec.spt, f"{rec.bv0:.2f}"))
            self.star_id_map[item] = rec

    def _on_star_selected(self, *_):
        selection = self.star_tree.selection()
        if not selection: return
        rec = self.star_id_map.get(selection[0])
        if not rec: return
        try:
            self.star_spec = scat.load_star_spectrum(rec, self._data_directory())
            self.selected_star = rec
            self.star_status.set(f"Flux-calibrated template loaded; mv0 = {rec.mv0:.3f} visual mag.")
            self._update_template_display(); self._update_combined_response_preview()
        except Exception as exc:
            self.star_spec = None; self.selected_star = None
            messagebox.showerror("Template error", str(exc))

    def _validate_selected_template(self):
        """Compare synthetic BPGS-style V and B-V against catalogue metadata."""
        if self.star_spec is None or self.selected_star is None or self.filter_resp_data is None:
            messagebox.showinfo("Template validation", "Load filter data and select a template first.")
            return
        try:
            v_profile = fcat.load_filter_profile(self.filter_resp_data, "Bessell.V", "Vega")
            b_profile = fcat.load_filter_profile(self.filter_resp_data, "Bessell.B", "Vega")
            synthetic_v = synthetic_magnitude(self.star_spec, v_profile.transmission, v_profile.zero_point_jy, v_profile.detector_type)
            synthetic_b = synthetic_magnitude(self.star_spec, b_profile.transmission, b_profile.zero_point_jy, b_profile.detector_type)
            synthetic_bv = synthetic_b - synthetic_v
            dv = synthetic_v - float(self.selected_star.mv0)
            dbv = synthetic_bv - float(self.selected_star.bv0)
            message = (f"Synthetic V: {synthetic_v:.3f}; catalogue mv0: {self.selected_star.mv0:.3f}; ΔV: {dv:+.3f}\n"
                       f"Synthetic B−V: {synthetic_bv:.3f}; catalogue B−V: {self.selected_star.bv0:.3f}; Δ(B−V): {dbv:+.3f}")
            self.template_display_var.set(message)
            messagebox.showinfo("Template V / B−V validation", message)
        except Exception as exc:
            messagebox.showerror("Template validation", f"Could not validate template:\n{exc}")

    def _data_directory(self):
        """Resolve relative data paths first beside the program, then in CWD."""
        raw = self.data_dir_var.get().strip() or "data"
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        beside_program = Path(__file__).resolve().parent / path
        if beside_program.exists():
            return beside_program
        return Path.cwd() / path

    @staticmethod
    def _query_simbad_coordinates(name):
        """Return decimal ICRS coordinates from SIMBAD's sexagesimal response."""
        result = Simbad.query_object(name)
        if result is None or len(result) == 0:
            raise ValueError(f"SIMBAD could not resolve target name: {name!r}")
        columns = {column.lower(): column for column in result.colnames}
        if "ra" not in columns or "dec" not in columns:
            raise ValueError("SIMBAD response did not include RA and Dec.")
        raw_ra, raw_dec = result[columns["ra"]][0], result[columns["dec"]][0]
        if isinstance(raw_ra, bytes):
            raw_ra = raw_ra.decode()
        if isinstance(raw_dec, bytes):
            raw_dec = raw_dec.decode()
        coordinate = SkyCoord(str(raw_ra), str(raw_dec), unit=(u.hourangle, u.deg), frame="icrs")
        return coordinate.ra.deg, coordinate.dec.deg

    def _resolve_target_name(self):
        """Resolve only on explicit user request; running the ETC never queries SIMBAD."""
        name = self.target_name_var.get().strip()
        if not name:
            self.target_resolution_status.set("Enter a SIMBAD target name first.")
            return
        try:
            ra, dec = self._query_simbad_coordinates(name)
            self.ra_var.set(f"{ra:.8f}")
            self.dec_var.set(f"{dec:.8f}")
            self.coord_format_var.set("decimal")
            self.target_resolution_status.set(f"Resolved {name}: RA {ra:.8f} deg, Dec {dec:.8f} deg.")
            self._save_config()
        except Exception as exc:
            self.target_resolution_status.set(f"Could not resolve {name!r}: {exc}")
            messagebox.showerror("SIMBAD resolution", str(exc))

    def _target_coordinates(self):
        """Parse the user-entered ICRS coordinates without any network request."""
        name = self.target_name_var.get().strip()
        if self.coord_format_var.get() == "sexagesimal":
            coordinate = SkyCoord(self.ra_var.get().strip(), self.dec_var.get().strip(),
                                  unit=(u.hourangle, u.deg), frame="icrs")
            return coordinate.ra.deg, coordinate.dec.deg, name or "Target"
        return float(self.ra_var.get()), float(self.dec_var.get()), name or "Target"

    def _ut_datetime(self):
        date = datetime.strptime(self.date_var.get() + " " + self.time_var.get(), "%Y-%m-%d %H:%M")
        if self.time_ref_var.get() == "local":
            if self.timezone_source_var.get() == "iana":
                zone_name = self.timezone_var.get().strip()
                if not zone_name:
                    raise ValueError("Select an IANA timezone or choose UTC offset.")
                try:
                    return date.replace(tzinfo=ZoneInfo(zone_name)).astimezone(timezone.utc).replace(tzinfo=None)
                except Exception as exc:
                    raise ValueError(f"Invalid IANA timezone {zone_name!r}.") from exc
            return date - timedelta(hours=float(self.utc_offset_var.get()))
        return date

    @staticmethod
    def _slice_track(track, first, last):
        return {key: np.asarray(value)[first:last + 1] for key, value in track.items()}

    def _observing_window(self, track, selected_idx):
        """Return the contiguous day or night interval containing selected time."""
        is_day = track["alt_sun"] >= 0
        selected_day = bool(is_day[selected_idx])
        mask = is_day if selected_day else ~is_day
        starts = np.where(mask & np.r_[True, ~mask[:-1]])[0]
        ends = np.where(mask & np.r_[~mask[1:], True])[0]
        for first, last in zip(starts, ends):
            if first <= selected_idx <= last:
                return self._slice_track(track, first, last), selected_idx - first
        raise ValueError("Could not identify the selected observing interval.")

    @staticmethod
    def _slider_indices(valid_indices, selected_idx, maximum=25):
        choices = np.unique(np.linspace(0, len(valid_indices) - 1, min(maximum, len(valid_indices)), dtype=int))
        return sorted(set(valid_indices[i] for i in choices) | {selected_idx})

    def _atmosphere_dict(self, airmass, transmission_curve):
        """Common atmosphere/PSF description consumed by the ETC engines."""
        return {"airmass": airmass, "seeing_arcsec": float(self.seeing_var.get()),
                "transmission_curve": transmission_curve,
                "psf_model": self.psf_model_var.get(),
                "moffat_beta": float(self.moffat_beta_var.get()),
                "elevation_m": float(self.elev_var.get())}

    def _source_geometry_kwargs(self):
        geometry = self.source_geometry_var.get().strip().lower()
        return {"source_geometry": geometry,
                "source_area_arcsec2": float(self.source_area_var.get()) if geometry == "extended" else None}

    def _sky_models_for_track(self, track, vega_profile, ra_deg, dec_deg):
        """Build observed ground-sky inputs for every planning time sample."""
        aperture = float(self.aperture_var.get())
        if self.sky_model_var.get() == "fixed_ab":
            fixed = {"sky_mag": float(self.sky_var.get()), "sky_zero_point_jy": 3631.0,
                     "sky_at_telescope": True, "aperture_radius_arcsec": aperture}
            return [fixed.copy() for _ in track["jd"]]
        pivot = vega_profile.pivot_wavelength_aa
        if not np.isfinite(pivot):
            pivot = float(np.average(vega_profile.transmission[:, 0], weights=vega_profile.transmission[:, 1]))
        # Zodiacal light and integrated starlight depend on where the
        # telescope points: evaluate the field's ecliptic and galactic
        # latitude once from the ICRS coordinates.
        field = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
        ecliptic_lat = float(field.barycentricmeanecliptic.lat.deg)
        galactic_lat = float(field.galactic.b.deg)
        models = []
        colour_wave = np.asarray(BAND_WAVELENGTH_NM, dtype=float) * 10.0
        for utc, alt, airmass, sun_alt, moon_alt, moon_sep, phase in zip(
                track["utc_datetime"], track["alt_target"], track["airmass_target"], track["alt_sun"],
                track["alt_moon"], track["moon_sep_deg"], track["phase_moon"]):
            base_mag = sky_magnitude_vega(pivot, utc, alt, airmass, sun_alt,
                                          ecliptic_lat_deg=ecliptic_lat, galactic_lat_deg=galactic_lat)
            # Krisciunas--Schaefer moonlight terms are evaluated in the nine
            # supplied broad bands, then used as a spectral colour model.  The
            # ING dark/twilight/day brightness remains the normalization.
            moon_airmass = 1.0 / np.sin(np.deg2rad(moon_alt)) if moon_alt >= 5.0 else 99.0
            common = dict(year=utc.year, month=utc.month, day=utc.day, hour=utc.hour, minute=utc.minute,
                          ecliptic_lat_deg=ecliptic_lat, galactic_lat_deg=galactic_lat,
                          airmass_target=max(float(airmass) if np.isfinite(airmass) else 1.0, 1.0),
                          airmass_moon=moon_airmass, lunar_phase_deg=max(0.0, 180.0 - float(phase)),
                          moon_separation_deg=float(moon_sep), moon_zenith_dist_deg=90.0 - float(moon_alt),
                          target_zenith_dist_deg=90.0 - float(np.clip(alt, 0.0, 90.0)))
            dark = sky_brightness_total(**common, include_moon=False, include_sun_twilight=False)
            total = sky_brightness_total(**common, include_moon=moon_alt >= 0.0, include_sun_twilight=False)
            pivot_dark = float(np.interp(pivot, colour_wave, dark))
            pivot_total = float(np.interp(pivot, colour_wave, total))
            moon_factor = 10.0**(-0.4 * (pivot_total - pivot_dark)) - 1.0
            total_mag = -2.5 * np.log10(10.0**(-0.4 * base_mag) * max(1.0 + moon_factor, 1e-12))
            colours = total - pivot_total
            spectral_mag = base_mag + colours
            spectral_wave = colour_wave * u.AA
            spectral_flam = np.asarray([
                magnitude_f_lambda(one_wave, zero_point).to_value(u.erg / (u.s * u.cm**2 * u.AA))
                for one_wave, zero_point in zip(spectral_wave, BAND_VEGA_ZEROPOINT_JY)
            ]) * 10.0**(-0.4 * spectral_mag)
            models.append({"sky_mag": float(total_mag), "sky_zero_point_jy": vega_profile.zero_point_jy,
                           "sky_at_telescope": True, "aperture_radius_arcsec": aperture,
                           "spectral_sky_mag_offsets": np.column_stack((colour_wave, colours)),
                           "spectral_sky_f_lambda": np.column_stack((colour_wave, spectral_flam)),
                           "moonlight_included": bool(moon_alt >= 0.0)})
        return models

    def _photometry_time_series(self, track, detector, telescope, target_mag, observing_band, reference_band,
                                atmo, sky_models, reference_zero_point_jy, template_mv0,
                                visual_band, visual_zero_point_jy, observing_zero_point_jy,
                                reference_detector_type, visual_detector_type, observing_detector_type, texp, target_snr):
        values = np.full(len(track["jd"]), np.nan)
        results = {}
        label = "Required exposure [s]" if target_snr is not None else "S/N"
        geometry_kwargs = self._source_geometry_kwargs()
        for i, airmass in enumerate(track["airmass_target"]):
            if not np.isfinite(airmass):
                continue
            atmosphere = self._atmosphere_dict(airmass, atmo)
            calculator = PhotometryETC(telescope, detector, atmosphere, sky_models[i])
            if target_snr is not None:
                probe = calculator.compute_photometry_single(
                    self.star_spec, observing_band, self.qe_curve, target_mag, 1.0,
                    reference_zero_point_jy, reference_band, template_mv0, visual_band, visual_zero_point_jy,
                    observing_zero_point_jy, reference_detector_type, visual_detector_type, observing_detector_type,
                    **geometry_kwargs)
                real_texp = exposure_time_for_snr(target_snr, probe["source_rate_per_s"], probe["sky_rate_per_s"],
                                                   detector.dark_current_e_s_pix * probe["n_pixels"], detector.read_noise_e,
                                                   probe["n_pixels"])
                values[i] = real_texp
            else:
                real_texp = texp
            result = calculator.compute_photometry_single(
                self.star_spec, observing_band, self.qe_curve, target_mag, real_texp,
                reference_zero_point_jy, reference_band, template_mv0, visual_band, visual_zero_point_jy,
                observing_zero_point_jy, reference_detector_type, visual_detector_type, observing_detector_type,
                **geometry_kwargs)
            results[i] = result
            if target_snr is None:
                values[i] = result["snr"]
        return values, results, label

    def _spectroscopy_time_series(self, track, detector, telescope, target_mag, reference_band, observing_band, atmo, sky_models,
                                  reference_zero_point_jy, template_mv0, visual_band, visual_zero_point_jy,
                                  reference_detector_type, visual_detector_type, texp, target_snr, selected_idx):
        values = np.full(len(track["jd"]), np.nan)
        saturation = np.full(len(track["jd"]), "NOT_VISIBLE", dtype=object)
        peak_e = np.full(len(track["jd"]), np.nan)
        max_unsaturated_exptime = np.full(len(track["jd"]), np.nan)
        valid_indices = [i for i, x in enumerate(track["airmass_target"]) if np.isfinite(x)]
        spectra = {}
        reference = float(self.reference_wavelength_var.get())
        resolution = float(self.resolution_var.get())
        slider_indices = self._slider_indices(valid_indices, selected_idx, maximum=12 if resolution >= 50000 else 25)
        label = "Required exposure [s]" if target_snr is not None else f"S/N at {reference:.0f} Å"
        grating_lines = float(self.grating_lines_var.get() or 0.0)
        for i in valid_indices:
            atmosphere = self._atmosphere_dict(track["airmass_target"][i], atmo)
            calculator = SpectroscopyETC(telescope, detector, atmosphere, sky_models[i])
            args = (self.star_spec, resolution, float(self.slit_var.get()), 1.0,
                    (reference * (1.0 - 0.5 / resolution), reference * (1.0 + 0.5 / resolution)), target_mag, self.qe_curve, reference_band,
                    float(self.sampling_var.get()), float(self.extract_var.get()), reference_zero_point_jy,
                    reference_band, template_mv0, visual_band, visual_zero_point_jy,
                    self.spectroscopy_mode_var.get(), float(self.slitless_width_var.get()),
                    float(self.slitless_dispersion_var.get()), float(self.slitless_lsf_var.get()),
                    reference_detector_type, visual_detector_type, observing_band,
                    grating_lines if grating_lines > 0 else None,
                    float(self.grating_distance_var.get()),
                    float(self.grating_efficiency_var.get()),
                    self.slit_orientation_var.get().strip().lower() == "parallactic",
                    True)
            if target_snr is not None:
                probe = calculator.compute_spectroscopy(*args)
                ref_index = int(np.argmin(np.abs(probe["wavelength_aa"].to_numpy() - reference)))
                real_texp = exposure_time_for_snr(target_snr, probe.iloc[ref_index]["photons_source_es"],
                                                   probe.iloc[ref_index]["photons_sky_es"],
                                                   detector.dark_current_e_s_pix * probe.attrs["n_pixels_per_resel"],
                                                   detector.read_noise_e, probe.attrs["n_pixels_per_resel"])
                values[i] = real_texp
            else:
                real_texp = texp
            reference_spectrum = calculator.compute_spectroscopy(*args[:3], real_texp, *args[4:])
            ref_index = int(np.argmin(np.abs(reference_spectrum["wavelength_aa"].to_numpy() - reference)))
            if target_snr is None:
                values[i] = reference_spectrum.iloc[ref_index]["snr"]
            saturation[i] = reference_spectrum.iloc[ref_index]["saturation_flag"]
            peak_e[i] = reference_spectrum.iloc[ref_index]["peak_e_unclipped"]
            max_unsaturated_exptime[i] = reference_spectrum.iloc[ref_index]["max_unsaturated_exptime_s"]
            if i in slider_indices:
                full_args = list(args)
                full_args[4] = (float(self.wlmin_var.get()), float(self.wlmax_var.get()))
                spectra[i] = calculator.compute_spectroscopy(*full_args[:3], real_texp, *full_args[4:])
        return values, spectra, slider_indices, label, saturation, peak_e, max_unsaturated_exptime

    def _build_time_series_dataframe(self, track, snr_values, exposure_values, target_mag, band, atmo,
                                     detector, telescope, sky_models, target_zero_point_jy,
                                     template_mv0=0.0, visual_band=None, visual_zero_point_jy=3631.0,
                                     reference_detector_type=1, visual_detector_type=1,
                                     saturation_flags=None, peak_e_values=None, max_unsaturated_exptime_values=None,
                                     reference_wavelength_aa=None):
        """Build the complete planning series, including band-effective extinction.

        The extinction is evaluated from the actual template, passband, QE and
        atmospheric curve.  It is therefore a band (colour-dependent) term,
        rather than an arbitrary monochromatic coefficient.
        """
        airmass = np.asarray(track["airmass_target"], dtype=float)
        clear_atmo = np.column_stack((atmo[:, 0], np.ones_like(atmo[:, 1], dtype=float)))
        clear_calculator = PhotometryETC(
            telescope, detector, self._atmosphere_dict(1.0, clear_atmo), sky_models[0])
        clear_rate = clear_calculator.compute_photometry_single(
            self.star_spec, band, self.qe_curve, target_mag, 1.0, target_zero_point_jy,
            band, template_mv0, visual_band, visual_zero_point_jy, 3631.0,
            reference_detector_type, visual_detector_type, reference_detector_type)["source_rate_per_s"]
        extinction = np.full(len(airmass), np.nan)
        for i, x in enumerate(airmass):
            if not np.isfinite(x):
                continue
            calculator = PhotometryETC(
                telescope, detector, self._atmosphere_dict(x, atmo), sky_models[i])
            rate = calculator.compute_photometry_single(
                self.star_spec, band, self.qe_curve, target_mag, 1.0, target_zero_point_jy,
                band, template_mv0, visual_band, visual_zero_point_jy, 3631.0,
                reference_detector_type, visual_detector_type, reference_detector_type)["source_rate_per_s"]
            if clear_rate > 0 and rate > 0:
                extinction[i] = -2.5 * np.log10(rate / clear_rate)
        extinction_per_airmass = np.divide(extinction, airmass, out=np.full_like(extinction, np.nan),
                                            where=np.isfinite(airmass) & (airmass > 0))
        frame = pd.DataFrame({
            "datetime_utc": [value.strftime("%Y-%m-%dT%H:%M:%S") for value in track["utc_datetime"]],
            "datetime_local": [value.strftime("%Y-%m-%dT%H:%M:%S") for value in track["local_datetime"]],
            "local_timezone": self._local_timezone_label(),
            "mjd": np.asarray(track["jd"], dtype=float) - 2400000.5,
            "elevation_deg": np.asarray(track["alt_target"], dtype=float),
            "azimuth_deg": np.asarray(track["az_target"], dtype=float),
            "parallactic_angle_deg": np.asarray(track["parallactic_deg"], dtype=float),
            "snr": np.asarray(snr_values, dtype=float),
            "exptime_s": np.asarray(exposure_values, dtype=float),
            "airmass": airmass,
            "sky_mag_arcsec2": [model["sky_mag"] for model in sky_models],
            "sky_zero_point_jy": [model["sky_zero_point_jy"] for model in sky_models],
            "band_extinction_mag": extinction,
            "band_extinction_mag_per_airmass": extinction_per_airmass,
            "apparent_reference_magnitude": target_mag + extinction,
        })
        if saturation_flags is not None:
            frame["saturation_flag"] = np.asarray(saturation_flags, dtype=object)
        if peak_e_values is not None:
            frame["peak_e_unclipped"] = np.asarray(peak_e_values, dtype=float)
        if max_unsaturated_exptime_values is not None:
            frame["max_unsaturated_exptime_s"] = np.asarray(max_unsaturated_exptime_values, dtype=float)
        if reference_wavelength_aa is not None:
            frame.insert(6, "reference_wavelength_aa", float(reference_wavelength_aa))
        return frame

    def _run_etc(self):
        try:
            self.status_var.set("Calculating..."); self.update_idletasks()
            if self.filter_resp_data is None or self.qe_curve is None or self.star_spec is None:
                raise ValueError("Load filter/QE data and select a spectral template first.")
            ra, dec, target_label = self._target_coordinates(); lat = float(self.lat_var.get()); lon = float(self.lon_var.get())
            elev = float(self.elev_var.get()); target_mag = float(self.mag_var.get()); texp = float(self.texp_var.get())
            target_snr_text = self.target_snr_var.get().strip()
            target_snr = float(target_snr_text) if target_snr_text else None
            if not 0 <= ra < 360 or not -90 <= dec <= 90:
                raise ValueError("RA must be in [0,360) and Dec in [-90,90].")
            if texp <= 0: raise ValueError("Exposure time must be positive.")
            if target_snr is not None and target_snr <= 0: raise ValueError("Target S/N must be positive.")
            utc = self._ut_datetime(); jd = Time(utc, scale="utc").jd
            timezone_name = self.timezone_var.get().strip() if self.timezone_source_var.get() == "iana" else None
            offset = float(self.utc_offset_var.get()) if self.timezone_source_var.get() == "offset" else 0.0
            full_track = compute_target_track(ra, dec, lat, lon, jd - 0.75, jd + 0.75, step_min=5,
                                              elev_m=elev, local_utc_offset_h=offset, timezone_name=timezone_name)
            full_idx = int(np.argmin(np.abs(full_track["jd"] - jd)))
            track, idx = self._observing_window(full_track, full_idx)
            altitude = track["alt_target"][idx]; airmass = track["airmass_target"][idx]
            if not np.isfinite(airmass):
                raise ValueError(f"Target altitude is {altitude:.2f} deg. ETC calculations and airmass are deliberately disabled below 5 deg.")
            reference_profile = fcat.load_filter_profile(
                self.filter_resp_data, self.reference_band_var.get(), self.mag_system_var.get())
            visual_profile = fcat.load_filter_profile(self.filter_resp_data, "Bessell.V", "Vega")
            observing_profile = fcat.load_filter_profile(self.filter_resp_data, self.band_var.get(), self.mag_system_var.get())
            observing_vega_profile = fcat.load_filter_profile(self.filter_resp_data, self.band_var.get(), "Vega")
            reference_band = reference_profile.transmission
            visual_band = visual_profile.transmission
            observing_band = observing_profile.transmission
            atmo = self.earth_atmosphere_curve if self.earth_atmosphere_curve is not None else fcat.generic_zenith_atmosphere_curve()
            detector = Detector(float(self.pixel_var.get()), float(self.gain_var.get()), float(self.fullwell_var.get()),
                                int(self.bitdepth_var.get()), float(self.readnoise_var.get()), float(self.dark_var.get()))
            telescope = {"diameter_mm": float(self.diam_var.get()), "obstruction_mm": float(self.obstruct_var.get()),
                         "efficiency": float(self.eff_var.get()), "focal_length_mm": float(self.focal_var.get()),
                         "throughput_curve": self.throughput_curve,
                         "slit_resolution_curve": self.slit_resolution_curve}
            sky_models = self._sky_models_for_track(track, observing_vega_profile, ra, dec)
            self._show_info(ra, dec, target_label, utc, altitude, airmass, track, idx,
                            sky_mag_arcsec2=sky_models[idx]["sky_mag"])
            if self.mode_var.get() == "photometry":
                values, results, label = self._photometry_time_series(
                    track, detector, telescope, target_mag, observing_band, reference_band, atmo, sky_models,
                    reference_profile.zero_point_jy, self.selected_star.mv0, visual_band,
                    visual_profile.zero_point_jy, observing_profile.zero_point_jy,
                    reference_profile.detector_type, visual_profile.detector_type, observing_profile.detector_type,
                    texp, target_snr)
                result = results[idx]
                frame = pd.DataFrame([result]); self._display_table(frame, "mag"); self.result_df = frame
                plot_args = ("photometry", track, values, label, idx, self.band_var.get())
                snr_values = values if target_snr is None else np.where(np.isfinite(values), target_snr, np.nan)
                exposure_values = np.where(np.isfinite(values), texp, np.nan) if target_snr is None else values
                saturation = np.array([results[i]["saturation_flag"] if i in results else "NOT_VISIBLE"
                                       for i in range(len(track["jd"]))], dtype=object)
                selected_saturation = saturation[idx]
                peak_e = np.array([results[i]["peak_e_unclipped"] if i in results else np.nan
                                   for i in range(len(track["jd"]))], dtype=float)
                max_unsaturated = np.array([results[i]["max_unsaturated_exptime_s"] if i in results else np.nan
                                            for i in range(len(track["jd"]))], dtype=float)
                self.time_series_df = self._build_time_series_dataframe(
                    track, snr_values, exposure_values, target_mag, reference_band, atmo, detector, telescope, sky_models,
                    reference_profile.zero_point_jy, self.selected_star.mv0, visual_band,
                    visual_profile.zero_point_jy, reference_profile.detector_type, visual_profile.detector_type,
                    saturation_flags=saturation, peak_e_values=peak_e,
                    max_unsaturated_exptime_values=max_unsaturated)
            else:
                values, spectra, slider_indices, label, saturation, peak_e, max_unsaturated = self._spectroscopy_time_series(
                    track, detector, telescope, target_mag, reference_band, observing_band, atmo, sky_models,
                    reference_profile.zero_point_jy, self.selected_star.mv0, visual_band,
                    visual_profile.zero_point_jy, reference_profile.detector_type, visual_profile.detector_type,
                    texp, target_snr, idx)
                spectrum = spectra[idx]
                self._display_table(spectrum, "wavelength_aa"); self.result_df = spectrum
                plot_args = ("spectroscopy", track, spectra, values, label, idx, slider_indices)
                snr_values = values if target_snr is None else np.where(np.isfinite(values), target_snr, np.nan)
                exposure_values = np.where(np.isfinite(values), texp, np.nan) if target_snr is None else values
                self.time_series_df = self._build_time_series_dataframe(
                    track, snr_values, exposure_values, target_mag, reference_band, atmo, detector, telescope, sky_models,
                    reference_profile.zero_point_jy, self.selected_star.mv0, visual_band,
                    visual_profile.zero_point_jy, reference_profile.detector_type, visual_profile.detector_type,
                    saturation_flags=saturation, peak_e_values=peak_e,
                    max_unsaturated_exptime_values=max_unsaturated,
                    reference_wavelength_aa=float(self.reference_wavelength_var.get()))
                selected_saturation = saturation[idx]
            if plot_args[0] == "photometry":
                self.plot_window = show_photometry_plot(self, *plot_args[1:])
            else:
                self.plot_window = show_spectroscopy_plot(self, *plot_args[1:])
            self._show_results_window()
            if target_snr is not None and selected_saturation != "NONE":
                self.status_var.set("Complete - target S/N exposure saturates; see max safe exposure in Results / CSV.")
            else:
                self.status_var.set("Complete")
            self._save_config()
#        except Exception as exc:
#            self.status_var.set("Error")
#            messagebox.showerror("ETC error", f"{exc}\n\n{traceback.format_exc()}")

        except ValueError as exc:
            self.status_var.set("Error")
            messagebox.showerror("Cannot run ETC", str(exc))
        
        except Exception as exc:
            self.status_var.set("Error")
            messagebox.showerror("Unexpected ETC error", str(exc))
    
    def _ensure_results_window(self):
        if self.results_window is not None and self.results_window.winfo_exists():
            return
        window = tk.Toplevel(self)
        window.title("ETC numerical results")
        window.geometry("1350x720")
        window.minsize(850, 500)
        window.protocol("WM_DELETE_WINDOW", self._hide_results_window)
        self.results_window = window
        tabs = ttk.Notebook(window)
        tabs.pack(fill="both", expand=True, padx=6, pady=6)

        result_frame = ttk.Frame(tabs)
        time_frame = ttk.Frame(tabs)
        info_frame = ttk.Frame(tabs)
        tabs.add(result_frame, text="Selected-time result")
        tabs.add(time_frame, text="Time series")
        #tabs.add(info_frame, text="Assumptions / visibility")
        #tabs.hide(info_frame)

        columns = ("X", "Source e-/s", "Sky e-/s", "S/N", "Peak ADU", "Sat")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings")
        for column, width in zip(columns, [130, 150, 150, 120, 120, 70]):
            self.tree.heading(column, text=column)
            self.tree.column(column, width=width, anchor="e")
        result_scroll_y = ttk.Scrollbar(result_frame, orient="vertical", command=self.tree.yview)
        result_scroll_x = ttk.Scrollbar(result_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=result_scroll_y.set, xscrollcommand=result_scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        result_scroll_y.grid(row=0, column=1, sticky="ns")
        result_scroll_x.grid(row=1, column=0, sticky="ew")
        result_frame.columnconfigure(0, weight=1); result_frame.rowconfigure(0, weight=1)

        self.time_tree = ttk.Treeview(time_frame, show="headings")
        time_scroll_y = ttk.Scrollbar(time_frame, orient="vertical", command=self.time_tree.yview)
        time_scroll_x = ttk.Scrollbar(time_frame, orient="horizontal", command=self.time_tree.xview)
        self.time_tree.configure(yscrollcommand=time_scroll_y.set, xscrollcommand=time_scroll_x.set)
        self.time_tree.grid(row=0, column=0, sticky="nsew")
        time_scroll_y.grid(row=0, column=1, sticky="ns")
        time_scroll_x.grid(row=1, column=0, sticky="ew")
        time_frame.columnconfigure(0, weight=1); time_frame.rowconfigure(0, weight=1)

        self.info_text = scrolledtext.ScrolledText(info_frame, bg="#f6f6f6", wrap="word")
        self.info_text.pack(fill="both", expand=True)

    def _hide_results_window(self):
        if self.results_window is not None and self.results_window.winfo_exists():
            self.results_window.withdraw()

    @staticmethod
    def _format_table_value(value):
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.6g}"
        return str(value)

    def _populate_time_table(self):
        if self.time_tree is None or self.time_series_df is None:
            return
        columns = tuple(self.time_series_df.columns)
        self.time_tree.configure(columns=columns)
        for column in columns:
            width = 175 if column.startswith("datetime") else 130
            self.time_tree.heading(column, text=column)
            self.time_tree.column(column, width=width, minwidth=90, anchor="e")
        self.time_tree.delete(*self.time_tree.get_children())
        for _, row in self.time_series_df.iterrows():
            self.time_tree.insert("", "end", values=[self._format_table_value(row[column]) for column in columns])

    def _show_results_window(self):
        self._ensure_results_window()
        self._populate_time_table()
        self.results_window.deiconify()
        self.results_window.lift()
        self.results_window.focus_force()

    def _show_info(self, ra, dec, target_label, utc, altitude, airmass, track, idx,
                   sky_mag_arcsec2=None):
        self._ensure_results_window()
        local = track["local_datetime"][idx]
        parallactic = float(track["parallactic_deg"][idx])
        sky_line = ("" if sky_mag_arcsec2 is None
                    else f"Sky surface brightness at selected time: {sky_mag_arcsec2:.2f} mag/arcsec2 (observing band)\n")
        text = f"""PHYSICAL ASSUMPTIONS
--------------------
Target: {target_label}; ICRS RA {ra:.6f} deg, Dec {dec:.6f} deg
Selected UTC: {utc:%Y-%m-%d %H:%M}; local display: {local:%Y-%m-%d %H:%M} ({self._local_timezone_label()})
Altitude: {altitude:.2f} deg; airmass: {airmass:.3f} (Pickering 2002 on the
refraction-corrected apparent altitude; not defined below 5 deg)
Parallactic angle: {parallactic:+.2f} deg (N through E; slit at this angle avoids
atmospheric-dispersion slit losses)
{sky_line}Azimuth convention: N=0, E=90, S=180, W=270 deg

Target reference magnitude: {self.mag_var.get()} {self.mag_system_var.get()} in {self.reference_band_var.get()}.
Observing filter: {self.band_var.get()} (used in photometry and spectroscopy). The template file
is a calibrated F_lambda distribution represented at visual mv0={self.selected_star.mv0:.3f};
it is first converted to visual zero as in interpola_spad, then scaled to the
target reference measurement. The observing response is applied afterwards.
Sky model: {self.sky_model_var.get()}. ING mode combines the dark-sky table
with van Rhijn airglow (solar-cycle scaled), zodiacal light at the field's
ecliptic latitude, integrated starlight at its galactic latitude, the
Krisciunas & Schaefer (1991) Moon model, twilight blending and the Weaver
daylight table; fixed_ab uses the manually entered observed ground sky.
PSF model: {self.psf_model_var.get()} (Moffat beta {self.moffat_beta_var.get()} where selected).

Detected counts include collecting area pi/4(D²-d²), scalar and optional
wavelength-dependent optics throughput, QE, and the loaded Earth-atmosphere
transmission curve (or the broad-band fallback) raised to the current airmass,
source aperture or slit losses, sky, dark current and read noise, plus
Young-law scintillation and ADC quantization (gain/sqrt(12)) noise terms.
'fixed' slit orientation additionally applies the per-wavelength Filippenko
(1982) atmospheric-dispersion slit loss. Sky is
already an observed ground brightness and is therefore not extinguished a
second time. Spectroscopic values are per resolution element. The native
Matplotlib spectrum slider is manual and has no autoplay.

SATURATION / TARGET-S/N POLICY
------------------------------
Peak-pixel electrons use the untruncated Gaussian PSF (and the spectral LSF
for spectra), while S/N uses the selected photometric aperture or extraction.
The saturation flag distinguishes detector full well from ADC clipping. CSV
output reports the maximum unsaturated single-frame exposure at each time;
split a longer required exposure into shorter frames and include one read-noise
term for each frame when planning a stack.
{('Slitless mode is EXPERIMENTAL: its dispersion/LSF geometry is internally consistent, but must be validated against a calibrated instrument.' if self.mode_var.get() == 'spectroscopy' and self.spectroscopy_mode_var.get() == 'slitless' else '')}
"""
        self.info_text.delete("1.0", tk.END); self.info_text.insert("1.0", text)

    def _display_table(self, frame, xcol):
        self._ensure_results_window()
        self.tree.delete(*self.tree.get_children())
        self.tree.heading("X", text=f"Magnitude [{self.mag_system_var.get()}]" if xcol == "mag" else "Wavelength [Å]")
        for _, row in frame.iterrows():
            x_value = f"{row[xcol]:.3f}" if xcol == "mag" else f"{row[xcol]:.2f}"
            saturation = row.get("saturation_flag", "SATURATED" if row["saturated"] else "")
            self.tree.insert("", "end", values=(x_value, f"{row['photons_source_es']:.3e}",
                                                   f"{row['photons_sky_es']:.3e}", f"{row['snr']:.3f}",
                                                   str(row["adu"]), saturation if row["saturated"] else ""))

    def _open_plot(self):
        if self.plot_window is None or not self.plot_window.winfo_exists():
            messagebox.showinfo("No plot", "Run the ETC first."); return
        self.plot_window.deiconify(); self.plot_window.lift(); self.plot_window.focus_force()

    def _export_csv(self):
        if self.result_df is None:
            messagebox.showinfo("No result", "Run the ETC first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path: self.result_df.to_csv(path, index=False)

    def _export_time_series_csv(self):
        if self.time_series_df is None:
            messagebox.showinfo("No time series", "Run the ETC first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="etc_time_series.csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            self.time_series_df.to_csv(path, index=False)

    def _on_exit(self):
        self._save_config(); self.destroy(); sys.exit(0)


if __name__ == "__main__":
    ETCGUI().mainloop()
