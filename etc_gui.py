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
from spectroscopy import (SpectroscopyETC, slit_spectrograph_resolving_power,
                          grating_dispersion_aa_per_pixel)
from detector import Detector, load_qe_curve, load_transmission_curve, load_gain_table
from spectral_utils import load_fits_transmission_curve, interpolate_zero_filled
from solvers import exposure_time_for_snr, plan_stack
from config_manager import OBSERVATORY_PRESETS
import filter_catalog as fcat
import star_catalog as scat
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib_plots import show_photometry_plot, show_spectroscopy_plot
from sky_background import sky_magnitude_vega, SKY_WAVELENGTH_AA, SKY_MAG_VEGA
from observing_conditions import (scintillation_variance_rate_e2_s, effective_seeing_arcsec,
                                  scintillation_fractional_rms)
from sky_brightness import sky_brightness_total, BAND_WAVELENGTH_NM, BAND_VEGA_ZEROPOINT_JY
from etc_physics import synthetic_magnitude, magnitude_f_lambda, transformed_template


CONFIG_FILE = Path(__file__).with_name("etc_user_config.json")
PROFILE_SCHEMA_VERSION = 1


class ETCGUI(tk.Tk):
    """The instrument configuration is automatically read/written as JSON."""

    def __init__(self):
        super().__init__()
        self.title("SPETC v10.0 - Spectro-Photometry Exposure Time Calculator")
        self.geometry("1850x980")
        self.minsize(1200, 720)
        self._saved = self._load_config()
        self._program_start_utc = datetime.now(timezone.utc)
        self.session_dir = None
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
        self.grating_efficiency_curve = None
        self.grating_efficiency_source_path = None
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

    def _config_dict(self):
        data = {key: var.get() for key, var in self._vars.items()}
        data["schema_version"] = 9
        data["active_instrument_profile"] = str(self.active_profile_path)
        return data

    def _save_config(self):
        data = self._config_dict()
        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
        except OSError as exc:
            self.status_var.set(f"Configuration not saved: {exc}")
        # Mirror the configuration into the active session directory.
        if self.session_dir is not None:
            try:
                with (self.session_dir / "config.json").open("w", encoding="utf-8") as stream:
                    json.dump(data, stream, indent=2, sort_keys=True)
            except OSError:
                pass

    def _create_session(self):
        """Create output/<session name>/ and record all outputs there from now on."""
        name = self.session_name_var.get().strip()
        if not name:
            messagebox.showerror("Session", "Enter a session name first.")
            return
        # Keep the name filesystem-safe.
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
        session_dir = Path(__file__).with_name("output") / safe
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Session", f"Could not create the session directory:\n{exc}")
            return
        self.session_dir = session_dir
        self.session_name_var.set(safe)
        self._save_config()
        self.session_status_var.set(f"Recording to output/{safe}/ — config, results and plots are saved here.")
        self.status_var.set(f"Session created: output/{safe}/")
        messagebox.showinfo(
            "Session created",
            f"From now on all outputs are recorded in:\n\noutput/{safe}/\n\n"
            "Each run writes the configuration, the selected-time result, the time series, "
            "and the plots (PNG) into this directory.")

    def _save_session_outputs(self):
        """Write the current run's config, tables and plots into the session dir."""
        if self.session_dir is None:
            return
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        try:
            if self.result_df is not None:
                self.result_df.to_csv(self.session_dir / f"result_{stamp}.csv", index=False)
            if self.time_series_df is not None:
                self.time_series_df.to_csv(self.session_dir / f"time_series_{stamp}.csv", index=False)
            if self.plot_window is not None and self.plot_window.winfo_exists():
                self.plot_window.figure.savefig(self.session_dir / f"plots_{stamp}.png", dpi=120)
        except (OSError, ValueError) as exc:
            self.status_var.set(f"Session outputs partly saved: {exc}")

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
        # Apply the initial photometry/spectroscopy and slit/slitless greying
        # now that every widget and status var exists.
        self._on_calculation_mode_changed()
        self._update_throughput_exclusive()
        self._update_detector_tech_state()

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
        # ------------------------------ SESSION ----------------------------
        # A session names an output/<session> directory into which every
        # config, result table, and plot PNG of subsequent runs is written.
        f = self._section(holder, "SESSION")
        default_session = "spetc_" + self._program_start_utc.strftime("%Y%m%dT%H%M")
        self.session_name_var = tk.StringVar(value=default_session)
        ttk.Entry(f, textvariable=self.session_name_var).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        ttk.Button(f, text="Create new session", command=self._create_session).grid(
            row=1, column=0, columnspan=2, sticky="ew")
        self.session_status_var = tk.StringVar(value="No session yet — outputs are not being recorded.")
        ttk.Label(f, textvariable=self.session_status_var, foreground="#1f4f82", wraplength=320,
                  justify="left").grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        f = self._section(holder, "OBSERVATION DATE")

        self.date_var = self._var("date", "")
        self.time_var = self._var("time", "")
            
        self.time_ref_var = self._var("time_reference", "UT")

        if not self.date_var.get().strip():
            self.date_var.set(datetime.now().strftime("%Y-%m-%d"))
        
        if not self.time_var.get().strip():
            self.time_var.set("00:00")
    
        self.utc_offset_var = self._var("utc_offset_h", "1")
        self.timezone_var = self._var("timezone", "Europe/Rome")
        self.timezone_source_var = self._var("timezone_source", "iana")
        if "timezone_source" not in self._saved:
            self.timezone_source_var.set("iana" if self.timezone_var.get().strip() else "offset")
        # Compact single row: Date [entry] Time [entry] [UT/local menu].
        datetime_row = ttk.Frame(f)
        datetime_row.grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(datetime_row, text="Date").pack(side="left")
        date_entry = ttk.Entry(datetime_row, textvariable=self.date_var, width=11)
        date_entry.pack(side="left", padx=(3, 8))
        
        ttk.Label(datetime_row, text="Time").pack(side="left")
        time_entry = ttk.Entry(datetime_row, textvariable=self.time_var, width=6)
        time_entry.pack(side="left", padx=(3, 8))
        
        def add_placeholder(entry, variable, placeholder):
            label = ttk.Label(entry, text=placeholder, foreground="grey")
            label.place(relx=0.04, rely=0.5, anchor="w")
        
            def refresh(*_):
                if variable.get().strip() or entry.focus_get() is entry:
                    label.place_forget()
                else:
                    label.place(relx=0.04, rely=0.5, anchor="w")
        
            def focus_entry(_event=None):
                entry.focus_set()
                label.place_forget()
        
            label.bind("<Button-1>", focus_entry)
            entry.bind("<FocusIn>", focus_entry)
            entry.bind("<FocusOut>", refresh)
            variable.trace_add("write", refresh)
            refresh()
        
        add_placeholder(date_entry, self.date_var, "YYYY-MM-DD")
        add_placeholder(time_entry, self.time_var, "HH:MM")
        ttk.Combobox(datetime_row, textvariable=self.time_ref_var, state="readonly", width=6,
                     values=("local", "UT")).pack(side="left")
        source_frame = ttk.Frame(f)
        source_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(source_frame, text="Local-time source:").pack(side="left")
        ttk.Radiobutton(source_frame, text="IANA timezone", variable=self.timezone_source_var, value="iana",
                        command=self._update_timezone_source_state).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(source_frame, text="UTC offset", variable=self.timezone_source_var, value="offset",
                        command=self._update_timezone_source_state).pack(side="left", padx=(6, 0))
        ttk.Label(f, text="IANA timezone:").grid(row=2, column=0, sticky="w")
        self.timezone_combo = ttk.Combobox(f, textvariable=self.timezone_var, state="readonly", width=25,
                                           values=tuple(sorted(available_timezones())))
        self.timezone_combo.grid(row=2, column=1, sticky="ew", pady=1)
        ttk.Label(f, text="UTC offset (h):").grid(row=3, column=0, sticky="w")
        self.utc_offset_entry = ttk.Entry(f, textvariable=self.utc_offset_var, width=20)
        self.utc_offset_entry.grid(row=3, column=1, sticky="ew", pady=1)
        self._update_timezone_source_state()

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
        # Save / import / open-in-Google-Earth on a single row.
        site_buttons = ttk.Frame(f)
        site_buttons.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 1))
        for i in range(3):
            site_buttons.columnconfigure(i, weight=1)
        ttk.Button(site_buttons, text="Save site", command=self._save_current_site).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(site_buttons, text="Import…", command=self._import_site_file).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(site_buttons, text="Google Earth", command=self._open_google_earth).grid(row=0, column=2, sticky="ew", padx=(2, 0))
        self.horizon_radius_var = self._var("horizon_radius", "10")
        self.horizon_unit_var = self._var("horizon_radius_unit", "km")
        ttk.Label(f, text="Horizon radius (1-100 km):").grid(row=6, column=0, sticky="w")
        horizon_row = ttk.Frame(f); horizon_row.grid(row=6, column=1, sticky="ew")
        horizon_row.columnconfigure(0, weight=1)
        ttk.Entry(horizon_row, textvariable=self.horizon_radius_var, width=7).grid(row=0, column=0, sticky="ew")
        ttk.Combobox(horizon_row, textvariable=self.horizon_unit_var, state="readonly", width=6,
                     values=("km", "miles")).grid(row=0, column=1, padx=(3, 0))
        # Generate / display horizon on a single row.
        horizon_buttons = ttk.Frame(f)
        horizon_buttons.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(3, 1))
        horizon_buttons.columnconfigure(0, weight=1); horizon_buttons.columnconfigure(1, weight=1)
        self.horizon_generate_button = ttk.Button(horizon_buttons, text="Generate horizon (DEM)…",
                                                  command=self._generate_horizon)
        self.horizon_generate_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(horizon_buttons, text="Display horizon", command=self._display_horizon).grid(
            row=0, column=1, sticky="ew", padx=(2, 0))
        self.horizon_status_var = tk.StringVar(
            value="Horizon from Copernicus GLO-30 (internet needed once; tiles cached). CSV+PNG saved automatically.")
        ttk.Label(f, textvariable=self.horizon_status_var, foreground="gray35", wraplength=320,
                  justify="left").grid(row=9, column=0, columnspan=2, sticky="w")

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
        self.target_resolution_status = tk.StringVar(value="Enter a name to resolve via SIMBAD query")
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
#        ttk.Label(f, text="The observing filter is independent.", foreground="gray35", wraplength=320, justify="left").grid(row=9, column=0, columnspan=2, sticky="w", pady=(4, 0))
#        ttk.Label(f, text="The template flux is scaled from its stored Vmag to this measurement. The observing filter is independent.", foreground="gray35", wraplength=320, justify="left").grid(row=9, column=0, columnspan=2, sticky="w", pady=(4, 0))
        # Radial velocity and interstellar reddening are properties of the
        # target, entered here and applied to the selected template before
        # calibration (the entered magnitude stays the observed one).
        self.radial_velocity_var = self._var("radial_velocity_kms", "0")
        self.ebv_var = self._var("ebv_mag", "0")
        self._entry(f, 10, "Radial velocity (km/s):", self.radial_velocity_var)
        self._entry(f, 11, "Reddening E(B-V) (mag):", self.ebv_var)
#        ttk.Label(f, text="Target RV shifts the template by (1+v/c); E(B-V) reddens it with "
#                          "CCM89 (R_V=3.1)",
#                  foreground="gray35", wraplength=320, justify="left").grid(
#                  row=12, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _build_instrument_column(self, holder):
        # ---------------------------- TELESCOPE ----------------------------
        f = self._section(holder, "TELESCOPE")
        self.diam_var = self._var("diameter_mm", "358")
        self.obstruct_var = self._var("obstruction_mm", "87.5")
        self.eff_var = self._var("throughput_excluding_qe", "0.70")
        self.focal_var = self._var("focal_length_mm", "2000")
        self.throughput_path_var = self._var("throughput_curve_path", "")
        self.throughput_unit_var = self._var("throughput_wavelength_unit", "AA")
        for row, label, var in [(0, "Primary diameter (mm):", self.diam_var),
                                (1, "Obstruction diameter (mm):", self.obstruct_var),
                                (2, "Focal length (mm):", self.focal_var)]:
            self._entry(f, row, label, var)
        # Optics throughput moved one row down (now below the focal length).
        ttk.Label(f, text="Optics throughput:").grid(row=3, column=0, sticky="w")
        self.throughput_scalar_entry = ttk.Entry(f, textvariable=self.eff_var, width=20)
        self.throughput_scalar_entry.grid(row=3, column=1, sticky="ew", pady=1)
        # Calibrated optics response: only a Browse button plus the wavelength
        # unit selector (no path field, no unit label).  A loaded response
        # curve and the scalar throughput are mutually exclusive.


        # ttk.Label(f, text="OptTroug File:").grid(row=4, column=0, sticky="w")
        # curve_row = ttk.Frame(f); curve_row.grid(row=4, column=1, sticky="ew")
        # self.throughput_browse_button = ttk.Button(curve_row, text="Browse…", command=self._choose_throughput_curve)
        # self.throughput_browse_button.grid(row=0, column=0, sticky="w")
        # ttk.Combobox(curve_row, textvariable=self.throughput_unit_var, state="readonly",
                     # values=("angstrom", "nm", "um"), width=8).grid(row=0, column=1, padx=(4, 0))
        # ttk.Button(curve_row, text="Clear", command=self._clear_throughput_curve).grid(row=0, column=2, padx=(4, 0))
        # self.throughput_curve_status = tk.StringVar(value="response curve: none (using scalar)")
        # ttk.Label(f, textvariable=self.throughput_curve_status, foreground="#1f4f82",
                  # wraplength=300).grid(row=5, column=0, columnspan=2, sticky="w")


        curve_row = ttk.Frame(f)
        curve_row.grid(row=4, column=0, columnspan=2, sticky="w")
        
        ttk.Label(curve_row, text="OptThroughput file:").grid(row=0, column=0, sticky="w")
        
        self.throughput_browse_button = ttk.Button(
            curve_row, text="Browse…", command=self._choose_throughput_curve
        )
        self.throughput_browse_button.grid(row=0, column=1, sticky="w", padx=(4, 0))
        
        ttk.Combobox(
            curve_row, textvariable=self.throughput_unit_var, state="readonly",
            values=("AA", "nm", "um"), width=8
        ).grid(row=0, column=2, padx=(4, 0))
        
        ttk.Button(
            curve_row, text="Clear", command=self._clear_throughput_curve
        ).grid(row=0, column=3, padx=(4, 0))
        self.throughput_curve_status = tk.StringVar(value="response curve: none (using scalar)")
        ttk.Label(f, textvariable=self.throughput_curve_status, foreground="#1f4f82",
                  wraplength=300).grid(row=5, column=0, columnspan=2, sticky="w")
    


        # ----------------------------- DETECTOR ----------------------------
        f = self._section(holder, "DETECTOR")
        self.pixel_var = self._var("pixel_size_um", "13.5")
        self.gain_var = self._var("gain_e_adu", "2.5")
        self.fullwell_var = self._var("full_well_e", "80000")
        self.bitdepth_var = self._var("bit_depth", "16")
        self.readnoise_var = self._var("read_noise_e", "5.0")
        self.dark_var = self._var("dark_current_e_s_pix", "0.0")
        self.qe_unit_var = self._var("qe_wavelength_unit", "AA")
        for row, label, var in [(0, "Pixel size (um):", self.pixel_var), (1, "Gain (e-/ADU):", self.gain_var),
                                (2, "Full well (e-):", self.fullwell_var), (3, "ADC bits:", self.bitdepth_var),
                                (4, "Read noise (e- rms/pix):", self.readnoise_var),
                                (5, "Dark current (e-/s/pix):", self.dark_var)]:
            self._entry(f, row, label, var)
        # Detector dimensions in pixels: L (length, dispersion direction) and
        # W (width, cross-dispersion), used to check whether a spectrum and
        # its diffraction orders fit on the chip.  Both must be > 0.
        self.det_length_var = self._var("detector_length_pix", "1024")
        self.det_width_var = self._var("detector_width_pix", "1024")
        ttk.Label(f, text="Detector L x W (pixels):").grid(row=6, column=0, sticky="w")
        lw_row = ttk.Frame(f); lw_row.grid(row=6, column=1, sticky="ew")
        ttk.Label(lw_row, text="L").pack(side="left")
        ttk.Entry(lw_row, textvariable=self.det_length_var, width=7).pack(side="left", padx=(2, 8))
        ttk.Label(lw_row, text="W").pack(side="left")
        ttk.Entry(lw_row, textvariable=self.det_width_var, width=7).pack(side="left", padx=(2, 0))
        # Quantum efficiency is loaded on demand from a two-column CSV/text
        # file (wavelength, QE) rather than silently from data/qe.dat.
        ttk.Label(f, text="Quantum efficiency (QE):").grid(row=7, column=0, sticky="w")
        qe_row = ttk.Frame(f); qe_row.grid(row=7, column=1, sticky="ew")
        qe_row.columnconfigure(0, weight=1)
        self.qe_status_var = tk.StringVar(value="no QE loaded")
        ttk.Label(qe_row, textvariable=self.qe_status_var, foreground="#1f4f82").grid(row=0, column=0, sticky="w")
        ttk.Button(qe_row, text="Load QE CSV…", command=self._choose_qe_curve).grid(row=0, column=1, padx=(3, 0))
        ttk.Label(f, text="QE wavelength unit:").grid(row=8, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.qe_unit_var, state="readonly",
                     values=("AA", "nm", "um"), width=10).grid(row=8, column=1, sticky="ew")
        # Non-linearity limit in ADU: the maximum count before the response
        # departs from linear.  Empty means the ADC ceiling (2^bits), i.e. no
        # non-linear regime is flagged; a value 0 < v <= 2^bits flags counts
        # above it as NON LIN (below hard saturation) in the result tables.
        self.nonlin_limit_var = self._var("nonlinearity_limit_adu", "")
        ttk.Label(f, text="Non-linearity limit (ADU):").grid(row=9, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.nonlin_limit_var, width=20).grid(row=9, column=1, sticky="ew", pady=1)
        # Sensor: detector technology (CCD/CMOS) and mono vs one-shot-colour.
        # The CMOS gain table is only active for a CMOS detector.
        self.detector_tech_var = self._var("detector_technology", "CMOS")
        self.sensor_type_var = self._var("sensor_type", "mono")
        if self.sensor_type_var.get() in ("monochrome", "osc"):  # migrate old values
            self.sensor_type_var.set({"monochrome": "mono", "osc": "color"}[self.sensor_type_var.get()])
        self.osc_channel_var = self._var("osc_channel", "G")
        ttk.Label(f, text="Sensor:").grid(row=10, column=0, sticky="w")
        sensor_row = ttk.Frame(f); sensor_row.grid(row=10, column=1, sticky="ew")
        self.detector_tech_combo = ttk.Combobox(sensor_row, textvariable=self.detector_tech_var,
                                                state="readonly", width=5, values=("CCD", "CMOS"))
        self.detector_tech_combo.pack(side="left")
        self.sensor_type_combo = ttk.Combobox(sensor_row, textvariable=self.sensor_type_var, state="readonly",
                                               width=6, values=("mono", "color"))
        self.sensor_type_combo.pack(side="left", padx=(3, 0))
        ttk.Label(sensor_row, text=" CH:").pack(side="left")
        self.osc_channel_combo = ttk.Combobox(sensor_row, textvariable=self.osc_channel_var, state="readonly",
                                               width=3, values=("R", "G", "B"))
        self.osc_channel_combo.pack(side="left")
        self.sensor_type_var.trace_add("write", lambda *_: self._update_sensor_channel_state())
        self.detector_tech_var.trace_add("write", lambda *_: self._update_detector_tech_state())
        self._update_sensor_channel_state()
        self.gain_table_path_var = self._var("gain_table_path", "")
        self.gain_setting_var = tk.StringVar(value="")
        self.gain_table_rows = []
        self.gain_table_label = ttk.Label(f, text="CMOS gain table:")
        self.gain_table_label.grid(row=12, column=0, sticky="w")
        gain_row = ttk.Frame(f); gain_row.grid(row=12, column=1, sticky="ew")
        gain_row.columnconfigure(0, weight=1)
        self.gain_setting_combo = ttk.Combobox(gain_row, textvariable=self.gain_setting_var,
                                               state="disabled", width=8, values=())
        self.gain_setting_combo.grid(row=0, column=0, sticky="ew")
        self.gain_setting_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_gain_setting())
        self.gain_table_browse = ttk.Button(gain_row, text="Browse…", command=self._choose_gain_table)
        self.gain_table_browse.grid(row=0, column=1, padx=(3, 0))
        self._update_detector_tech_state()
 #       ttk.Label(f, text="Gain table rows: setting, e-/ADU, read noise e-, full well e-. Selecting a"
 #                         " setting fills the three detector fields.",
 #                 foreground="gray35", wraplength=300, justify="left").grid(row=13, column=0, columnspan=2, sticky="w")
        # Instrument profile save/load lives at the end of the detector box,
        # both buttons on one row.
        self.instrument_profile_section = f
        self.profile_status = tk.StringVar(value="Save/Load bundles the telescope, detector, QE and response curves.")
        ttk.Label(f, textvariable=self.profile_status, foreground="blue", wraplength=300,
                  justify="left").grid(row=14, column=0, columnspan=2, sticky="w", pady=(4, 0))
        profile_row = ttk.Frame(f); profile_row.grid(row=15, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        profile_row.columnconfigure(0, weight=1); profile_row.columnconfigure(1, weight=1)
        ttk.Button(profile_row, text="Save profile…", command=self._save_instrument_profile).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(profile_row, text="Load profile…", command=self._load_instrument_profile).grid(row=0, column=1, sticky="ew", padx=(2, 0))

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
        ttk.Combobox(f, textvariable=self.sky_model_var, state="readonly", values=("ing", "fixed_ab", "sqm")).grid(row=1, column=1, sticky="ew")
        # A single sky-brightness input serves both the fixed-AB and the SQM
        # modes (both are a mag/arcsec2 surface brightness); it is interpreted
        # in the system implied by the selected sky-background mode.
        self._entry(f, 2, "Sky mag / SQM mag (mag/arcsec2):", self.sky_var)
        self.sqm_var = self.sky_var  # merged input; sqm mode reads the same value


        self.bortle_var = tk.StringVar(value="")
        ttk.Label(f, text="Bortle class preset:").grid(row=3, column=0, sticky="w")
        bortle_combo = ttk.Combobox(f, textvariable=self.bortle_var, state="readonly", width=6,
                                    values=tuple(str(i) for i in range(1, 10)))
        bortle_combo.grid(row=3, column=1, sticky="ew")
        bortle_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_bortle_preset())
#        ttk.Label(f, text="sqm mode: your zenith SQM reading sets the V sky; band colours and the"
#                          " Moon model are applied on top. Bortle preset fills a typical SQM.",
#                  foreground="gray35", wraplength=300, justify="left").grid(row=8, column=0, columnspan=2, sticky="w")






        self.psf_model_var = self._var("psf_model", "gaussian")
        self.moffat_beta_var = self._var("moffat_beta", "2.5")
        self.seeing_scaling_var = self._var("seeing_wavelength_scaling", "0")

        ttk.Label(f, text="PSF model:").grid(row=4, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.psf_model_var, state="readonly",
                     values=("gaussian", "moffat")).grid(row=4, column=1, sticky="ew")
        self._entry(f, 5, "Moffat beta (2.5<beta<4.7):", self.moffat_beta_var)
#        ttk.Label(f, text="ING: pos-dep sky model; fixed_ab: observed sky. Moffat real seeing wings; 2.5<= beta <=4.7",
#                  foreground="gray35", wraplength=300, justify="left").grid(row=5, column=0, columnspan=2, sticky="w")
#        ttk.Label(f, text="ING: position-dependent night/twilight/day model (van Rhijn airglow,\n"
#                          "zodiacal, starlight, Krisciunas-Schaefer Moon); fixed_ab: manual observed sky.\n"
#                          "Moffat reproduces real seeing wings; beta 2.5-4.7 (Gaussian limit).",
#                  foreground="gray35", wraplength=300, justify="left").grid(row=5, column=0, columnspan=2, sticky="w")


        ttk.Checkbutton(f, text="Scale seeing with wavelength and airmass "
                                "(entered seeing = zenith V)", variable=self.seeing_scaling_var,
                        onvalue="1", offvalue="0").grid(row=9, column=0, columnspan=2, sticky="w")





        self.extra_background_var = self._var("extra_background_e_s_pixel", "0")
        self._entry(f, 10, "Extra background (e-/s/pixel):", self.extra_background_var)
#        ttk.Label(f, text="Catch-all background per pixel: detector glow, stray/scattered light, "
#                          "ghosting. 0 = none.", foreground="gray35", wraplength=300,
#                  justify="left").grid(row=11, column=0, columnspan=2, sticky="w")
        self.guiding_rms_var = self._var("guiding_rms_arcsec", "0")
        self._entry(f, 12, "Guiding rms (arcsec, 0=perfect):", self.guiding_rms_var)
        self.telluric_var = self._var("include_telluric_bands", "0")
        ttk.Checkbutton(f, text="Add built-in parametric telluric O2/H2O bands",
                        variable=self.telluric_var, onvalue="1", offvalue="0").grid(
                        row=13, column=0, columnspan=2, sticky="w")
        self.sky_annulus_var = self._var("sky_annulus_pixels", "0")
        self._entry(f, 14, "Sky annulus pixels (0=ideal):", self.sky_annulus_var)
#        ttk.Label(f, text="Telluric bands: a BUILT-IN parametric model (no file needed) of the O2 B/A "
#                          "and H2O bands beyond ~6300 A, scaled with airmass; a smooth approximation "
#                          "for red/NIR S/N, not a measured site spectrum. Guiding rms adds "
#                          "image-motion blur. Sky annulus: pixels used to measure the background; a "
#                          "finite value adds the (1+n_pix/n_sky) sky-subtraction noise.",
#                  foreground="gray35", wraplength=300,
#                  justify="left").grid(row=15, column=0, columnspan=2, sticky="w")

        # The SYSTEM RESPONSE preview is built at the end of the data column
        # (below the stellar-template selector), not here.
        # The instrument-profile save/load buttons now live at the end of the
        # DETECTOR box (built above).
        self.eff_var.trace_add("write", lambda *_: self._update_combined_response_preview())
        self.throughput_unit_var.trace_add("write", lambda *_: self._on_throughput_unit_changed())
        self.qe_unit_var.trace_add("write", lambda *_: self._on_qe_unit_changed())
        self._load_gain_table_from_path(silent=True)

    def _update_sensor_channel_state(self):
        """Grey out the R/G/B channel selector unless a colour sensor is set."""
        if hasattr(self, "osc_channel_combo"):
            is_color = self.sensor_type_var.get().strip().lower() in ("color", "colour", "osc")
            self.osc_channel_combo.configure(state="readonly" if is_color else "disabled")

    def _update_detector_tech_state(self):
        """The CMOS gain table is active only for a CMOS detector."""
        if not hasattr(self, "gain_setting_combo"):
            return
        is_cmos = self.detector_tech_var.get().strip().upper() == "CMOS"
        self.gain_table_browse.configure(state="normal" if is_cmos else "disabled")
        # Keep the setting selector readonly (and only if a table is loaded).
        if not is_cmos:
            self.gain_setting_combo.configure(state="disabled")
        elif self.gain_table_rows:
            self.gain_setting_combo.configure(state="readonly")
        try:
            self.gain_table_label.configure(foreground="" if is_cmos else "#a0a0a0")
        except tk.TclError:
            pass

    def _labeled_entry(self, frame, row, text, variable, width=20):
        """Grid a label + entry like _entry, but return the Entry widget."""
        ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
        entry = ttk.Entry(frame, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=1)
        return entry

    @staticmethod
    def _set_widget_enabled(widget, enabled):
        """Enable/disable an input widget, honouring readonly comboboxes."""
        import tkinter.ttk as _ttk
        if isinstance(widget, _ttk.Combobox):
            widget.configure(state="readonly" if enabled else "disabled")
        else:
            try:
                widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass

    def _set_frame_enabled(self, frame, enabled):
        """Recursively enable/disable every input widget in a frame (labels stay)."""
        import tkinter.ttk as _ttk
        for child in frame.winfo_children():
            if isinstance(child, (ttk.Frame, _ttk.LabelFrame)):
                self._set_frame_enabled(child, enabled)
            elif isinstance(child, (ttk.Entry, _ttk.Combobox, ttk.Button, ttk.Checkbutton,
                                    ttk.Radiobutton, ttk.Scale)):
                self._set_widget_enabled(child, enabled)

    def _update_spectroscopy_mode_state(self):
        """Grey out the fields of the non-selected spectroscopy mode; the whole
        box is disabled entirely when the calculation mode is photometry."""
        if not hasattr(self, "spectroscopy_box"):
            return
        photometry = self.mode_var.get() == "photometry"
        if photometry:
            self._set_frame_enabled(self.spectroscopy_box, False)
            return
        self._set_frame_enabled(self.spectroscopy_box, True)
        is_slit = self.spectroscopy_mode_var.get() == "slit"
        for widget in self._slit_widgets:
            self._set_widget_enabled(widget, is_slit)
        for widget in self._slitless_widgets:
            self._set_widget_enabled(widget, not is_slit)
        self._update_grating_eff_state()

    def _update_grating_eff_state(self):
        """value: scalar entry active, Load button greyed. file: the reverse."""
        if not hasattr(self, "grating_eff_value_entry"):
            return
        use_value = self.grating_efficiency_kind_var.get() == "value"
        self._set_widget_enabled(self.grating_eff_value_entry, use_value)
        self._set_widget_enabled(self.grating_eff_load_button, not use_value)
        if use_value:
            # Selecting the scalar drops any previously loaded curve.
            self.grating_efficiency_curve = None
            self.grating_efficiency_status_var.set("scalar value applied")
        elif self.grating_efficiency_curve is not None:
            name = getattr(self.grating_efficiency_source_path, "name", "curve")
            self.grating_efficiency_status_var.set(f"curve applied: {name}")
        else:
            self.grating_efficiency_status_var.set("choose a CSV with Load CSV…")

    def _update_photometry_geometry_state(self):
        """Grey the photometry fields that do not apply to the chosen geometry.

        point   : aperture radius active; source area, focus position and the
                  defocused-PSF button greyed.
        extended: source area active; aperture radius, focus position and the
                  button greyed.
        defocus : focus position and the button active; aperture radius (set
                  from the PSF window) and source area greyed.
        """
        if not hasattr(self, "aperture_entry"):
            return
        geometry = self.source_geometry_var.get().strip().lower()
        self._set_widget_enabled(self.aperture_entry, geometry == "point")
        self._set_widget_enabled(self.source_area_entry, geometry == "extended")
        self._set_widget_enabled(self.defocus_entry, geometry == "defocus")
        self._set_widget_enabled(self.defocus_button, geometry == "defocus")

    WL_UNIT_TO_AA = {"AA": 1.0, "nm": 10.0, "um": 10000.0}

    def _wl_to_aa(self, value):
        """Convert an entered spectroscopy wavelength to Angstrom via the unit selector."""
        return float(value) * self.WL_UNIT_TO_AA.get(self.wl_unit_var.get(), 1.0)

    def _spectrum_geometry_lines(self, spectrum):
        """Output lines: spectrum length in pixels, visible grating orders and
        their start pixel vs the detector L/W, and the Horne optimal extraction
        width."""
        lines = []
        try:
            lo_aa = self._wl_to_aa(self.wlmin_var.get())
            hi_aa = self._wl_to_aa(self.wlmax_var.get())
            length_pix = float(self.det_length_var.get())
            width_pix = float(self.det_width_var.get())
            pixel_um = float(self.pixel_var.get())
            focal_mm = float(self.focal_var.get())
            mode = spectrum.attrs.get("spectroscopy_mode", "slit")
        except (ValueError, KeyError, tk.TclError):
            return lines
        if length_pix <= 0 or width_pix <= 0:
            return ["detector L/W       : must be > 0 to report spectrum geometry"]
        plate_scale = 206265.0 * (pixel_um * 1e-3) / focal_mm  # arcsec/pixel

        # Spectrum length in pixels over the requested wavelength range.
        if mode == "slitless":
            dispersion = float(spectrum.attrs.get("dispersion_aa_pix", 0.0))
            spectrum_len = (hi_aa - lo_aa) / dispersion if dispersion > 0 else float("nan")
        else:
            resolution = float(self.resolution_var.get())
            sampling = float(self.sampling_var.get())
            # Å/pixel = (lambda/R)/sampling, so pixels = sampling*R*ln(hi/lo).
            spectrum_len = sampling * resolution * np.log(hi_aa / lo_aa)
        fits = "fits L" if spectrum_len <= length_pix else "EXCEEDS L"
        lines.append(f"spectrum length   : {spectrum_len:.0f} px over {lo_aa:.0f}-{hi_aa:.0f} A "
                     f"({mode}); detector L={length_pix:.0f}, W={width_pix:.0f} px -> {fits}")

        # Visible diffraction orders (slitless / objective grating): the zero
        # order (star) is at pixel 0; order m disperses lambda to pixel
        # N = lambda[A] * L[mm] * n[l/mm] * m / (1e4 * p[um]) from the star.
        grating_lines = float(self.grating_lines_var.get() or 0.0)
        if mode == "slitless" and grating_lines > 0:
            grating_distance = float(self.grating_distance_var.get())
            lines.append("orders (from zero order at px 0, along L):")
            for m in (1, 2, 3):
                start = lo_aa * grating_distance * grating_lines * m / (1.0e4 * pixel_um)
                end = hi_aa * grating_distance * grating_lines * m / (1.0e4 * pixel_um)
                if start > length_pix:
                    note = f"start px {start:.0f} > L={length_pix:.0f} -> off chip"
                elif end > length_pix:
                    note = f"px {start:.0f}-{end:.0f}, truncated at L={length_pix:.0f}"
                else:
                    note = f"px {start:.0f}-{end:.0f} on chip"
                lines.append(f"  order {m}         : {note}")

        # Horne (1986) optimal extraction: the noise-equivalent width of the
        # variance-weighted extraction is 1/Integral(P^2 dx) = 2*sqrt(pi)*sigma
        # for a Gaussian spatial profile of the (effective) seeing.
        seeing = float(self.seeing_var.get())
        sigma = seeing / 2.35482
        horne_arcsec = 2.0 * np.sqrt(np.pi) * sigma
        horne_pix = horne_arcsec / plate_scale if plate_scale > 0 else float("nan")
        lines.append(f"Horne extraction  : optimal width {horne_arcsec:.2f} arcsec "
                     f"({horne_pix:.1f} px), = 2*sqrt(pi)*sigma noise-equivalent (Horne 1986)")
        if horne_pix > width_pix:
            lines.append(f"  note            : optimal width {horne_pix:.1f} px exceeds detector W={width_pix:.0f}")
        return lines

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
        self.sub_exposure_var = self._var("stack_sub_exposure_s", "0")
        self._entry(f, 5, "Stack sub-exposure (s, 0=auto):", self.sub_exposure_var)
        self.stack_plan_var = tk.StringVar(value="Stack plan appears here after a run with a target S/N.")
        ttk.Label(f, textvariable=self.stack_plan_var, foreground="#1f4f82", wraplength=300,
                  justify="left").grid(row=6, column=0, columnspan=2, sticky="w", pady=(3, 0))

        f = self._section(holder, "PHOTOMETRY")
        self.photometry_box = f
        # Source geometry is the first row: it decides which of the fields
        # below apply.  'point' uses the aperture radius; 'extended' uses the
        # source area; 'defocus' uses the intra/extra-focal position and the
        # defocused-PSF (donut) calculator, which sets the aperture radius.
        self.source_geometry_var = self._var("source_geometry", "point")
        ttk.Label(f, text="Source geometry:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.source_geometry_var, state="readonly",
                     values=("point", "extended", "defocus")).grid(row=0, column=1, sticky="ew")
        self.source_geometry_var.trace_add("write", lambda *_: self._update_photometry_geometry_state())
        self.aperture_var = self._var("photometric_aperture_radius_arcsec", "1.0")
        self.aperture_entry = self._labeled_entry(f, 1, "Aperture radius (arcsec):", self.aperture_var)
        self.source_area_var = self._var("source_area_arcsec2", "100.0")
        self.source_area_entry = self._labeled_entry(f, 2, "Extended source area (arcsec2):",
                                                     self.source_area_var)
        # Intra/extra-focal position (um): 0 = focused; +/- move the detector
        # behind/ahead of focus.  The button opens the defocused-PSF window.
        self.defocus_position_var = self._var("defocus_position_um", "0")
        ttk.Label(f, text="Focus position (um, +/- intra/extra):").grid(row=3, column=0, sticky="w")
        defocus_row = ttk.Frame(f)
        defocus_row.grid(row=3, column=1, sticky="ew")
        self.defocus_entry = ttk.Entry(defocus_row, textvariable=self.defocus_position_var, width=8)
        self.defocus_entry.grid(row=0, column=0, sticky="w")
        self.defocus_button = ttk.Button(defocus_row, text="Calculate defocused PSF",
                                         command=self._calculate_defocus_psf)
        self.defocus_button.grid(row=0, column=1, padx=(4, 0))
        self.comparison_mag_var = self._var("comparison_star_mag", "")
        self._entry(f, 4, "Comparison star mag (optional):", self.comparison_mag_var)
        self.differential_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.differential_var, foreground="#1f4f82", wraplength=280,
                  justify="left").grid(row=5, column=0, columnspan=2, sticky="w", pady=(3, 0))

        f = self._section(holder, "SPECTROSCOPY")
        self.spectroscopy_box = f
        # Widgets that belong only to one mode; the other set is greyed out.
        self._slit_widgets = []
        self._slitless_widgets = []
        self.spectroscopy_mode_var = self._var("spectroscopy_mode", "slit")
        ttk.Label(f, text="Mode:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.spectroscopy_mode_var, state="readonly",
                     values=("slit", "slitless")).grid(row=0, column=1, sticky="ew")
        self.spectroscopy_mode_var.trace_add("write", lambda *_: self._update_spectroscopy_mode_state())
        self.resolution_var = self._var("resolution_R", "10000")
        self.slit_var = self._var("slit_width_arcsec", "1.0")
        self.extract_var = self._var("extraction_height_arcsec", "1.0")
        self.sampling_var = self._var("pixels_per_resolution_element", "2.0")
        self.slitless_width_var = self._var("slitless_extraction_width_arcsec", "1.0")
        self.slitless_dispersion_var = self._var("slitless_dispersion_aa_pix", "10.0")
        self.slitless_lsf_var = self._var("slitless_intrinsic_fwhm_pix", "1.0")
        # A single grating groove density feeds both the slitless disperser
        # (Star Analyser dispersion) and the slit-spectrograph Littrow geometry
        # helper below.
        self.grating_lines_var = self._var("grating_lines_mm", str(self._saved.get("slitless_grating_lines_mm", "0")))
        self.grating_distance_var = self._var("slitless_grating_distance_mm", "42.0")
        self.grating_efficiency_var = self._var("slitless_grating_efficiency", "1.0")
        self.grating_efficiency_path_var = self._var("grating_efficiency_curve_path", "")
        self.grating_efficiency_kind_var = self._var("grating_efficiency_kind", "value")
        self.slit_orientation_var = self._var("slit_orientation", "parallactic")
        self.wlmin_var = self._var("wavelength_min_aa", "4000")
        self.wlmax_var = self._var("wavelength_max_aa", "10000")
        self.reference_wavelength_var = self._var("reference_wavelength_aa", "5500")
        self.wl_unit_var = self._var("spectroscopy_wavelength_unit", "AA")

        # Slit-only fields.
        for row, label, var in [(1, "Slit resolving power R:", self.resolution_var),
                                (2, "Slit width (arcsec):", self.slit_var),
                                (3, "Extraction height (arcsec):", self.extract_var),
                                (4, "Pixels / slit res. element:", self.sampling_var)]:
            self._slit_widgets.append(self._labeled_entry(f, row, label, var))
        # Calibrated slit-R(width) curve (moved here from the detector box):
        # a slit-mode input that overrides the nominal R within its range.
        self.slit_resolution_path_var = self._var("slit_resolution_curve_path", "")
        slit_r_label = ttk.Label(f, text="Calibrated R(width) curve:")
        slit_r_label.grid(row=22, column=0, sticky="w")
        slit_r_row = ttk.Frame(f); slit_r_row.grid(row=22, column=1, sticky="ew")
        slit_r_browse = ttk.Button(slit_r_row, text="Browse…", command=self._choose_slit_resolution_curve)
        slit_r_browse.pack(side="left")
        slit_r_clear = ttk.Button(slit_r_row, text="Clear", command=self._clear_slit_resolution_curve)
        slit_r_clear.pack(side="left", padx=(3, 0))
        self.slit_resolution_status = tk.StringVar(value="none")
        ttk.Label(f, textvariable=self.slit_resolution_status, foreground="#1f4f82",
                  wraplength=280).grid(row=23, column=0, columnspan=2, sticky="w")
        self._slit_widgets += [slit_r_browse, slit_r_clear]
        # Slitless-only fields.
        for row, label, var in [(5, "Slitless cross-disp extraction:", self.slitless_width_var),
                                (6, "Slitless dispersion (A/pix):", self.slitless_dispersion_var),
                                (7, "Slitless intrinsic FWHM (pix):", self.slitless_lsf_var),
                                (9, "Grating-sensor distance (mm):", self.grating_distance_var)]:
            self._slitless_widgets.append(self._labeled_entry(f, row, label, var))
        # Shared: grating groove density.
        self._labeled_entry(f, 8, "Grating lines/mm (0=manual):", self.grating_lines_var)

        # Grating efficiency condensed to one row: [value/file] [scalar] [Load CSV].
        ttk.Label(f, text="Grating efficiency:").grid(row=10, column=0, sticky="w")
        geff_row = ttk.Frame(f); geff_row.grid(row=10, column=1, sticky="ew")
        self.grating_eff_kind_combo = ttk.Combobox(geff_row, textvariable=self.grating_efficiency_kind_var,
                                                    state="readonly", width=6, values=("value", "file"))
        self.grating_eff_kind_combo.pack(side="left")
        self.grating_eff_value_entry = ttk.Entry(geff_row, textvariable=self.grating_efficiency_var, width=6)
        self.grating_eff_value_entry.pack(side="left", padx=(3, 0))
        self.grating_eff_load_button = ttk.Button(geff_row, text="Load CSV…",
                                                  command=self._choose_grating_efficiency_curve)
        self.grating_eff_load_button.pack(side="left", padx=(3, 0))
        self.grating_efficiency_status_var = tk.StringVar(value="scalar")
        ttk.Label(f, textvariable=self.grating_efficiency_status_var, foreground="#1f4f82",
                  wraplength=280).grid(row=11, column=0, columnspan=2, sticky="w")
        self.grating_efficiency_kind_var.trace_add("write", lambda *_: self._update_grating_eff_state())

        # Wavelength range + reference on one row, with a unit selector.

        wl_row = ttk.Frame(f)
        wl_row.grid(row=12, column=0, columnspan=2, sticky="w")
        
        ttk.Label(wl_row, text="WL").pack(side="left")
        ttk.Label(wl_row, text="min").pack(side="left", padx=(4, 0))
        ttk.Entry(wl_row, textvariable=self.wlmin_var, width=4).pack(side="left", padx=(2, 5))
        ttk.Label(wl_row, text="max").pack(side="left")
        ttk.Entry(wl_row, textvariable=self.wlmax_var, width=4).pack(side="left", padx=(2, 5))
        ttk.Label(wl_row, text="ref").pack(side="left")
        ttk.Entry(wl_row, textvariable=self.reference_wavelength_var, width=4).pack(side="left", padx=(2, 5))
        ttk.Combobox(
            wl_row, textvariable=self.wl_unit_var, state="readonly",
            width=4, values=("AA", "nm", "um")
        ).pack(side="left")
    

        # Slit orientation (slit-only) and the Littrow geometry helper.
        ttk.Label(f, text="Slit orientation:").grid(row=15, column=0, sticky="w")
        slit_orient_combo = ttk.Combobox(f, textvariable=self.slit_orientation_var, state="readonly",
                                         values=("parallactic", "fixed"))
        slit_orient_combo.grid(row=15, column=1, sticky="ew")
        self._slit_widgets.append(slit_orient_combo)
        self.spec_collimator_var = self._var("spectrograph_collimator_fl_mm", "130")
        self.spec_camera_var = self._var("spectrograph_camera_fl_mm", "130")
        self._slit_widgets.append(self._labeled_entry(f, 17, "Collimator focal length (mm):", self.spec_collimator_var))
        self._slit_widgets.append(self._labeled_entry(f, 18, "Camera focal length (mm):", self.spec_camera_var))
#        ttk.Label(f, text="Grating lines/mm above feeds both modes. R depends on slit width, "
#                          "collimator FL, grating and telescope FL — NOT the camera FL (it cancels in "
#                          "the Littrow equation; the camera FL sets only the pixel sampling of the "
#                          "resolution element). Collimator, camera and telescope are three distinct "
#                         "focal lengths.",
#                  foreground="gray35", wraplength=280, justify="left").grid(row=16, column=0, columnspan=2, sticky="w")
        compute_r_button = ttk.Button(f, text="Compute R from geometry → fill R field",
                                      command=self._compute_slit_resolving_power)
        compute_r_button.grid(row=19, column=0, columnspan=2, sticky="ew", pady=(3, 1))
        self._slit_widgets.append(compute_r_button)
#        self.spec_geometry_status_var = tk.StringVar(
#            value="Littrow grating equation; uses telescope FL, slit width, seeing and the S/N reference wavelength. R stays editable.")
#        ttk.Label(f, textvariable=self.spec_geometry_status_var, foreground="gray35",
#                  wraplength=280, justify="left").grid(row=20, column=0, columnspan=2, sticky="w")
        self.clamp_r_var = self._var("clamp_r_to_geometry", "0")
        self.clamp_r_check = ttk.Checkbutton(f, text="Clamp R to spectrograph geometry (engine-side sanity limit)",
                        variable=self.clamp_r_var, onvalue="1", offvalue="0")
        self.clamp_r_check.grid(row=21, column=0, columnspan=2, sticky="w")
        self._slit_widgets.append(self.clamp_r_check)
        self._update_spectroscopy_mode_state()
        self._update_grating_eff_state()
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
        # The data directory is fixed to the bundled 'data/' folder and the
        # catalogue loads automatically at startup; only the number of loaded
        # spectra is reported here.  Full loader diagnostics (filters, QE,
        # atmosphere) go to the status bar instead.
        self.data_dir_var = self._var("data_directory", "data")
        self.template_count_var = tk.StringVar(value="Templates not loaded yet.")
        ttk.Label(f, textvariable=self.template_count_var, foreground="blue", wraplength=420,
                  justify="left").grid(row=2, column=0, columnspan=2, sticky="w")
        # Retained for compatibility with loader code that still writes to it;
        # not shown in this box.
        self.catalog_status = tk.StringVar(value="")
        self.star_search_var = tk.StringVar()
        ttk.Label(f, text="Spectral type search:").grid(row=3, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.star_search_var).grid(row=3, column=1, sticky="ew")
        self.star_search_var.trace_add("write", lambda *_: self._refresh_star_list())
        self.star_tree = ttk.Treeview(f, columns=("name", "spt", "bv"), show="headings", height=5)
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
        # Radial velocity and reddening are properties of the TARGET, so their
        # input fields live in the target box (column 1); they are applied to
        # the selected template.
        self.template_display_var = tk.StringVar(value="Selected template spectrum will appear here.")
        ttk.Label(f, textvariable=self.template_display_var, foreground="blue", wraplength=310,
                  justify="left").grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.template_canvas = tk.Canvas(f, height=145, background="white", highlightthickness=1,
                                         highlightbackground="#b5b5b5")
        self.template_canvas.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.template_canvas.bind("<Configure>", lambda _event: self._update_template_display())

        # System-response preview lives here, below the template selector.
        self.response_preview_section = self._section(holder, "SYSTEM RESPONSE")
        self.response_preview_status = tk.StringVar(value="Select an observing filter and a stellar template.")
        ttk.Label(self.response_preview_section, textvariable=self.response_preview_status, foreground="blue",
                  wraplength=320, justify="left").grid(row=0, column=0, sticky="w")
        self.response_preview_figure = Figure(figsize=(3.4, 1.8), dpi=100)
        self.response_preview_axis = self.response_preview_figure.add_subplot(111)
        self.response_preview_canvas = FigureCanvasTkAgg(self.response_preview_figure, master=self.response_preview_section)
        self.response_preview_canvas.get_tk_widget().grid(row=1, column=0, sticky="ew", pady=(3, 0))
        self.response_preview_section.pack_forget()

    def _build_actions_column(self, holder):
        f = self._section(holder, "RUN ETC")
#        ttk.Button(f, text="Run ETC", command=self._run_etc).grid(row=0, column=0, sticky="ew", pady=2)
#        ttk.Button(f, text="Exit", command=self._on_exit).grid(row=1, column=0, sticky="ew", pady=(14, 2))
#        tk.Button(
#            f, text="Run ETC", command=self._run_etc,
#            bg="#22a447", activebackground="#178134",
#            fg="white", activeforeground="white",
#        ).grid(row=0, column=0, sticky="ew", pady=2)

        # Run ETC and Exit aligned on one row.
        button_row = ttk.Frame(f); button_row.grid(row=0, column=0, sticky="ew", pady=2)
        button_row.columnconfigure(0, weight=1); button_row.columnconfigure(1, weight=1)
        self.run_etc_button = tk.Button(
            button_row, text="Run ETC", command=self._run_etc,
            bg="#22a447", activebackground="#178134", fg="white", activeforeground="white")
        self.run_etc_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        tk.Button(
            button_row, text="Exit", command=self._on_exit,
            bg="#c62828", activebackground="#8e1b1b", fg="white", activeforeground="white",
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0))


        f = self._section(holder, "OBSERVATORY STATUS")
        self.observatory_status_var = tk.StringVar(value="Updating current observatory status...")
        ttk.Label(f, textvariable=self.observatory_status_var, justify="left", wraplength=220,
                  foreground="#1f4f82").grid(row=0, column=0, sticky="w")

        f = self._section(holder, "TIME CONVERTER")
        # Default input: the date/time at which the program started, in ISO.
        start_iso = Time(self._program_start_utc, scale="utc")
        start_iso.precision = 3
        self.time_convert_input_kind_var = tk.StringVar(value="ISO")
        self.time_convert_output_kind_var = tk.StringVar(value="MJD")
        self.time_convert_input_var = tk.StringVar(value=start_iso.iso)
        self.time_convert_output_var = tk.StringVar()
        # Wide, left-aligned input that fits a full ISOT string, with a Now
        # button that fills the current date/time in the selected format.
        input_row = ttk.Frame(f); input_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        input_row.columnconfigure(0, weight=1)
        ttk.Entry(input_row, textvariable=self.time_convert_input_var, width=26,
                  justify="left").grid(row=0, column=0, sticky="ew")
        ttk.Button(input_row, text="Now", width=4,
                   command=self._fill_time_now).grid(row=0, column=1, padx=(3, 0))
        # Input / Output type selectors on one row.
        types_row = ttk.Frame(f); types_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(types_row, text="Input").pack(side="left")
        ttk.Combobox(types_row, textvariable=self.time_convert_input_kind_var,
                     values=("ISO", "ISOT", "MJD", "JD"), state="readonly", width=6).pack(side="left", padx=(2, 8))
        ttk.Label(types_row, text="Output").pack(side="left")
        output_combo = ttk.Combobox(types_row, textvariable=self.time_convert_output_kind_var,
                                    values=("ISO", "ISOT", "MJD", "JD"), state="readonly", width=6)
        output_combo.pack(side="left", padx=(2, 0))
        output_combo.bind("<<ComboboxSelected>>", lambda _event: self._convert_time_value())
        ttk.Button(f, text="Convert", command=self._convert_time_value).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(3, 1))
        output = ttk.Entry(f, textvariable=self.time_convert_output_var, state="readonly", justify="left")
        output.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(3, 0))

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

        f = self._section(holder, "LENGTH CONVERTER")
        self.length_input_var = tk.StringVar(value="1")
        self.length_from_unit_var = tk.StringVar(value="m")
        self.length_to_unit_var = tk.StringVar(value="ly")
        self.length_output_var = tk.StringVar()
        units = tuple(self.LENGTH_UNITS_TO_M.keys())
        in_row = ttk.Frame(f); in_row.grid(row=0, column=0, columnspan=3, sticky="ew")
        in_row.columnconfigure(0, weight=1)
        ttk.Entry(in_row, textvariable=self.length_input_var, width=12).grid(row=0, column=0, sticky="ew")
        ttk.Combobox(in_row, textvariable=self.length_from_unit_var, state="readonly", width=7,
                     values=units).grid(row=0, column=1, padx=(3, 0))
        ttk.Label(f, text="TO").grid(row=1, column=0, sticky="w", pady=(2, 0))
        out_row = ttk.Frame(f); out_row.grid(row=2, column=0, columnspan=3, sticky="ew")
        out_row.columnconfigure(0, weight=1)
        ttk.Entry(out_row, textvariable=self.length_output_var, state="readonly", width=12).grid(row=0, column=0, sticky="ew")
        ttk.Combobox(out_row, textvariable=self.length_to_unit_var, state="readonly", width=7,
                     values=units).grid(row=0, column=1, padx=(3, 0))
        ttk.Button(f, text="Convert", command=self._convert_length).grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(3, 1))

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

    # Length units expressed in metres (SI and astronomical).
    LENGTH_UNITS_TO_M = {
        "angstrom": 1.0e-10, "nm": 1.0e-9, "um": 1.0e-6, "mm": 1.0e-3, "cm": 1.0e-2,
        "inch": 0.0254, "foot": 0.3048, "yard": 0.9144, "m": 1.0, "km": 1.0e3,
        "mile": 1609.344, "ly": 9.4607304725808e15, "parsec": 3.0856775814913673e16,
    }

    def _convert_length(self):
        """Convert a positive length between the listed units.

        Negative input is made positive (abs).  Results in
        [1e-5, 1e5] are shown in full decimal; outside that range in
        exponential notation.
        """
        try:
            value = abs(float(self.length_input_var.get()))
            from_u = self.length_from_unit_var.get()
            to_u = self.length_to_unit_var.get()
            if from_u == to_u:
                self.length_output_var.set("choose two different units")
                return
            metres = value * self.LENGTH_UNITS_TO_M[from_u]
            result = metres / self.LENGTH_UNITS_TO_M[to_u]
            self.length_input_var.set(f"{value:g}")  # reflect the abs()
            if result == 0.0:
                text = "0"
            elif 1.0e-5 <= result <= 1.0e5:
                text = f"{result:.10f}".rstrip("0").rstrip(".")
            else:
                text = f"{result:.6e}"
            self.length_output_var.set(text)
        except (ValueError, KeyError, ZeroDivisionError) as exc:
            self.length_output_var.set(f"Conversion failed: {exc}")

    def _fill_time_now(self):
        """Fill the time-converter input with the current UTC in the input format."""
        now = Time(datetime.now(timezone.utc), scale="utc")
        kind = self.time_convert_input_kind_var.get()
        if kind in ("ISO", "ISOT"):
            now.precision = 3
            self.time_convert_input_var.set(now.iso if kind == "ISO" else now.isot)
        elif kind == "MJD":
            self.time_convert_input_var.set(f"{now.mjd:.8f}")
        else:
            self.time_convert_input_var.set(f"{now.jd:.8f}")
        self._convert_time_value()

    def _convert_time_value(self):
        try:
            value, kind = self.time_convert_input_var.get().strip(), self.time_convert_input_kind_var.get()
            if kind in ("ISO", "ISOT"):
                # astropy's 'iso' accepts both 'YYYY-MM-DD HH:MM:SS' and the
                # 'T'-separated ISOT form after stripping a trailing Z.
                parsed = Time(value.replace("Z", "").replace("T", " "), format="iso", scale="utc")
            elif kind == "MJD":
                parsed = Time(float(value), format="mjd", scale="utc")
            else:
                parsed = Time(float(value), format="jd", scale="utc")
            output_kind = self.time_convert_output_kind_var.get()
            if output_kind in ("ISO", "ISOT"):
                parsed.precision = 3
                output = parsed.iso if output_kind == "ISO" else parsed.isot
            elif output_kind == "MJD":
                output = f"{parsed.mjd:.8f}"
            else:
                output = f"{parsed.jd:.8f}"
            self.time_convert_output_var.set(output)
        except Exception as exc:
            self.time_convert_output_var.set(f"Conversion failed: {exc}")

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
            self.dms_output_var.set(f"Conversion failed: {exc}")

    def _load_site_records(self):
        """Load the site list from observatories.json (the single site store).

        ``observatories.json`` now carries the timezone and UTC offset of each
        site, so selecting a site sets the observation date/time reference
        automatically.  The obsolete ``etc_sites.json`` is no longer read.
        """
        from config_manager import load_observatories
        return {
            name: {"name": name, "lat": obs.lat, "lon": obs.lon, "elev": obs.elev,
                   "utc_offset_h": obs.utc_offset_h,
                   "timezone": obs.timezone or "",
                   "sky_ab_mag_arcsec2": obs.mag_sky}
            for name, obs in load_observatories().items()
        }

    def _site_names(self):
        return sorted(self._site_records, key=str.casefold)

    def _write_site_records(self):
        """Persist the site list back into observatories.json."""
        from config_manager import ObservatoryPreset, save_observatories
        presets = {
            name: ObservatoryPreset(
                name=name, lon=rec["lon"], lat=rec["lat"], elev=rec["elev"],
                mag_sky=rec.get("sky_ab_mag_arcsec2", 21.5),
                timezone=rec.get("timezone", ""),
                utc_offset_h=rec.get("utc_offset_h", round(rec["lon"] / 15.0)))
            for name, rec in self._site_records.items()
        }
        save_observatories(presets)

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

    def _open_google_earth(self):
        """Open the default browser at the site coordinates in Google Earth."""
        try:
            lat, lon = float(self.lat_var.get()), float(self.lon_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Google Earth", "Enter numeric latitude and longitude first.")
            return
        # Google Earth web camera URL: @lat,lon,altitude a,distance d,y,heading,tilt.
        # A ~2 km camera distance gives a useful site overview.
        url = f"https://earth.google.com/web/@{lat:.6f},{lon:.6f},1000a,2000d,35y,0h,0t,0r"
        import webbrowser
        webbrowser.open(url)
        self.status_var.set(f"Opened Google Earth at {lat:.5f}, {lon:.5f}.")

    def _horizon_radius_km(self):
        value = float(self.horizon_radius_var.get())
        if self.horizon_unit_var.get() == "miles":
            value *= 1.609344
        import horizon_profile
        if not horizon_profile.MIN_RADIUS_KM <= value <= horizon_profile.MAX_RADIUS_KM:
            raise ValueError(f"Horizon radius must be 1-100 km "
                             f"({value:.1f} km requested).")
        return value

    def _generate_horizon(self):
        import horizon_profile
        try:
            lat, lon = float(self.lat_var.get()), float(self.lon_var.get())
            radius_km = self._horizon_radius_km()
        except ValueError as exc:
            messagebox.showerror("Horizon profile", str(exc)); return
        self.horizon_generate_button.configure(state="disabled")
        self.horizon_status_var.set(
            f"Downloading DEM and computing horizon (r = {radius_km:.1f} km)… "
            "The window stays responsive; this can take a few minutes on first use.")

        def worker():
            try:
                outcome = horizon_profile.generate_horizon(lat, lon, radius_km)
            except Exception as exc:  # shown to the user, never silent
                message = str(exc) or exc.__class__.__name__
                self.after(0, lambda msg=message: self._horizon_done(None, msg))
                return
            self.after(0, lambda: self._horizon_done(outcome, None))

        import threading
        threading.Thread(target=worker, daemon=True).start()

    def _horizon_done(self, outcome, error):
        self.horizon_generate_button.configure(state="normal")
        if error:
            self.horizon_status_var.set(f"Horizon generation failed: {error}")
            return
        self.horizon_status_var.set(
            f"Horizon saved: {outcome['path'].name} (highest obstruction "
            f"{outcome['max_horizon_deg']:.1f} deg at azimuth {outcome['max_horizon_azimuth_deg']:.0f}).")
        self._display_horizon()

    def _current_horizon(self):
        """Return (azimuths_deg, horizon_deg) for the current site, or None.

        Used to overlay the terrain horizon on the sky path and to mask the
        target's visibility on the altitude panel.  Any saved profile for
        these coordinates is used; the search radius is not required to match.
        """
        import horizon_profile
        try:
            lat, lon = float(self.lat_var.get()), float(self.lon_var.get())
        except (ValueError, tk.TclError):
            return None
        candidates = sorted(horizon_profile.HORIZON_DIR.glob(
            f"horizon_lat{lat:+.4f}_lon{lon:+.4f}_r*.csv"))
        if not candidates:
            return None
        try:
            azimuths, horizon, _ = horizon_profile.load_horizon_csv(candidates[-1])
        except (OSError, ValueError):
            return None
        return azimuths, horizon

    def _display_horizon(self):
        import horizon_profile
        try:
            lat, lon = float(self.lat_var.get()), float(self.lon_var.get())
            radius_km = self._horizon_radius_km()
        except ValueError as exc:
            messagebox.showerror("Horizon profile", str(exc)); return
        path = horizon_profile.horizon_csv_path(lat, lon, radius_km)
        if not path.is_file():
            candidates = sorted(horizon_profile.HORIZON_DIR.glob(
                f"horizon_lat{lat:+.4f}_lon{lon:+.4f}_r*.csv"))
            if not candidates:
                messagebox.showinfo("Horizon profile",
                                    "No saved horizon for these coordinates. Generate one first.")
                return
            path = candidates[-1]
        try:
            azimuths, horizon, metadata = horizon_profile.load_horizon_csv(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Horizon profile", str(exc)); return
        window = tk.Toplevel(self)
        window.title(f"Horizon profile - {path.name}")
        figure = Figure(figsize=(7.4, 3.4), dpi=100)
        axis = figure.add_subplot(111)
        axis.fill_between(azimuths, horizon, horizon_profile.HORIZON_MIN_DEG,
                          where=horizon > horizon_profile.HORIZON_MIN_DEG, color="#b0c4de", alpha=0.7)
        axis.plot(azimuths, horizon, color="#1f4f82", linewidth=1.2)
        axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        axis.set(xlim=(0, 360),
                 ylim=(horizon_profile.HORIZON_MIN_DEG, max(10.0, float(np.max(horizon)) + 3.0)),
                 xlabel="Azimuth [deg]  (N=0, E=90, S=180, W=270)",
                 ylabel="Horizon elevation [deg]",
                 title=f"r = {metadata.get('radius_km', '?')} km, site elevation "
                       f"{metadata.get('center_elevation_m', float('nan')):.0f} m")
        axis.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        axis.grid(True, color="#dddddd")
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        canvas.draw_idle()

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
            self.status_var.set(f"Site {name!r} saved to observatories.json.")
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
            self.status_var.set(f"Imported {imported} site(s) into observatories.json.")
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
            "guiding_rms_arcsec", "include_telluric_bands", "sky_annulus_pixels",
        )

    def _update_stack_plan(self, target_snr, source_rate, sky_rate, dark_rate_total,
                           n_pixels, max_unsaturated_s, detector, what="aperture",
                           airmass=1.0):
        """Fill the stack-plan label from the selected-time rates."""
        user_sub = float(self.sub_exposure_var.get() or 0.0)
        scint_rate = scintillation_variance_rate_e2_s(
            source_rate, float(self.diam_var.get()), airmass, float(self.elev_var.get()))
        if target_snr is None:
            plan = plan_stack(100.0, source_rate, sky_rate, dark_rate_total,
                              detector.read_noise_e, n_pixels, max_unsaturated_s, user_sub,
                              extra_variance_rate_e2_s=scint_rate)
            if plan is None:
                self.stack_plan_var.set("Stack plan unavailable for the current rates.")
                return
            self.stack_plan_var.set(
                f"Sub-exposure guidance ({what}): saturation caps single frames at "
                f"{max_unsaturated_s:.0f} s; background dominates read noise above "
                f"{plan['sky_limited_sub_s']:.0f} s per frame. Enter a target S/N for a full plan.")
            return
        plan = plan_stack(target_snr, source_rate, sky_rate, dark_rate_total,
                          detector.read_noise_e, n_pixels, max_unsaturated_s, user_sub,
                          extra_variance_rate_e2_s=scint_rate)
        if plan is None:
            self.stack_plan_var.set("Stack plan unavailable for the current rates.")
            return
        self.stack_plan_var.set(
            f"Stack plan ({what}): {plan['n_frames']} × {plan['sub_exposure_s']:.0f} s "
            f"(sub limited by {plan['limited_by']}) = {plan['total_time_s']:.0f} s total for "
            f"S/N {plan['achieved_snr']:.0f}; read-noise penalty +{plan['read_noise_penalty_percent']:.1f}% "
            f"vs one ideal exposure. Background beats read noise above {plan['sky_limited_sub_s']:.0f} s/frame.")

    def _update_differential_precision(self, calculator, result, exposure_s, run_kwargs):
        """Differential-photometry error in mmag against a comparison star."""
        text = self.comparison_mag_var.get().strip()
        if not text:
            self.differential_var.set("")
            return
        try:
            comparison_mag = float(text)
            comparison = calculator.compute_photometry_single(
                self.star_spec, run_kwargs["observing_band"], self.qe_curve, comparison_mag,
                exposure_s, run_kwargs["reference_zero_point_jy"], run_kwargs["reference_band"],
                self.selected_star.mv0, run_kwargs["visual_band"], run_kwargs["visual_zero_point_jy"],
                run_kwargs["observing_zero_point_jy"], run_kwargs["reference_detector_type"],
                run_kwargs["visual_detector_type"], run_kwargs["observing_detector_type"],
                **self._source_geometry_kwargs())
        except (ValueError, KeyError) as exc:
            self.differential_var.set(f"Differential precision unavailable: {exc}")
            return
        # 1.0857 mag per unit fractional error; target and comparison noise
        # (scintillation included in each S/N) added in quadrature.  Their
        # scintillation is treated as uncorrelated, which is conservative.
        sigma_mmag = 1085.7 * np.sqrt(result["snr"] ** -2 + comparison["snr"] ** -2)
        self.differential_var.set(
            f"Differential precision vs m={comparison_mag:g} comparison: "
            f"{sigma_mmag:.1f} mmag per {exposure_s:.0f} s frame "
            f"(target S/N {result['snr']:.0f}, comparison S/N {comparison['snr']:.0f}).")

    def _compute_slit_resolving_power(self):
        try:
            seeing = float(self.seeing_var.get())
            reference_aa = self._wl_to_aa(self.reference_wavelength_var.get())
            if self.seeing_scaling_var.get() == "1":
                # With seeing scaling enabled the entered value is the zenith
                # V seeing; the helper evaluates the Kolmogorov-scaled seeing
                # at the S/N reference wavelength (zenith).
                seeing = float(effective_seeing_arcsec(seeing, reference_aa, 1.0))
            result = slit_spectrograph_resolving_power(
                reference_aa,
                float(self.grating_lines_var.get()),
                float(self.spec_collimator_var.get()),
                float(self.spec_camera_var.get()),
                float(self.focal_var.get()),
                float(self.slit_var.get()),
                seeing)
        except (ValueError, KeyError) as exc:
            self.spec_geometry_status_var.set(f"Cannot compute R: {exc}")
            return
        self.resolution_var.set(f"{result['resolving_power']:.0f}")
        self.spec_geometry_status_var.set(
            f"R = {result['resolving_power']:.0f} ({result['limited_by']}-limited), "
            f"{result['resolution_element_aa']:.2f} Å element, "
            f"{result['dispersion_aa_mm']:.1f} Å/mm at the detector. R field updated; edit freely.")

    # Typical zenith SQM readings at the midpoint of each Bortle class
    # (Bortle 2001, Sky & Telescope; commonly used amateur conversion).
    BORTLE_SQM = {1: 21.9, 2: 21.8, 3: 21.6, 4: 21.1, 5: 20.5,
                  6: 19.8, 7: 19.0, 8: 18.0, 9: 17.0}

    def _apply_bortle_preset(self):
        try:
            bortle = int(self.bortle_var.get())
        except ValueError:
            return
        self.sqm_var.set(f"{self.BORTLE_SQM[bortle]:.1f}")
        self.status_var.set(f"Bortle {bortle}: typical zenith SQM {self.BORTLE_SQM[bortle]:.1f} mag/arcsec2.")

    def _choose_gain_table(self):
        path = filedialog.askopenfilename(title="Select CMOS gain table",
                                          filetypes=[("Data files", "*.dat *.txt *.csv"), ("All files", "*")])
        if not path:
            return
        self.gain_table_path_var.set(str(Path(path).resolve()))
        self._load_gain_table_from_path()

    def _load_gain_table_from_path(self, silent=False):
        path = self.gain_table_path_var.get().strip()
        if not path:
            return
        try:
            self.gain_table_rows = load_gain_table(path)
        except (OSError, ValueError) as exc:
            self.gain_table_rows = []
            self.gain_setting_combo.configure(state="disabled", values=())
            if not silent:
                messagebox.showerror("Gain table", str(exc))
            return
        values = tuple(f"{row['gain_setting']:g}" for row in self.gain_table_rows)
        self.gain_setting_combo.configure(state="readonly", values=values)
        if self.gain_setting_var.get() not in values:
            self.gain_setting_var.set(values[0])
        self._apply_gain_setting()

    def _apply_gain_setting(self):
        selected = self.gain_setting_var.get().strip()
        for row in self.gain_table_rows:
            if f"{row['gain_setting']:g}" == selected:
                self.gain_var.set(f"{row['gain_e_adu']:g}")
                self.readnoise_var.set(f"{row['read_noise_e']:g}")
                self.fullwell_var.set(f"{row['full_well_e']:g}")
                self.status_var.set(f"Detector set from gain table: setting {selected}.")
                return

    def _choose_grating_efficiency_curve(self):
        """Load a two-column (wavelength, efficiency 0-1) grating-efficiency CSV."""
        path = filedialog.askopenfilename(title="Select grating efficiency curve (CSV)",
                                          filetypes=[("Data files", "*.csv *.dat *.txt"), ("All files", "*")])
        if not path:
            return
        try:
            curve = load_transmission_curve(path)
            curve = curve[np.argsort(curve[:, 0])]
            if np.any(curve[:, 1] < 0) or np.any(curve[:, 1] > 1.5):
                raise ValueError("Efficiency values must be fractions in [0, 1].")
            self.grating_efficiency_curve = curve
            self.grating_efficiency_source_path = Path(path).resolve()
            self.grating_efficiency_path_var.set(str(self.grating_efficiency_source_path))
            self.grating_efficiency_kind_var.set("file")  # applies the curve, greys the scalar
            self.grating_efficiency_status_var.set(f"curve applied: {self.grating_efficiency_source_path.name}")
            self.status_var.set("Grating efficiency curve loaded (overrides the scalar).")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Grating efficiency", str(exc))

    def _choose_qe_curve(self):
        """Load the detector QE from a two-column (wavelength, QE) CSV/text file."""
        path = filedialog.askopenfilename(title="Select detector QE curve (CSV)",
                                          filetypes=[("Data files", "*.csv *.dat *.txt"), ("All files", "*")])
        if not path:
            return
        try:
            self.qe_curve = load_qe_curve(path, self.qe_unit_var.get())
            self.qe_source_path = Path(path).resolve()
            self.qe_status_var.set(self.qe_source_path.name)
            self.status_var.set(f"QE curve loaded: {self.qe_source_path.name}.")
            self._update_combined_response_preview()
        except (OSError, ValueError) as exc:
            self.qe_curve = None
            self.qe_status_var.set("QE load failed")
            messagebox.showerror("Detector QE", str(exc))

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
            self.status_var.set("Instrument transmission curve loaded.")
            self._update_throughput_exclusive()
            self._update_combined_response_preview()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Instrument transmission", str(exc))

    def _clear_throughput_curve(self):
        """Drop the loaded response curve and re-enable the scalar throughput."""
        self.throughput_curve = None
        self.throughput_source_path = None
        self.throughput_path_var.set("")
        self._update_throughput_exclusive()
        self._update_combined_response_preview()

    def _update_throughput_exclusive(self):
        """The scalar optics throughput and a loaded response curve are mutually
        exclusive: whichever is active greys out the other."""
        if not hasattr(self, "throughput_scalar_entry"):
            return
        has_curve = self.throughput_curve is not None
        self._set_widget_enabled(self.throughput_scalar_entry, not has_curve)
        if has_curve:
            name = getattr(self.throughput_source_path, "name", "curve")
            self.throughput_curve_status.set(f"response curve: {name} (scalar throughput ignored)")
        else:
            self.throughput_curve_status.set("response curve: none (using scalar throughput)")

    def _on_qe_unit_changed(self):
        """Reinterpret the QE file when the user explicitly changes its unit."""
        path = self.qe_source_path
        if path is not None and Path(path).is_file():
            try:
                self.qe_curve = load_qe_curve(path, self.qe_unit_var.get())
                if hasattr(self, "qe_status_var"):
                    self.qe_status_var.set(Path(path).name)
            except (OSError, ValueError) as exc:
                self.qe_curve = None
                if hasattr(self, "qe_status_var"):
                    self.qe_status_var.set(f"QE error: {exc}")
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
            if hasattr(self, "slit_resolution_status"):
                self.slit_resolution_status.set(f"R(width): {self.slit_resolution_source_path.name}")
            self.status_var.set("Calibrated slit-resolution curve loaded.")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Slit resolution curve", str(exc))

    def _clear_slit_resolution_curve(self):
        self.slit_resolution_curve = None
        self.slit_resolution_source_path = None
        self.slit_resolution_path_var.set("")
        if hasattr(self, "slit_resolution_status"):
            self.slit_resolution_status.set("none")

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

    def _reload_catalog(self):
        base = self._data_directory()
        messages = []
        try:
            self.star_catalog = scat.parse_interpola_db(base / "interpola.db.csv")
            messages.append(f"{len(self.star_catalog)} templates")
            self.template_count_var.set(f"{len(self.star_catalog)} spectra loaded.")
        except Exception as exc:
            self.star_catalog = []; messages.append(f"catalog: {exc}")
            self.template_count_var.set(f"Templates failed to load: {exc}")
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
            if hasattr(self, "qe_status_var"):
                self.qe_status_var.set("qe.dat (default)")
        except Exception as exc:
            self.qe_curve = None; self.qe_source_path = None; messages.append(f"QE: {exc}")
            if hasattr(self, "qe_status_var"):
                self.qe_status_var.set("no QE loaded")
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
        self.catalog_status.set(" | ".join(messages))
        # Full loader diagnostics go to the general status bar; the template
        # box shows only the spectra count (set above).
        self.status_var.set(" | ".join(messages))
        self._refresh_star_list(); self._update_filter_display(); self._update_template_display(); self._update_combined_response_preview()

    def _on_calculation_mode_changed(self):
        """Use the filter-free response by default for spectroscopy, and grey
        out the box (photometry or spectroscopy) that is not in use."""
        photometry = self.mode_var.get() == "photometry"
        if hasattr(self, "photometry_box"):
            self._set_frame_enabled(self.photometry_box, photometry)
            # Within an enabled photometry box, grey the geometry-specific fields.
            if photometry:
                self._update_photometry_geometry_state()
        # The spectroscopy box (and its slit/slitless sub-state) is handled here.
        self._update_spectroscopy_mode_state()
        if not photometry and self.filter_resp_data is not None:
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
        # The preview x-axis is always the 3000-9000 A optical window, so the
        # visible portion of every template is directly comparable regardless
        # of the template's own coverage.
        xmin, xmax = 3000.0, 9000.0
        window = (wave >= xmin) & (wave <= xmax)
        if window.sum() < 2:
            self.template_display_var.set(
                "The selected template has no positive flux in 3000-9000 Å.")
            return
        wave, flux = wave[window], flux[window]
        log_flux = np.log10(flux)
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
            f"Template Fλ distribution over 3000–9000 Å; log flux span {ymin:.2f} to {ymax:.2f}.")
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
                self.response_preview_section.pack(fill="x", padx=5, pady=4)
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
                self.response_preview_section.pack(fill="x", padx=5, pady=4)
    


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
                "elevation_m": float(self.elev_var.get()),
                "seeing_wavelength_scaling": self.seeing_scaling_var.get() == "1",
                "guiding_rms_arcsec": float(self.guiding_rms_var.get() or 0.0),
                "include_telluric_bands": self.telluric_var.get() == "1"}

    def _source_transform_kwargs(self):
        """Radial-velocity and reddening transformation of the template."""
        return {"radial_velocity_kms": float(self.radial_velocity_var.get() or 0.0),
                "ebv": float(self.ebv_var.get() or 0.0)}

    def _source_geometry_kwargs(self):
        geometry = self.source_geometry_var.get().strip().lower()
        return {"source_geometry": geometry,
                "source_area_arcsec2": float(self.source_area_var.get()) if geometry == "extended" else None,
                "defocus_position_um": float(self.defocus_position_var.get()) if geometry == "defocus" else None}

    def _calculate_defocus_psf(self):
        """Compute and display the realistic (seeing-convolved) defocused-PSF
        for the current telescope/detector/atmosphere, letting the user pick
        the photometry aperture radius from its encircled-energy curve."""
        from defocus import defocused_star_profile
        try:
            defocus_um = float(self.defocus_position_var.get())
            diameter_mm = float(self.diam_var.get())
            focal_mm = float(self.focal_var.get())
            obstruction_mm = float(self.obstruct_var.get())
            pixel_um = float(self.pixel_var.get())
            seeing_arcsec = float(self.seeing_var.get())
            moffat_beta = float(self.moffat_beta_var.get())
            guiding_rms = float(self.guiding_rms_var.get() or 0.0)
        except ValueError as exc:
            messagebox.showerror("Defocused PSF", f"Check the telescope/detector/seeing numbers: {exc}")
            return
        psf_model = self.psf_model_var.get()
        try:
            profile = defocused_star_profile(defocus_um, diameter_mm, focal_mm,
                                             obstruction_mm, pixel_um, seeing_arcsec,
                                             psf_model=psf_model, moffat_beta=moffat_beta,
                                             guiding_rms_arcsec=guiding_rms)
        except ValueError as exc:
            messagebox.showerror("Defocused PSF", str(exc))
            return

        win = tk.Toplevel(self)
        win.title("Defocused PSF (seeing-convolved donut)")
        figure = Figure(figsize=(7.5, 6.6))
        ax_profile = figure.add_subplot(211)
        ax_ee = figure.add_subplot(212)
        # Symmetric radial cross-section through the donut centre (-r..+r).
        r_arcsec = profile["radius_arcsec"]
        intensity = profile["intensity_norm"]
        r_sym = np.concatenate([-r_arcsec[::-1], r_arcsec])
        i_sym = np.concatenate([intensity[::-1], intensity])
        ax_profile.plot(r_sym, i_sym, color="#1f4f82")
        # Mark the ideal geometric annulus edges for reference.
        for edge in (profile["r_in_arcsec"], profile["r_out_arcsec"]):
            for sign in (-1.0, 1.0):
                ax_profile.axvline(sign * edge, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax_profile.set_xlabel("radius (arcsec)")
        ax_profile.set_ylabel("intensity / peak")
        kind = "RC / obstructed" if profile["epsilon"] > 0 else "classical reflector"
        ax_profile.set_title(
            f"{kind}: defocus {defocus_um:+.0f} um, {psf_model} seeing "
            f"{profile['fwhm_arcsec']:.2f}\"  ->  donut "
            f"r_in={profile['r_in_arcsec']:.2f}\", r_out={profile['r_out_arcsec']:.2f}\"")
        ax_profile.grid(True, alpha=0.3)
        ax_ee.plot(r_arcsec, profile["ee_percent"], color="#8b3a3a")
        ax_ee.set_xlabel("aperture radius (arcsec)")
        ax_ee.set_ylabel("encircled energy (%)")
        ax_ee.grid(True, alpha=0.3)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        controls = ttk.Frame(win)
        controls.pack(fill="x", pady=4)
        ttk.Label(controls, text="Photometry aperture radius (arcsec):").pack(side="left", padx=(6, 2))
        default_radius = f"{profile['r_out_arcsec']:.3f}"
        radius_var = tk.StringVar(value=(self.aperture_var.get() or default_radius))
        ttk.Entry(controls, textvariable=radius_var, width=10).pack(side="left")

        def _use_radius():
            try:
                radius = float(radius_var.get())
                if radius <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Defocused PSF", "Enter a positive aperture radius in arcsec.")
                return
            self.aperture_var.set(f"{radius:g}")
            fraction = float(np.interp(radius, profile["radius_arcsec"],
                                       profile["ee_fraction"], left=0.0, right=1.0))
            self.status_var.set(f"Defocus aperture set to {radius:g}\" "
                                f"(encircled energy {100.0 * fraction:.1f}%).")

        ttk.Button(controls, text="Use this radius for photometry",
                   command=_use_radius).pack(side="left", padx=6)
        ttk.Button(controls, text="Save figure + data",
                   command=lambda: self._save_defocus_outputs(figure, profile, defocus_um)
                   ).pack(side="left", padx=6)

    def _save_defocus_outputs(self, figure, profile, defocus_um):
        """Save the donut figure (PNG) and the two data tables (profile and
        encircled energy) into the session directory, or a chosen folder."""
        from pathlib import Path
        target = self._export_dir()
        if not target:
            target = filedialog.askdirectory(title="Save defocused-PSF outputs to…")
            if not target:
                return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        base = Path(target)
        png_path = base / f"defocus_psf_{stamp}.png"
        profile_csv = base / f"defocus_profile_{stamp}.csv"
        ee_csv = base / f"defocus_encircled_energy_{stamp}.csv"
        try:
            figure.savefig(png_path, dpi=120)
            np.savetxt(profile_csv,
                       np.column_stack([profile["radius_px"], profile["radius_arcsec"],
                                        profile["intensity_norm"]]),
                       delimiter=",", header="radius_pixel,radius_arcsec,intensity_norm",
                       comments="")
            np.savetxt(ee_csv,
                       np.column_stack([profile["radius_px"], profile["radius_arcsec"],
                                        profile["ee_percent"]]),
                       delimiter=",", header="radius_pixel,radius_arcsec,encircled_energy_percent",
                       comments="")
        except OSError as exc:
            messagebox.showerror("Defocused PSF", f"Could not save outputs:\n{exc}")
            return
        messagebox.showinfo("Defocused PSF",
                            f"Saved:\n{png_path.name}\n{profile_csv.name}\n{ee_csv.name}\n\n"
                            f"in {base}")

    def _sky_models_for_track(self, track, vega_profile, ra_deg, dec_deg):
        """Build observed ground-sky inputs for every planning time sample."""
        aperture = float(self.aperture_var.get())
        extra_bg = float(self.extra_background_var.get() or 0.0)
        sky_annulus = float(self.sky_annulus_var.get() or 0.0)
        models = self._sky_models_core(track, vega_profile, ra_deg, dec_deg, aperture)
        for model in models:
            model["extra_background_e_s_pixel"] = extra_bg
            model["sky_annulus_pixels"] = sky_annulus
        return models

    def _sky_models_core(self, track, vega_profile, ra_deg, dec_deg, aperture):
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
        sqm_mode = self.sky_model_var.get() == "sqm"
        if sqm_mode:
            # The user's zenith SQM reading is an observed V surface
            # brightness that already contains their light pollution and
            # airglow; the built-in table only supplies the band colour.
            sqm_v = float(self.sqm_var.get())
            v_table = float(np.interp(5510.0, SKY_WAVELENGTH_AA, SKY_MAG_VEGA))
            pivot_colour = float(np.interp(pivot, SKY_WAVELENGTH_AA, SKY_MAG_VEGA)) - v_table
        sun_geometric = track.get("alt_sun_geometric", track["alt_sun"])
        elongations = track.get("sun_sep_deg", np.full(len(track["jd"]), 180.0))
        moon_distances = track.get("moon_distance_km", np.full(len(track["jd"]), 384400.0))
        for utc, alt, airmass, sun_alt, moon_alt, moon_sep, phase, sun_sep, moon_km in zip(
                track["utc_datetime"], track["alt_target"], track["airmass_target"], sun_geometric,
                track["alt_moon"], track["moon_sep_deg"], track["phase_moon"],
                elongations, moon_distances):
            if sqm_mode:
                base_mag = sqm_v + pivot_colour
            else:
                # Twilight/day thresholds use the geometric Sun altitude (the
                # standard convention); the zodiacal light uses the target's
                # solar elongation.
                base_mag = sky_magnitude_vega(pivot, utc, alt, airmass, sun_alt,
                                              ecliptic_lat_deg=ecliptic_lat, galactic_lat_deg=galactic_lat,
                                              solar_elongation_deg=float(sun_sep))
            # Krisciunas--Schaefer moonlight terms are evaluated in the nine
            # supplied broad bands, then used as a spectral colour model.  The
            # ING dark/twilight/day brightness remains the normalization.
            moon_airmass = 1.0 / np.sin(np.deg2rad(moon_alt)) if moon_alt >= 5.0 else 99.0
            common = dict(year=utc.year, month=utc.month, day=utc.day, hour=utc.hour, minute=utc.minute,
                          ecliptic_lat_deg=ecliptic_lat, galactic_lat_deg=galactic_lat,
                          airmass_target=max(float(airmass) if np.isfinite(airmass) else 1.0, 1.0),
                          airmass_moon=moon_airmass, lunar_phase_deg=max(0.0, 180.0 - float(phase)),
                          moon_separation_deg=float(moon_sep), moon_zenith_dist_deg=90.0 - float(moon_alt),
                          target_zenith_dist_deg=90.0 - float(np.clip(alt, 0.0, 90.0)),
                          solar_elongation_deg=float(sun_sep),
                          moon_distance_km=float(moon_km))
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
        transform_kwargs = self._source_transform_kwargs()
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
                    **geometry_kwargs, **transform_kwargs)
                scint_rate = scintillation_variance_rate_e2_s(
                    probe["source_rate_per_s"], float(self.diam_var.get()), airmass, float(self.elev_var.get()))
                # The sky-subtraction factor inflates the background terms in
                # the closed-form solver exactly as it does in the S/N.
                sub_factor = probe.get("sky_subtraction_factor", 1.0)
                real_texp = exposure_time_for_snr(target_snr, probe["source_rate_per_s"],
                                                   sub_factor * probe["sky_rate_per_s"],
                                                   sub_factor * detector.dark_current_e_s_pix * probe["n_pixels"],
                                                   detector.read_noise_e * np.sqrt(sub_factor),
                                                   probe["n_pixels"], extra_variance_rate_e2_s=scint_rate)
                values[i] = real_texp
            else:
                real_texp = texp
            result = calculator.compute_photometry_single(
                self.star_spec, observing_band, self.qe_curve, target_mag, real_texp,
                reference_zero_point_jy, reference_band, template_mv0, visual_band, visual_zero_point_jy,
                observing_zero_point_jy, reference_detector_type, visual_detector_type, observing_detector_type,
                **geometry_kwargs, **transform_kwargs)
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
        reference = self._wl_to_aa(self.reference_wavelength_var.get())
        resolution = float(self.resolution_var.get())
        slider_indices = self._slider_indices(valid_indices, selected_idx, maximum=12 if resolution >= 50000 else 25)
        label = "Required exposure [s]" if target_snr is not None else f"S/N at {reference:.0f} Å"
        grating_lines = float(self.grating_lines_var.get() or 0.0)
        transform_kwargs = self._source_transform_kwargs()
        if self.grating_efficiency_curve is not None:
            transform_kwargs["grating_efficiency_curve"] = self.grating_efficiency_curve
        if self.clamp_r_var.get() == "1" and self.spectroscopy_mode_var.get() == "slit":
            transform_kwargs["slit_geometry"] = {
                "grating_lines_mm": float(self.grating_lines_var.get()),
                "collimator_fl_mm": float(self.spec_collimator_var.get()),
                "camera_fl_mm": float(self.spec_camera_var.get())}
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
                probe = calculator.compute_spectroscopy(*args, **transform_kwargs)
                ref_index = int(np.argmin(np.abs(probe["wavelength_aa"].to_numpy() - reference)))
                scint_rate = scintillation_variance_rate_e2_s(
                    probe.iloc[ref_index]["photons_source_es"], float(self.diam_var.get()),
                    track["airmass_target"][i], float(self.elev_var.get()))
                sub_factor = float(probe.attrs.get("sky_subtraction_factor", 1.0))
                real_texp = exposure_time_for_snr(target_snr, probe.iloc[ref_index]["photons_source_es"],
                                                   sub_factor * probe.iloc[ref_index]["photons_sky_es"],
                                                   sub_factor * detector.dark_current_e_s_pix * probe.attrs["n_pixels_per_resel"],
                                                   detector.read_noise_e * np.sqrt(sub_factor),
                                                   probe.attrs["n_pixels_per_resel"],
                                                   extra_variance_rate_e2_s=scint_rate)
                values[i] = real_texp
            else:
                real_texp = texp
            reference_spectrum = calculator.compute_spectroscopy(*args[:3], real_texp, *args[4:], **transform_kwargs)
            ref_index = int(np.argmin(np.abs(reference_spectrum["wavelength_aa"].to_numpy() - reference)))
            if target_snr is None:
                values[i] = reference_spectrum.iloc[ref_index]["snr"]
            saturation[i] = reference_spectrum.iloc[ref_index]["saturation_flag"]
            peak_e[i] = reference_spectrum.iloc[ref_index]["peak_e_unclipped"]
            max_unsaturated_exptime[i] = reference_spectrum.iloc[ref_index]["max_unsaturated_exptime_s"]
            if i in slider_indices:
                full_args = list(args)
                full_args[4] = (self._wl_to_aa(self.wlmin_var.get()), self._wl_to_aa(self.wlmax_var.get()))
                spectra[i] = calculator.compute_spectroscopy(*full_args[:3], real_texp, *full_args[4:],
                                                             **transform_kwargs)
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
        # Per-sample photometric error budget in millimagnitudes: the total
        # error is 1085.7 / (S/N), and the scintillation contribution is
        # 1085.7 times the Young fractional rms at that airmass and exposure.
        snr_array = np.asarray(snr_values, dtype=float)
        exposure_array = np.asarray(exposure_values, dtype=float)
        total_error_mmag = np.where(snr_array > 0, 1085.7 / np.where(snr_array > 0, snr_array, np.nan), np.nan)
        scint_mmag = np.full(len(airmass), np.nan)
        try:
            diameter_mm = float(self.diam_var.get())
            elevation_m = float(self.elev_var.get())
            for i, x in enumerate(airmass):
                if not np.isfinite(x) or not np.isfinite(exposure_array[i]) or exposure_array[i] <= 0:
                    continue
                scint_mmag[i] = 1085.7 * scintillation_fractional_rms(
                    diameter_mm, x, elevation_m, exposure_array[i])
        except (ValueError, tk.TclError):
            pass
        frame = pd.DataFrame({
            "datetime_utc": [value.strftime("%Y-%m-%dT%H:%M:%S") for value in track["utc_datetime"]],
            "datetime_local": [value.strftime("%Y-%m-%dT%H:%M:%S") for value in track["local_datetime"]],
            "local_timezone": self._local_timezone_label(),
            "mjd": np.asarray(track["jd"], dtype=float) - 2400000.5,
            "elevation_deg": np.asarray(track["alt_target"], dtype=float),
            "azimuth_deg": np.asarray(track["az_target"], dtype=float),
            "parallactic_angle_deg": np.asarray(track["parallactic_deg"], dtype=float),
            "snr": snr_array,
            "total_error_mmag": total_error_mmag,
            "scintillation_mmag": scint_mmag,
            "exptime_s": exposure_array,
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
            nonlin_text = self.nonlin_limit_var.get().strip()
            detector = Detector(float(self.pixel_var.get()), float(self.gain_var.get()), float(self.fullwell_var.get()),
                                int(self.bitdepth_var.get()), float(self.readnoise_var.get()), float(self.dark_var.get()),
                                sensor_type=self.sensor_type_var.get(), osc_channel=self.osc_channel_var.get(),
                                nonlinearity_limit_adu=(nonlin_text or None))
            # Scalar throughput and a calibrated response curve are mutually
            # exclusive: when a curve is loaded the scalar efficiency is set to
            # 1 so the two are not multiplied together.
            optics_efficiency = 1.0 if self.throughput_curve is not None else float(self.eff_var.get())
            telescope = {"diameter_mm": float(self.diam_var.get()), "obstruction_mm": float(self.obstruct_var.get()),
                         "efficiency": optics_efficiency, "focal_length_mm": float(self.focal_var.get()),
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
                exposure_used = float(values[idx]) if target_snr is not None else texp
                self._update_stack_plan(target_snr, result["source_rate_per_s"], result["sky_rate_per_s"],
                                        detector.dark_current_e_s_pix * result["n_pixels"],
                                        result["n_pixels"], result["max_unsaturated_exptime_s"], detector,
                                        airmass=track["airmass_target"][idx])
                selected_calculator = PhotometryETC(
                    telescope, detector, self._atmosphere_dict(track["airmass_target"][idx], atmo), sky_models[idx])
                run_kwargs = dict(
                    observing_band=observing_band, reference_band=reference_band,
                    reference_zero_point_jy=reference_profile.zero_point_jy, visual_band=visual_band,
                    visual_zero_point_jy=visual_profile.zero_point_jy,
                    observing_zero_point_jy=observing_profile.zero_point_jy,
                    reference_detector_type=reference_profile.detector_type,
                    visual_detector_type=visual_profile.detector_type,
                    observing_detector_type=observing_profile.detector_type)
                self._update_differential_precision(selected_calculator, result, exposure_used, run_kwargs)
                scint_fraction = (result["scintillation_noise_e"]
                                  / max(result["photons_source_es"] * exposure_used, 1e-300))
                scint_mmag = 1085.7 * scint_fraction
                total_error_mmag = 1085.7 / result["snr"] if result["snr"] > 0 else float("inf")
                flag_counts = self._frame_flag_counts(
                    [results[i]["saturation_flag"] for i in results])
                self._append_info_outputs([
                    f"mode / exposure   : photometry ({result['source_geometry']}) / {exposure_used:.1f} s",
                    f"S/N               : {result['snr']:.1f}",
                    f"total error       : {total_error_mmag:.2f} mmag (1085.7 / S/N, this frame)",
                    f"aperture          : {result['n_pixels']:.0f} px "
                    f"(radius {result.get('aperture_radius_arcsec', 0.0):.3f} arcsec)"
                    + (f", defocus {result['defocus_position_um']:+.0f} um, "
                       f"encircled energy {100.0 * result['defocus_captured_fraction']:.1f}%"
                       if result.get('source_geometry') == 'defocus' else ""),
                    f"scintillation     : {result['scintillation_noise_e']:.1f} e-  "
                    f"({100.0 * scint_fraction:.2f}% of source, {scint_mmag:.2f} mmag)",
                    f"ADC quantization  : {result['digitization_noise_e']:.1f} e-",
                    f"peak pixel        : {result['peak_e_unclipped']:.0f} e- / "
                    f"{result['peak_adu_unclipped']:.0f} ADU  sat={result['saturation_flag']}  "
                    f"max single frame {result['max_unsaturated_exptime_s']:.1f} s",
                    f"frames (visible)  : {flag_counts[0]} linear, {flag_counts[1]} non-linear, "
                    f"{flag_counts[2]} saturated",
                    f"magnitudes        : standard {result['estimated_observing_magnitude']:.3f}, "
                    f"instrumental {result['instrumental_response_magnitude']:.3f} ({self.band_var.get()})",
                    f"sky used          : {result['sky_mag_arcsec2']:.2f} mag/arcsec2"]
                    + self._airy_disk_lines(observing_profile.pivot_wavelength_aa)
                    + self._template_colour_lines())
                # Magnitude sweep: the same configuration at target +/- 1..7 mag,
                # shown greyed under a separator below the selected result.
                sweep_rows = []
                geometry_kwargs = self._source_geometry_kwargs()
                transform_kwargs = self._source_transform_kwargs()
                for offset in (-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7):
                    try:
                        sweep_rows.append(selected_calculator.compute_photometry_single(
                            self.star_spec, observing_band, self.qe_curve, target_mag + offset, exposure_used,
                            reference_profile.zero_point_jy, reference_band, self.selected_star.mv0, visual_band,
                            visual_profile.zero_point_jy, observing_profile.zero_point_jy,
                            reference_profile.detector_type, visual_profile.detector_type,
                            observing_profile.detector_type, **geometry_kwargs, **transform_kwargs))
                    except (ValueError, KeyError):
                        continue
                frame = pd.DataFrame([result] + sweep_rows)
                self._display_table(frame, "mag", grey_from_row=1 if sweep_rows else None,
                                    separator_before=1 if sweep_rows else None)
                self.result_df = frame
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
                reference_wl = self._wl_to_aa(self.reference_wavelength_var.get())
                ref_row = spectrum.iloc[int(np.argmin(np.abs(spectrum["wavelength_aa"].to_numpy() - reference_wl)))]
                resel_pixels = float(spectrum.attrs["n_pixels_per_resel"])
                self._update_stack_plan(target_snr, float(ref_row["photons_source_es"]),
                                        float(ref_row["photons_sky_es"]),
                                        detector.dark_current_e_s_pix * resel_pixels, resel_pixels,
                                        float(ref_row["max_unsaturated_exptime_s"]), detector,
                                        what=f"res.el at {reference_wl:.0f} Å",
                                        airmass=track["airmass_target"][idx])
                self.differential_var.set(
                    f"σ(EW) at {reference_wl:.0f} Å: {float(ref_row['sigma_ew_mangstrom']):.1f} mÅ "
                    "(Cayrel) for the selected exposure.")
                dispersion_line = (f"dispersion        : {spectrum.attrs['dispersion_aa_pix']:.2f} A/pix, "
                                   f"grating efficiency {spectrum.attrs['grating_efficiency']:.2f}"
                                   if "dispersion_aa_pix" in spectrum.attrs else "")
                self._append_info_outputs([
                    f"mode              : spectroscopy ({spectrum.attrs['spectroscopy_mode']}), "
                    f"median R = {spectrum.attrs['effective_resolution_R']:.0f}",
                    f"at {reference_wl:.0f} A       : S/N {float(ref_row['snr']):.1f} /res.el, "
                    f"sigma(EW) {float(ref_row['sigma_ew_mangstrom']):.1f} mA (Cayrel)",
                    dispersion_line,
                    f"res.el            : {spectrum.attrs['n_pixels_per_resel']:.0f} px; median scintillation "
                    f"{spectrum.attrs['scintillation_noise_e_median']:.1f} e-, ADC "
                    f"{spectrum.attrs['digitization_noise_e']:.1f} e-",
                    f"at {reference_wl:.0f} A       : total error "
                    f"{1085.7 / float(ref_row['snr']) if float(ref_row['snr']) > 0 else float('inf'):.2f} mmag, "
                    f"scintillation {1085.7 * spectrum.attrs['scintillation_noise_e_median'] / max(float(ref_row['photons_source_es']) * exposure_used, 1e-300):.2f} mmag",
                    f"saturation at ref : {ref_row['saturation_flag']}, max single frame "
                    f"{float(ref_row['max_unsaturated_exptime_s']):.1f} s",
                    f"frames (visible)  : "
                    + "%d linear, %d non-linear, %d saturated"
                      % self._frame_flag_counts(list(saturation)),
                    f"sky used          : {spectrum.attrs['sky_mag_arcsec2']:.2f} mag/arcsec2"]
                    + self._airy_disk_lines(reference_wl)
                    + self._template_colour_lines() + self._rv_dispersion_lines()
                    + self._spectrum_geometry_lines(spectrum))
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
                    reference_wavelength_aa=self._wl_to_aa(self.reference_wavelength_var.get()))
                selected_saturation = saturation[idx]
            horizon = self._current_horizon()
            if plot_args[0] == "photometry":
                self.plot_window = show_photometry_plot(self, *plot_args[1:], horizon=horizon)
            else:
                self.plot_window = show_spectroscopy_plot(self, *plot_args[1:], horizon=horizon)
            self._show_results_window()
            if target_snr is not None and selected_saturation != "NONE":
                self.status_var.set("Complete - target S/N exposure saturates; see max safe exposure in Results / CSV.")
            else:
                self.status_var.set("Complete")
            self._save_config()
            self._save_session_outputs()
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
        assumptions_frame = ttk.Frame(tabs)
        outputs_frame = ttk.Frame(tabs)
        tabs.add(result_frame, text="Selected-time result")
        tabs.add(time_frame, text="Time series")
        tabs.add(assumptions_frame, text="Assumptions")
        tabs.add(outputs_frame, text="Outputs")

        self.tree = ttk.Treeview(result_frame, show="headings")
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

        self.assumptions_text = scrolledtext.ScrolledText(assumptions_frame, bg="#f6f6f6", wrap="word")
        self.assumptions_text.pack(fill="both", expand=True)
        self.outputs_text = scrolledtext.ScrolledText(outputs_frame, bg="#f6f6f6", wrap="word")
        self.outputs_text.pack(fill="both", expand=True)
        # Backwards-compatible alias: some code paths still reference info_text.
        self.info_text = self.assumptions_text

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
        sky_source = {"ing": "ING dark table + van Rhijn airglow + zodiacal(|beta|) + starlight(|b|) "
                             "+ K&S Moon + twilight/Weaver",
                      "fixed_ab": f"fixed AB {self.sky_var.get()} mag/arcsec2 (observed)",
                      "sqm": f"SQM {self.sqm_var.get()} V mag/arcsec2 zenith + band colours + K&S Moon",
                      }.get(self.sky_model_var.get(), self.sky_model_var.get())
        detector_source = (f"gain table setting {self.gain_setting_var.get()}" if self.gain_table_rows
                           else "manual entry")
        lines = [
            "RUN PARAMETERS",
            "--------------",
            f"target            : {target_label}  ICRS {ra:.6f} {dec:+.6f} deg",
            f"time              : {utc:%Y-%m-%d %H:%M} UTC  |  {local:%Y-%m-%d %H:%M} {self._local_timezone_label()}",
            f"altitude/airmass  : {altitude:.2f} deg / {airmass:.3f}  (Pickering 2002, refracted; none below 5 deg)",
            f"parallactic angle : {parallactic:+.2f} deg at this time (N through E; "
            f"varies across the night — see the CSV time series)",
            f"pixel scale       : {self._plate_scale_arcsec_pix():.3f} arcsec/pixel "
            f"(206265 x pixel / focal length)",
        ]
        if sky_mag_arcsec2 is not None:
            lines.append(f"sky brightness    : {sky_mag_arcsec2:.2f} mag/arcsec2 (observing band, selected time)")
        lines += [
            f"sky model         : {self.sky_model_var.get()} = {sky_source}",
            f"reference mag     : {self.mag_var.get()} {self.mag_system_var.get()} in {self.reference_band_var.get()}",
            f"observing filter  : {self.band_var.get()}",
            f"template          : {self.selected_star.name.strip()}  mv0={self.selected_star.mv0:.3f}  "
            f"(visual-zero convention, then rescaled to the reference magnitude)",
            f"PSF               : {self.psf_model_var.get()}"
            + (f" beta={self.moffat_beta_var.get()}" if self.psf_model_var.get() == "moffat" else ""),
            f"detector          : {detector_source}; gain {self.gain_var.get()} e-/ADU, "
            f"RN {self.readnoise_var.get()} e-, FWC {self.fullwell_var.get()} e-, "
            f"{self.sensor_type_var.get()}"
            + (f" channel {self.osc_channel_var.get()}"
               if self.sensor_type_var.get().strip().lower() in ("color", "colour", "osc") else ""),
            "noise terms       : photon, sky, dark, read, scintillation (Young), ADC quantization (g/sqrt(12))"
            + (f", sky-subtraction (n_sky={self.sky_annulus_var.get()} px)"
               if float(self.sky_annulus_var.get() or 0.0) > 0 else ""),
            "extinction        : zenith curve ^ airmass on source; sky used as observed (not re-extinguished)"
            + ("; telluric O2/H2O bands included" if self.telluric_var.get() == "1" else ""),
            "sky physics       : zodiacal(|beta|, elongation), airglow van Rhijn x slant extinction, "
            "K&S Moon x (d_mean/d)^2; twilight on geometric Sun altitude",
            "saturation        : peak pixel from unextracted PSF/LSF; flags FULL_WELL / ADC / BOTH",
        ]
        rv = float(self.radial_velocity_var.get() or 0.0)
        ebv = float(self.ebv_var.get() or 0.0)
        if rv != 0.0 or ebv != 0.0:
            lines.append(f"template transform: RV {rv:+.1f} km/s, E(B-V) {ebv:.3f} (CCM89, R_V=3.1), "
                         "applied before calibration")
        if float(self.guiding_rms_var.get() or 0.0) > 0:
            lines.append(f"guiding blur      : rms {self.guiding_rms_var.get()} arcsec added "
                         "in quadrature to the seeing FWHM")
        if self.mode_var.get() == "spectroscopy":
            lines += [
                f"slit orientation  : {self.slit_orientation_var.get()}"
                + ("" if self.slit_orientation_var.get() == "parallactic"
                   else " (per-wavelength Filippenko dispersion offset applied)"),
                f"spectrograph      : {self.spec_geometry_status_var.get()}",
            ]
            if self.spectroscopy_mode_var.get() == "slitless":
                lines.append("slitless          : full-band sky per pixel (uniform source stays uniform "
                             "under dispersion); pending validation on a calibrated instrument")
            if self.grating_efficiency_curve is not None:
                lines.append("grating efficiency: wavelength-dependent curve loaded (overrides the scalar)")
        self.assumptions_text.delete("1.0", tk.END)
        self.assumptions_text.insert("1.0", "\n".join(lines) + "\n")

    def _plate_scale_arcsec_pix(self):
        try:
            return 206265.0 * (float(self.pixel_var.get()) * 1e-3) / float(self.focal_var.get())
        except (ValueError, ZeroDivisionError, tk.TclError):
            return float("nan")

    def _airy_disk_lines(self, pivot_aa):
        """Airy-disk diameter 2.44 lambda/D in arcsec and detector pixels."""
        try:
            diam_m = float(self.diam_var.get()) * 1e-3
            lam_m = float(pivot_aa) * 1e-10
            if diam_m <= 0 or lam_m <= 0:
                return []
            diameter_arcsec = 2.44 * lam_m / diam_m * 206265.0
            plate = self._plate_scale_arcsec_pix()
            diameter_pix = diameter_arcsec / plate if plate > 0 else float("nan")
            return [f"Airy disk         : {diameter_arcsec:.3f} arcsec diameter "
                    f"({diameter_pix:.2f} px) at {float(pivot_aa):.0f} A"]
        except (ValueError, ZeroDivisionError, tk.TclError):
            return []

    @staticmethod
    def _frame_flag_counts(flags):
        """(linear, non-linear, saturated) counts from a list of SAT flags."""
        linear = nonlin = saturated = 0
        for flag in flags:
            if flag in (None, "NOT_VISIBLE"):
                continue
            if flag == "NONE":
                linear += 1
            elif flag == "NON_LIN":
                nonlin += 1
            else:
                saturated += 1
        return linear, nonlin, saturated

    SPEED_OF_LIGHT_KMS = 299792.458

    def _template_colour_lines(self):
        """(B-V)0 of the unreddened template and (B-V) after E(B-V).

        The synthetic colours are computed from the *full* template spectrum
        through the Bessell B and V passbands.  By construction the reddened
        minus unreddened difference recovers the entered E(B-V), which is a
        useful consistency check shown alongside.
        """
        if self.star_spec is None or self.filter_resp_data is None:
            return []
        try:
            b_profile = fcat.load_filter_profile(self.filter_resp_data, "Bessell.B", "Vega")
            v_profile = fcat.load_filter_profile(self.filter_resp_data, "Bessell.V", "Vega")
        except (KeyError, FileNotFoundError, ValueError):
            return []
        ebv = float(self.ebv_var.get() or 0.0)

        def colour(template):
            try:
                m_b = synthetic_magnitude(template, b_profile.transmission,
                                          b_profile.zero_point_jy, b_profile.detector_type)
                m_v = synthetic_magnitude(template, v_profile.transmission,
                                          v_profile.zero_point_jy, v_profile.detector_type)
            except ValueError:
                return None
            return m_b - m_v

        bv0 = colour(self.star_spec)
        if bv0 is None:
            return []
        line = f"template colour   : (B-V)0 = {bv0:+.3f} (unreddened, full template, Bessell B/V)"
        if ebv != 0.0:
            reddened = transformed_template(self.star_spec, ebv=ebv)
            bv = colour(reddened)
            if bv is not None:
                line += (f"; (B-V) = {bv:+.3f} reddened by E(B-V)={ebv:.3f} "
                         f"(Δ = {bv - bv0:+.3f})")
        return [line]

    def _rv_dispersion_lines(self):
        """RV shift in pixels and the km/s-per-pixel scale at 4000-7000 Å.

        The RV displaces the spectrum by Δλ = λ v/c; expressed in detector
        pixels this is (λ v/c) / (Å per pixel).  The velocity sampling is the
        inverse, c·(Å per pixel)/λ [km/s per pixel].  The dispersion is the
        slitless Å/pixel, or in slit mode (λ/R)/(pixels per element).
        """
        rv = float(self.radial_velocity_var.get() or 0.0)
        wavelengths = np.array([4000.0, 5000.0, 6000.0, 7000.0])
        mode = self.spectroscopy_mode_var.get()
        try:
            if mode == "slitless":
                grating_lines = float(self.grating_lines_var.get() or 0.0)
                if grating_lines > 0:
                    aa_per_pix = np.full(wavelengths.size, grating_dispersion_aa_per_pixel(
                        grating_lines, float(self.grating_distance_var.get()),
                        float(self.pixel_var.get())))
                else:
                    aa_per_pix = np.full(wavelengths.size, float(self.slitless_dispersion_var.get()))
            else:
                resolution = float(self.resolution_var.get())
                sampling = float(self.sampling_var.get())
                aa_per_pix = (wavelengths / resolution) / max(sampling, 1e-9)
        except (ValueError, ZeroDivisionError):
            return []
        kms_per_pix = self.SPEED_OF_LIGHT_KMS * aa_per_pix / wavelengths
        shift_pix = (wavelengths * rv / self.SPEED_OF_LIGHT_KMS) / aa_per_pix
        vel = "  ".join(f"{w:.0f}Å:{k:.1f}" for w, k in zip(wavelengths, kms_per_pix))
        lines = [f"RV sampling       : {vel}  (km/s per pixel)"]
        if rv != 0.0:
            disp = "  ".join(f"{w:.0f}Å:{s:+.2f}" for w, s in zip(wavelengths, shift_pix))
            lines.append(f"RV shift          : {disp}  (pixels for RV {rv:+.1f} km/s)")
        return lines

    def _append_info_outputs(self, lines):
        """Write the selected-time output block to the Outputs tab."""
        block = ["RUN OUTPUTS", "-----------"]
        block += [line for line in lines if line]
        for extra in (self.stack_plan_var.get(), self.differential_var.get()):
            if extra and not extra.startswith("Stack plan appears"):
                block.append(extra)
        self.outputs_text.delete("1.0", tk.END)
        self.outputs_text.insert("1.0", "\n".join(block) + "\n")

    # Column key -> visible heading; only keys present in the result frame are
    # shown, so photometric and spectroscopic runs each display every quantity
    # the engine produced (the CSV export always carries the full table).
    _RESULT_COLUMNS = (
        ("mag", "Magnitude"), ("wavelength_aa", "Wavelength [Å]"),
        ("resolution_element_aa", "Resel [Å]"), ("effective_resolution_R", "R"),
        ("photons_source_es", "Source e-/s"), ("photons_sky_es", "Sky e-/s"),
        ("snr", "S/N"), ("sigma_ew_mangstrom", "σ(EW) [mÅ]"),
        ("scintillation_noise_e", "Scint [e-]"), ("digitization_noise_e", "ADC noise [e-]"),
        ("n_pixels", "N pix"), ("peak_e_unclipped", "Peak [e-]"), ("adu", "Peak ADU"),
        ("saturation_flag", "Sat"), ("max_unsaturated_exptime_s", "Max t [s]"),
        ("estimated_observing_magnitude", "Std obs mag"),
        ("instrumental_response_magnitude", "Instr mag"), ("sky_mag_arcsec2", "Sky mag/\"²"),
        ("sky_subtraction_factor", "Sky-sub ×"),
    )

    def _display_table(self, frame, xcol, grey_from_row=None, separator_before=None):
        """Render the result frame in the tree.

        ``grey_from_row`` greys every row from that index on (the photometry
        magnitude sweep); ``separator_before`` inserts a dashed divider row
        just before that index.
        """
        self._ensure_results_window()
        available = [(key, heading) for key, heading in self._RESULT_COLUMNS if key in frame.columns]
        columns = tuple(key for key, _ in available)
        self.tree.configure(columns=columns)
        self.tree.tag_configure("sweep", foreground="#8a8a8a")
        self.tree.tag_configure("separator", foreground="#b0b0b0")
        for key, heading in available:
            if key == "mag":
                heading = f"Magnitude [{self.mag_system_var.get()}]"
            self.tree.heading(key, text=heading)
            self.tree.column(key, width=118 if key in (xcol, "saturation_flag") else 105,
                             minwidth=70, anchor="e")
        self.tree.delete(*self.tree.get_children())
        for position, (_, row) in enumerate(frame.iterrows()):
            if separator_before is not None and position == separator_before:
                self.tree.insert("", "end", values=["─" * 6 for _ in columns], tags=("separator",))
            tags = ("sweep",) if grey_from_row is not None and position >= grey_from_row else ()
            self.tree.insert("", "end",
                             values=[self._format_table_value(row[key]) for key in columns], tags=tags)

    def _open_plot(self):
        if self.plot_window is None or not self.plot_window.winfo_exists():
            messagebox.showinfo("No plot", "Run the ETC first."); return
        self.plot_window.deiconify(); self.plot_window.lift(); self.plot_window.focus_force()

    def _export_dir(self):
        """Default the save dialogs to the active session directory, if any."""
        return str(self.session_dir) if self.session_dir is not None else ""

    def _export_csv(self):
        if self.result_df is None:
            messagebox.showinfo("No result", "Run the ETC first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialdir=self._export_dir())
        if path: self.result_df.to_csv(path, index=False)

    def _export_time_series_csv(self):
        if self.time_series_df is None:
            messagebox.showinfo("No time series", "Run the ETC first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="etc_time_series.csv",
                                            filetypes=[("CSV", "*.csv")], initialdir=self._export_dir())
        if path:
            self.time_series_df.to_csv(path, index=False)

    def _on_exit(self):
        self._save_config(); self.destroy(); sys.exit(0)


if __name__ == "__main__":
    ETCGUI().mainloop()
