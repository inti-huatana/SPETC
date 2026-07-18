"""Compact native Tk/Matplotlib plot windows for SPETC."""

import tkinter as tk
from tkinter import ttk
from datetime import datetime

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import dates as mdates


AZ_TICKS = [0, 45, 90, 135, 180, 225, 270, 315, 360]
AZ_LABELS = ["N", "45", "E", "135", "S", "225", "W", "315", "N"]


def _time_ticks(axis, track):
    utc, local = track["utc_datetime"], track["local_datetime"]
    ids = np.unique(np.linspace(0, len(utc) - 1, min(8, len(utc)), dtype=int))
    axis.set_xticks(mdates.date2num([utc[i] for i in ids]))
    axis.set_xticklabels([f"UT {utc[i]:%H:%M}\nLT {local[i]:%H:%M}" for i in ids], fontsize=8)


def _above(az, alt):
    az, alt = np.asarray(az, float).copy(), np.asarray(alt, float).copy()
    az[alt < 0], alt[alt < 0] = np.nan, np.nan
    jumps = np.abs(np.diff(az)) > 180
    az[1:][jumps], alt[1:][jumps] = np.nan, np.nan
    return az, alt


class ETCPlotWindow(tk.Toplevel):
    def __init__(self, master, kind, track, time_values, time_label, selected_idx, *, band_label=None,
                 spectra=None, slider_indices=None):
        super().__init__(master)
        self.kind, self.track, self.time_values = kind, track, np.asarray(time_values, float)
        self.time_label, self.selected_idx = time_label, int(selected_idx)
        self.spectra = spectra or {}
        self.slider_indices = list(slider_indices or [])
        self.title("SPETC scientific plots")
        self.geometry("1050x920" if kind == "photometry" else "1050x1080")
        self.minsize(760, 620)
        self.figure = Figure(figsize=(9.4, 8.4 if kind == "photometry" else 10.0), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        toolbar_frame = ttk.Frame(self); toolbar_frame.pack(fill="x", padx=6, pady=(5, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.pack(anchor="w")
        controls_frame = ttk.LabelFrame(self, text="Axis range controls", padding=4)
        controls_frame.pack(fill="x", anchor="w", padx=6, pady=(2, 0))
        self._build_controls(controls_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        self._build_axes(band_label)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

    def _build_controls(self, parent):
        controls = parent
        self.axis_choice = tk.StringVar(value="Metric")
        self.xmin, self.xmax, self.ymin, self.ymax = (tk.StringVar() for _ in range(4))
        ttk.Label(controls, text="Plot panel").grid(row=0, column=0, sticky="w", padx=2)
        ttk.Combobox(controls, textvariable=self.axis_choice, state="readonly", width=14,
                     values=("Metric", "Altitude", "Sky path", "Spectrum", "Source rate", "S/N spectrum")).grid(row=1, column=0, padx=2, sticky="w")
        for col, var, text in ((1, self.xmin, "X minimum"), (2, self.xmax, "X maximum"),
                               (3, self.ymin, "Y minimum"), (4, self.ymax, "Y maximum")):
            ttk.Label(controls, text=text).grid(row=0, column=col, sticky="w", padx=2)
            ttk.Entry(controls, textvariable=var, width=12).grid(row=1, column=col, padx=2, sticky="w")
        ttk.Button(controls, text="Apply range", command=self._apply_range).grid(row=1, column=5, padx=(8, 2))
        ttk.Button(controls, text="Autoscale", command=self._autoscale).grid(row=1, column=6, padx=2)
        self.range_status = tk.StringVar(value="Leave a limit blank to retain it. Time-panel X limits accept ISO UTC, for example 2026-07-17T22:00.")
        ttk.Label(controls, textvariable=self.range_status, foreground="gray35").grid(
            row=2, column=0, columnspan=7, sticky="w", padx=2, pady=(3, 0))
        if self.kind == "spectroscopy":
            ttk.Label(controls, text="Manual spectrum time:").grid(row=3, column=0, sticky="e")
            self.slider = ttk.Scale(controls, from_=0, to=max(len(self.slider_indices) - 1, 0), orient="horizontal",
                                    command=None)
            self.slider.grid(row=3, column=1, columnspan=5, sticky="ew", padx=2)
            initial = self.slider_indices.index(self.selected_idx) if self.selected_idx in self.slider_indices else 0
            self.slider.set(initial)
            self.slider_label = ttk.Label(controls, text="")
            self.slider_label.grid(row=3, column=6, sticky="w")

    def _build_axes(self, band_label):
        self.figure.clear()
        if self.kind == "photometry":
            grid = self.figure.add_gridspec(3, 1, height_ratios=[1.15, .85, 1.0])
            self.ax_metric = self.figure.add_subplot(grid[0]); self.ax_alt = self.figure.add_subplot(grid[1]); self.ax_sky = self.figure.add_subplot(grid[2])
            self.ax_source = self.ax_snr = None
        else:
            grid = self.figure.add_gridspec(5, 1, height_ratios=[.92, .92, 1.0, .75, .9])
            self.ax_source = self.figure.add_subplot(grid[0]); self.ax_snr = self.figure.add_subplot(grid[1])
            self.ax_metric = self.figure.add_subplot(grid[2]); self.ax_alt = self.figure.add_subplot(grid[3]); self.ax_sky = self.figure.add_subplot(grid[4])
        self._draw_common(band_label)
        if self.kind == "spectroscopy":
            self._draw_spectrum(self.selected_idx)
            # Do not invoke the callback until the spectrum axes and markers
            # exist: ttk may emit a callback for the initial ``set`` above.
            self.slider.configure(command=self._slider_changed)
        self.canvas.draw_idle()

    def _draw_common(self, band_label):
        track, utc = self.track, self.track["utc_datetime"]
        self.ax_metric.plot(utc, self.time_values, color="#1f77b4", label=self.time_label)
        self.metric_marker, = self.ax_metric.plot([utc[self.selected_idx]], [self.time_values[self.selected_idx]], "o", color="#d62728")
        self.ax_metric.set_ylabel(self.time_label); self.ax_metric.grid(True, color="#dddddd"); self.ax_metric.set_title(
            f"{self.time_label} versus time" + (f" - {band_label}" if band_label else ""))
        self.ax_airmass = self.ax_metric.twinx()
        self.ax_airmass.plot(utc, track["airmass_target"], "--", color="#d62728", linewidth=1, label="Airmass")
        self.ax_airmass.set_ylabel("Airmass"); self.ax_airmass.set_ylim(bottom=0)
        _time_ticks(self.ax_metric, track)
        altitude = np.asarray(track["alt_target"], float); altitude[altitude < 0] = np.nan
        self.ax_alt.plot(utc, altitude, color="#1f77b4")
        self.ax_alt.axhline(0, color="black", linewidth=.8); self.ax_alt.set_ylim(0, 90); self.ax_alt.set_ylabel("Altitude [deg]")
        self.ax_alt.set_title("Observable altitude versus time"); self.ax_alt.grid(True, color="#dddddd"); _time_ticks(self.ax_alt, track)
        self._draw_sky()

    def _draw_sky(self):
        track = self.track
        self.ax_sky.clear()
        for az, alt, label, color in ((track["az_target"], track["alt_target"], "Target", "#1f77b4"),
                                      (track["az_sun"], track["alt_sun"], "Sun (above horizon)", "#e6a700"),
                                      (track["az_moon"], track["alt_moon"], "Moon (above horizon)", "#777777")):
            x, y = _above(az, alt); self.ax_sky.plot(x, y, label=label, color=color, linewidth=1.3)
        self.sky_marker, = self.ax_sky.plot([track["az_target"][self.selected_idx]], [track["alt_target"][self.selected_idx]],
                                             marker="*", color="#d62728", markersize=10, linestyle="None")
        self.ax_sky.set(xlim=(0, 360), ylim=(0, 90), xlabel="Azimuth", ylabel="Altitude [deg]", title="Sky path")
        self.ax_sky.set_xticks(AZ_TICKS, AZ_LABELS); self.ax_sky.grid(True, color="#dddddd"); self.ax_sky.legend(fontsize=8, ncol=2)

    def _draw_spectrum(self, index):
        spectrum = self.spectra[index]
        self.ax_source.clear(); self.ax_snr.clear()
        self.ax_source.plot(spectrum["wavelength_aa"], spectrum["photons_source_es"], color="#1f77b4")
        self.ax_snr.plot(spectrum["wavelength_aa"], spectrum["snr"], color="#2ca02c")
        for axis, ylabel, title in ((self.ax_source, "e⁻/s/resel", "Detected source rate"),
                                    (self.ax_snr, "S/N/resel", "S/N per resolution element")):
            axis.set(ylabel=ylabel, xlabel="Wavelength [Å]", title=title)
            axis.minorticks_on(); axis.grid(True, which="major", color="#dddddd"); axis.grid(True, which="minor", color="#eeeeee")
        utc, local = self.track["utc_datetime"][index], self.track["local_datetime"][index]
        self.slider_label.configure(text=f"UT {utc:%H:%M} / LT {local:%H:%M}")

    def _slider_changed(self, value):
        if not self.slider_indices:
            return
        index = self.slider_indices[int(round(float(value)))]
        self.selected_idx = index
        self._draw_spectrum(index)
        self.metric_marker.set_data([self.track["utc_datetime"][index]], [self.time_values[index]])
        self.sky_marker.set_data([self.track["az_target"][index]], [self.track["alt_target"][index]])
        self.canvas.draw_idle()

    def _selected_axis(self):
        mapping = {"Metric": self.ax_metric, "Altitude": self.ax_alt, "Sky path": self.ax_sky,
                   "Spectrum": self.ax_snr, "Source rate": self.ax_source, "S/N spectrum": self.ax_snr}
        return mapping.get(self.axis_choice.get()) or self.ax_metric

    def _apply_range(self):
        axis = self._selected_axis()
        try:
            x = [self._parse_axis_limit(v.get(), axis, is_x=True) for v in (self.xmin, self.xmax)]
            y = [float(v.get()) if v.get().strip() else None for v in (self.ymin, self.ymax)]
            current_x, current_y = axis.get_xlim(), axis.get_ylim()
            if any(item is not None for item in x):
                axis.set_xlim(x[0] if x[0] is not None else current_x[0],
                              x[1] if x[1] is not None else current_x[1])
            if any(item is not None for item in y):
                axis.set_ylim(y[0] if y[0] is not None else current_y[0],
                              y[1] if y[1] is not None else current_y[1])
            self.range_status.set("Applied selected axis limits.")
            self.canvas.draw_idle()
        except ValueError as exc:
            self.range_status.set(f"Range not applied: {exc}")

    def _parse_axis_limit(self, value, axis, *, is_x):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            if is_x and axis in (self.ax_metric, self.ax_alt):
                return mdates.date2num(datetime.fromisoformat(text.replace("Z", "+00:00")))
            raise ValueError("numeric limits are required; time panels also accept ISO UTC")

    def _autoscale(self):
        axis = self._selected_axis(); axis.relim(); axis.autoscale(True, axis="both")
        if axis is self.ax_sky: axis.set(xlim=(0, 360), ylim=(0, 90))
        self.range_status.set("Restored automatic limits for the selected panel.")
        self.canvas.draw_idle()


def show_photometry_plot(master, track, values, label, selected_idx, band_label):
    return ETCPlotWindow(master, "photometry", track, values, label, selected_idx, band_label=band_label)


def show_spectroscopy_plot(master, track, spectra, values, label, selected_idx, slider_indices):
    return ETCPlotWindow(master, "spectroscopy", track, values, label, selected_idx,
                         spectra=spectra, slider_indices=slider_indices)
