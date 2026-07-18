# SPETC v10.0

Spectro-Photometry Exposure Time Calculator.

Version 10.0

2007-2026

Mauro Barbieri (`mauro.barbieri@pm.me`)

## Run

```bash
python3 -m pip install -r requirements.txt
python3 etc_gui.py
```

The data directory must contain `interpola.db.csv`, the template spectra it
references, `qe.dat`, and a SVO filter list. The usual current layout is
`data/filters.list`, with entries such as `filters/Bessell.V_Vega` and
profiles in `data/filters/`; the older `data/filters/filters.list` layout is
also accepted. Entries with or without the `filters/` prefix work. Each
logical filter needs its `_Vega` and `_AB` profiles. The obsolete
`resp_filters.dat` is no longer read.

`BLANK.xml` is included with this release. Copy it to `data/filters/BLANK.xml`
to enable the **BLANK** selector without editing `filters.list`. It is a
unity-transmission, 3000--26000 Å synthetic passband. Because an unfiltered
measurement has no defined Vega magnitude, its calibration is explicitly the
AB 3631-Jy zero point even if the magnitude-system selector displays Vega.

An optional target name can be resolved through SIMBAD with the explicit
**Resolve name** button; this requires an internet connection. Resolution
writes decimal ICRS coordinates into the RA and Dec fields. Running the ETC
never makes a network request: it uses the coordinates currently shown in
those fields, which may instead be entered directly in decimal or sexagesimal
form.

## Configuration

`etc_user_config.json` is read at startup and is rewritten after each
successful calculation and on normal exit. It contains the telescope,
detector, atmospheric, sky, spectroscopic and UI settings, so it can be copied
to create a preset for another instrument. The default preset is Asiago
(Cima Ekar); the shipped data-directory entry is resolved beside `etc_gui.py`,
so starting the program from another working directory does not lose `data/`.

v10 uses explicit wavelength units for user two-column QE and instrument-response
files (`Angstrom`, `nm`, or `um`), saved in the configuration/profile. SVO XML
wavelength units and `DetectorType` are honoured for synthetic magnitude
calculations. The ETC now rejects missing wavelength coverage instead of
silently returning zero counts.

If `data/earth_atmospheric_transmission.fits` is available, the local Paranal
table is read through `WAVEAIR` and `EXTINCTION`. `EXTINCTION` is the
coefficient in mag per airmass and is converted to a zenith transmission as
`10^(-0.4 k_lambda)`; the ETC then applies the selected target airmass once.
Its actual wavelength coverage is a hard validity limit. A calculation or
live response preview outside the file coverage fails explicitly rather than
extrapolating or reverting to broad-band extinction. Direct-transmission FITS
tables with wavelength (`wavelength`, `wave`, `lambda`, `lam`, or `wl`) and
transmission (`transmission`, `trans`, or `throughput`) columns are also
accepted. It is used both by the ETC and the live second-column plot of
template × filter × QE × optics × Earth atmosphere.
A simple two-column FITS image is also accepted; use `CUNIT1` in its header
to state the wavelength unit. `NIGHT_SKY_EM` is deliberately not used yet:
the supplied combined table does not specify its physical flux unit, so using
it as a sky-count rate would not be defensible.

The main window is organised into five operational columns: observation/site
and target; telescope/detector and conditions; photometry/spectroscopy
settings; filter/template selection; and run actions. Numerical output
opens in a separate reusable Results window. Scientific plots open in a native
Matplotlib/Tk window with pan/zoom/save toolbar, manual spectrum-time slider,
autoscaling, and optional explicit axis limits.

Use **Save instrument profile** to write an instrument JSON file and copy its
current QE curve beside it as `<profile>_qe.dat`. Optional calibrated
instrument-response and slit-width/resolving-power curves are copied beside it
as well. The profile records their relative filenames, making the set portable.
The most recently saved or loaded profile is restored automatically at startup. The site selector merges
the built-in observatories with the persistent `etc_sites.json`; **Save current
site** adds or updates an entry there, while **Import site file** merges JSON
site entries into the same local list.

The Filter Selector has independent **reference magnitude** and **observing**
filters. The first calibrates the target flux; the second is the physical
response used for source and sky counts in both photometry and spectroscopy.
Spectroscopy selects **BLANK** by default, but an order-separation filter can
be selected when required. It includes a compact gridded
display of the observing SVO transmission curve, its pivot wavelength,
effective width, FWHM and zero point. The selected template has a separate
gridded flux-distribution preview. The target magnitude selector defaults to
Vega; AB is also supported.

The photometric selected-time result includes both the predicted standard
magnitude in the observing filter and the explicitly labelled instrumental
response magnitude, the latter weighted by filter, QE, optics throughput and
the current source atmosphere. They are intentionally not treated as the same
quantity.

An optional two-column wavelength/transmission curve can be selected for the
telescope/instrument. It multiplies the scalar optics throughput and QE, is
saved alongside an instrument profile as `<profile>_throughput.dat`, and is
automatically restored with that profile.

## Amateur-oriented features (v10.1)

* **Coverage policy**: observing-side curves (template, QE) are zero-filled
  outside their coverage; the **calibration bands stay strict** — a template
  must cover at least 99% of the response-weighted reference and visual
  bands, because a truncated calibration integral silently rescales the
  whole spectrum. Atmospheric transmission curves are **edge-extended**
  outside their tabulated range (an atmosphere is never opaque there).
* **Slit-spectrograph geometry helper**: grating lines/mm, collimator and
  camera focal lengths (plus the telescope FL, slit width, seeing and the
  S/N reference wavelength already entered) compute R via the Littrow
  grating equation and fill the R field, which remains editable.
* **CMOS gain table**: a per-camera file (gain setting, e-/ADU, read noise,
  full well; `#` comments) loads into a gain-setting selector that fills
  the three detector fields, matching how CMOS cameras actually behave.
* **SQM / Bortle sky mode**: enter your zenith SQM reading (V mag/arcsec²),
  or pick a Bortle class to fill a typical value; band colours and the
  Krisciunas & Schaefer Moon model are applied on top.
* **Stack planner**: with a target S/N, the ETC reports N × sub-exposure
  (sub capped by saturation or your preference), total time, the
  read-noise penalty versus one ideal exposure, and the frame length above
  which the background dominates read noise.
* **Differential photometry**: an optional comparison-star magnitude turns
  the result into an error budget in mmag per frame (both stars' full
  noise, scintillation treated as uncorrelated).
* **σ(EW)**: every spectroscopic result includes the Cayrel (1988)
  equivalent-width uncertainty per wavelength (mÅ), reported at the S/N
  reference wavelength.
* **`make_filter_profile.py`**: converts any two-column filter transmission
  (Astrodon/Astronomik/Baader RGB, narrowband, digitized curves; 0–1 or
  percent) into calibrated `_Vega`/`_AB` XML profiles. The Vega zero point
  is computed synthetically from the bundled CALSPEC Vega spectrum — the
  same convention SVO uses (verified to ~1% against the shipped SVO
  profiles).

## Physical conventions

* Internally, wavelengths are Angstrom and all radiometric calculations use
  `astropy.units` and Astropy physical constants. User curve units are explicit;
  SVO units are read from the VOTable metadata.
* The target magnitude can be **Vega** (default) or **AB** in a reference
  filter independent from the observing filter. Template files retain their
  calibrated `F_lambda` values. The catalogue `mv0` tells the ETC the visual
  magnitude represented by the file; it first applies the original Fortran
  `10^(0.4 mv0)` visual-zero conversion, derives any required synthetic colour
  from the template, and only then applies the target reference magnitude.
  When the reference response is the visual-V response, no synthetic
  integration is attempted: this reduces exactly to the Fortran scale factor
  `10^(-0.4 (V_target - mv0))`.
* The sky background selector has two modes. `ing` uses the supplied
  solar-minimum cycle, U--K dark-sky table and Weaver (1947) daylight V-band
  table; it is treated as an observed ground sky and varies with time.
  `fixed_ab` uses the entered observed AB sky surface brightness. In the
  ING mode, spectroscopy receives a U--K continuum colour model whose
  normalization follows the supplied ING model and whose Moon contribution
  is added only while the Moon is above the horizon. Both are
  explicitly not attenuated again as if they originated above the atmosphere.
  The daylight colour treatment is a broad-band approximation; a
  site-specific spectral sky model remains a later improvement.
* Telescope area is `pi/4 (D^2 - d^2)`. The setup-specific extra `/4` in
  `interpola_spad.f90` is not present.
* `efficiency` is the optical/instrument throughput excluding QE. QE is read
  separately from `qe.dat`. Until a site-specific extinction curve is added,
  zenith atmospheric transmission is constructed from the supplied U--K
  `skyext` broad-band values and applied as `T_zenith^airmass` to the source.
* The PSF is selectable: **Gaussian** or **Moffat** (beta > 1, default 2.5).
  The Moffat option reproduces the extended wings of real seeing-limited
  images in aperture, slit and extraction losses (its slit coupling uses the
  exact Student-t marginal of the circular Moffat profile).
* Photometry includes PSF aperture losses, sky aperture area, dark current,
  read noise, **Young-law scintillation** and **ADC quantization**
  (gain/sqrt(12)) noise. Sources can be **point** or **extended**
  (galaxy/nebula/planet): extended sources keep the entered integrated
  magnitude spread uniformly over a stated angular area, with area-fraction
  aperture capture and a surface-brightness peak pixel (valid when the source
  is much larger than the seeing disc). Spectroscopy supports `slit` and
  `slitless` modes. Slit mode uses both slit-width and finite
  extraction-height losses; with the slit **not** at the parallactic angle it
  additionally applies the per-wavelength Filippenko (1982)
  atmospheric-dispersion offset loss (worst-case geometry). Slitless mode
  uses cross-dispersion extraction geometry and derives effective resolution
  from dispersion, seeing, pixel scale, intrinsic LSF width and the
  atmospheric-dispersion smear. A transmission grating such as the
  **Star Analyser 100/200** is described physically by its groove density and
  grating-to-sensor distance (dispersion `1e4 p_um / (L_mm n_mm m)` A/pixel;
  an SA100 at 42 mm on 4.63 um pixels gives ~11 A/pixel) plus an optional
  first-order efficiency. Source rates and S/N are per resolution element.
  Slitless remains explicitly **experimental** until it is validated with a
  calibrated instrument.
* The ING sky model is position-dependent: van Rhijn airglow (solar-cycle
  scaled), zodiacal light at the field's |ecliptic latitude| and integrated
  starlight at its |galactic latitude| correct the tabulated U--K dark sky;
  the Moon term uses the published Krisciunas & Schaefer (1991) scattering
  function. The Results window reports the sky surface brightness used.
* Saturation output preserves unclipped peak electrons/ADU and labels the
  cause as `NONE`, `FULL_WELL`, `ADC`, or `BOTH`. Time-series CSV files include
  the reference-bin saturation flag, peak electron prediction and the maximum
  unsaturated single-frame exposure. Target-S/N exposure requests are not
  silently treated as feasible when this limit is exceeded.
* High-resolution spectroscopic time series evaluate only the reference bin at
  every time point; full spectra are calculated only for the selected time and
  the manual native plot slider positions.
* Azimuth is N=0, E=90, S=180, W=270. Visibility uses Astropy `AltAz` with
  ERFA atmospheric refraction at ISA site pressure/temperature; airmass is
  Pickering (2002) on the apparent altitude, which stays physical at low
  altitude where plane-parallel sec z overestimates. The parallactic angle is
  reported for every time sample (slit at that angle avoids
  atmospheric-dispersion slit losses). Airmass is not calculated or plotted
  below altitude 5 degrees. Plot ticks
  show both UTC and local time. Local conversion uses either a selectable IANA
  timezone (default `Europe/Rome`, including daylight-saving changes) or a
  mutually exclusive fixed UTC offset.

The spectroscopic ETC performs one calculation at the manually selected time.
The native plot window provides a manual, non-autoplay slider through sampled
times in the selected observing interval. If the selected time is daytime,
the interval and sky-path plot include the Sun; otherwise the Sun is omitted.
The Moon is shown only above the horizon. Both modes plot S/N versus time for
a fixed exposure, or required exposure versus time when a target S/N is
entered. The Results window shows UTC/local datetime, MJD, elevation, azimuth,
S/N, exposure time, airmass, the passband-effective extinction and its
per-airmass colour term, the absorption-adjusted apparent reference magnitude,
local timezone, and saturation information. Spectroscopy also records the S/N
reference wavelength.

## Included numerical smoke test

After installing dependencies, run:

```bash
python3 self_check.py
python3 tests/test_spectral_pipeline.py
```

It uses synthetic flat spectra and a top-hat bandpass. It checks that the
default 358-mm reference setup produces an optical S/N of order hundreds per
R=10000 resolution element, not thousands, and that the broad-band result for
a 60-s AB=5 target is scintillation-limited near S/N ~ 1.3e3 (the Young-law
fractional rms caps bright-star photometry; a pure Poisson model would claim
~2e4). The pipeline test also includes regression checks of the
Krisciunas & Schaefer scattering function, position-dependent sky, Pickering
airmass, parallactic angle, Moffat PSF, Filippenko dispersion, scintillation
and quantization noise scales, Star Analyser dispersion and extended-source
photometry.
