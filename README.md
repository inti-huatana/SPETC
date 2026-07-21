# SPETC v10.2

Spectro-Photometry Exposure Time Calculator.

Version 10.2

2007-2026

Mauro Barbieri (`mauro.barbieri@pm.me`)

Full documentation (compiled PDFs and LaTeX sources) is in `docs/`:
`SPETC_physics.pdf` (the models and formulae), `SPETC_user_guide.pdf`
(operation, every input file format specified in full) and
`SPETC_maintainer_guide.pdf` (architecture, modification recipes, Git
workflow, packaging).

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
The most recently saved or loaded profile is restored automatically at startup. All
sites live in a single `observatories.json` (name, lon/lat, elev, mag_sky,
timezone, utc_offset_h — so selecting a site sets the time reference
automatically); **Save site** adds or updates an entry, **Import…** merges
another JSON into it, and **Google Earth** opens the browser at the site
coordinates. The obsolete `etc_sites.json` is no longer used.

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
* **Wavelength- and airmass-dependent seeing** (opt-in checkbox): when
  enabled, the entered seeing is treated as the zenith V value and scaled as
  `FWHM = seeing_V x airmass^0.6 x (lambda/5000)^-0.2` (Kolmogorov turbulence;
  the convention used by ETC-42, CeSAM/LAM). Blue light is then more blurred
  than red — colouring the spectroscopic slit losses — and the seeing
  worsens as the target descends, which the whole-night time series now
  reflects. Off by default (flat seeing, unchanged).
* **Generic extra background** (e-/s/pixel): a catch-all Poisson background
  term for detector glow, stray/scattered light or ghosting that the
  physical model does not otherwise cover (cf. ETC-42's ExtraBackgroundNoise).
  It adds to the background noise and the peak-pixel/saturation prediction.
* **One-shot-colour (Bayer/OSC) sensors**: select `osc` and the channel
  (R/G/B). A single-channel extraction sees only that channel's share of
  the mosaic (G 1/2, R/B 1/4): aperture source and sky rates and the
  channel-pixel count scale accordingly, while the peak pixel does not (a
  centred channel pixel still receives the full local flux), so saturation
  stays correct. Supply the channel's *effective* QE (sensor QE × CFA dye
  transmission, digitizable from the maker's curves) as the QE curve.
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
  percent) into **full SVO-format VOTables** (`_Vega`/`_AB`), with the
  complete photdm utype/UCD annotation and every characterising quantity
  measured from the curve using the SVO definitions: WavelengthRef/Mean/
  Eff/Min/Max/Cen/Pivot/Peak/Phot, WidthEff, FWHM and Fsun (from the
  bundled CALSPEC solar spectrum). The Vega zero point is computed
  synthetically from the bundled CALSPEC Vega spectrum — the same
  convention SVO uses. Validated against the shipped 2MASS.H profile:
  pivot exact, Fsun −0.5%, Vega zero point +0.6%.
* **`spetc_batch.py`**: headless mode — `python3 spetc_batch.py
  examples/batch_photometry.json out/` runs the same engines without the
  GUI from a JSON run description and writes the result CSV plus a
  self-contained one-page HTML summary (configuration, sky, key numbers,
  stack plan, embedded S/N figure). Batch sky modes are `fixed_ab` and
  `sqm` (Moon/twilight terms need the GUI's time machinery and are not
  applied). Two example configurations ship in `examples/`.
* The target-S/N solver and the stack planner include the **scintillation
  variance** in closed form (it is linear in time, so it enters as an
  extra Poisson-like rate): solved exposures reproduce the requested S/N
  even for bright, scintillation-limited stars.

## Physics-audit fixes and additions (v10.2)

A full physics audit of v10.1 (`PHYSICS_AUDIT_v10.1.md`) found two
demonstrable errors, one suspect dataset and a list of missing standard
terms. All of them are implemented and regression-tested in v10.2:

* **Slitless sky (severe fix)**: dispersing a uniform source leaves the
  detector uniformly illuminated, so every slitless pixel receives sky from
  the **entire grating bandpass**. The sky is now the full-band per-pixel
  integral (~118× larger than the old per-resel booking for a typical Star
  Analyser setup); slitless S/N under bright sky is accordingly — and
  correctly — much lower.
* **R-invariant line fluxes (severe fix)**: per-resel source counts are now
  exact **bin integrals** of the calibrated template (cumulative trapezoid
  on a union grid) instead of centre-point samples; total counts of a
  narrow emission line no longer depend on the chosen R (previously up to
  8× spread). σ(EW) inherits the fix.
* **Daylight sky re-anchored**: the supplied daylight table read as
  mag/arcsec² was ~4.5 mag too faint; its shape is kept and its brightest
  entry re-anchored to the physical V = 4.0 mag/arcsec². Twilight blending
  inherits the correction and now runs on the **geometric** Sun altitude
  (the standard convention); sunrise/set use the −50′ upper-limb rule.
* **Sky-subtraction noise**: a new *sky annulus pixels* field adds the
  Merline & Howell `(1 + n_pix/n_sky)` factor to every background variance
  term in both engines and in the exposure solver.
* **Moonlight distance**: the Krisciunas & Schaefer illuminance is scaled
  by `(384400/d_moon)²` from the topocentric track distance (±15% over the
  month).
* **Zodiacal elongation**: the zodiacal component now brightens toward the
  Sun (Leinert-style power law, ~4× at 60° elongation).
* **Airglow slant extinction**: the van Rhijn enhancement is attenuated by
  Kasten–Young slant-path tropospheric extinction, matching the observed
  horizon behaviour.
* **Telluric bands** (opt-in checkbox): parametric O₂/H₂O absorption bands
  (B and A bands, water bands to 9400 Å) with curve-of-growth airmass
  scaling on top of the smooth extinction curve.
* **Guiding blur**: an entered guiding rms adds image-motion blur in
  quadrature to the seeing in every PSF coupling.
* **Template transforms**: optional radial velocity (λ(1+v/c)) and
  interstellar reddening E(B−V) (CCM89, R_V = 3.1) applied to the template
  before calibration — the entered magnitude stays the observed one.
* **Consistent QE policy**: spectroscopy now zero-fills QE and observing
  filter outside their coverage exactly like photometry (no more crashes on
  red-end ranges photometry accepts).
* **Alias-free band integrals**: photometric integration runs on the union
  of the filter grid, the template grid and a dense baseline, so coarse
  filter tables no longer skip narrow template lines.
* **R sanity**: an optional clamp limits the entered slit R to what the
  entered spectrograph geometry can deliver; an underfilled slit
  (seeing ≪ slit) is flagged as seeing-limited (entered R conservative).

## Local horizon profile from the Copernicus DEM

The OBSERVATORY panel can compute the real terrain horizon of your site.
Enter a search radius (default 10, range 1–100 km; a `miles` unit selector
converts for you) and press **Generate horizon profile**: the program
downloads the Copernicus GLO-30 digital elevation model tiles around your
coordinates (public AWS bucket, no credentials; the tiles are cached in
`dem_cache/` so later runs are offline), resamples them onto a local
metric grid, and traces the apparent horizon elevation in 360 azimuth
steps, including the Earth-curvature drop and the standard 34′ horizontal
refraction. The result is saved automatically as CSV in `data/horizons/`
(`horizon_lat…_lon…_r…km.csv`: a commented header with the site metadata,
then `azimuth_deg,horizon_elevation_deg` rows; azimuth N=0/E=90). The
computation runs in the background — the window stays responsive — and
**Display horizon** plots the saved profile for the current coordinates.
This feature needs two extra libraries, installed on demand:
`pip install rasterio pyproj` (everything else in SPETC works without
them).

## Adding template spectra: CALSPEC FITS and STScI atlases

Template spectra can be two-column ASCII **or CALSPEC/STScI-style FITS
binary tables** (a WAVELENGTH/FLUX table; TUNIT in Angstrom, nm or micron
is honoured). This covers the whole STScI reference-atlas family: the
CALSPEC standards, the solar-system surface-brightness atlas
(Jupiter/Saturn/Uranus/Neptune, solar spectrum), the galactic
emission-line atlas (Orion, NGC 7009), the transient templates (SN
Ia/Iax/91bg/Ib/II, kilonova) and the CLOUDY planetary-nebula grids.

Import with:

```bash
python3 add_template.py path/to/spectrum_or_directory [name] [spt]
```

The tool copies the file under `data/imported/`, computes the **synthetic
Vega V magnitude of the file** (this is the catalogue `mv0` — the
magnitude the file represents) and B−V against the shipped Bessell
profiles, reports the response-weighted V coverage, and appends the
`interpola.db.csv` row. Because the ETC always rescales templates to your
entered magnitude, files with arbitrary or surface-brightness flux units
(the transients, the planet atlas) are fully usable: only the spectral
shape matters, and the computed `mv0` keeps the normalization
self-consistent.

Caveats the tool enforces or flags: a template must at least reach the V
band (`mv0` cannot be derived otherwise); when it covers less than 99% of
the V response (e.g. the planet atlas, which starts at 0.53 µm) the `mv0`
is flagged approximate and you should use **V as the reference filter**
for that template, so the exact same-response path is used and no
truncated synthetic integral enters the calibration. For the planet
surface-brightness spectra pair the template with the **extended-source
mode** (enter the integrated magnitude and the disc area). Line-dominated
templates (PN grids at R=6000, the emission-line atlas) are valid up to
the resolution they were computed at — requesting much higher R dilutes
the lines.

## Outputs and where to find them

The Results window has three tabs; everything below is also in the CSV
exports (**Save result CSV** carries every engine field, **Save time-series
CSV** the per-time table).

**Selected-time result** (per magnitude row in photometry; per wavelength in
spectroscopy) shows every quantity the engine produced:

| Quantity | Meaning |
|---|---|
| `Source e-/s`, `Sky e-/s` | detected rates in the aperture / per resolution element |
| `S/N` | full noise budget: photon + sky + dark + read + **scintillation** (Young law) + **ADC quantization** (gain/√12) |
| `σ(EW) [mÅ]` | **spectral-line criterion** (Cayrel 1988): 1.5·√(FWHM·δx)/(S/N per pixel). A line is measurable when its expected equivalent width exceeds ≈3 σ(EW) |
| `Scint [e-]`, `ADC noise [e-]` | the two non-Poisson noise terms, in electrons, so their weight is visible |
| `Peak [e-]`, `Peak ADU`, `Sat`, `Max t [s]` | brightest-pixel prediction, saturation cause (`FULL_WELL`/`ADC`/`BOTH`/`NONE`) and the longest unsaturated single frame |
| `Std obs mag` / `Instr mag` | synthetic standard magnitude in the observing filter vs. the instrument-weighted response magnitude |
| `Sky mag/"²` | the sky surface brightness actually used at the selected time |

**Time series** adds per time sample: UTC/local time, MJD, elevation,
azimuth, **parallactic angle**, Pickering airmass, S/N or required exposure,
sky brightness, band-effective extinction (total and per airmass),
absorption-adjusted apparent magnitude, saturation flag, peak electrons and
maximum unsaturated exposure.

**Assumptions / outputs** states, in plain language, every model the run
used (sky mode and its SQM/table inputs, PSF model, gain-table setting,
slit-spectrograph geometry, noise terms, saturation policy) followed by a
**SELECTED-TIME OUTPUTS** block with the S/N, the scintillation and ADC
noise in electrons (and the scintillation as a percentage of the source),
the peak-pixel/saturation numbers, the predicted magnitudes, the σ(EW)
line criterion, the **stack plan** (N × sub-exposure, limiting factor,
read-noise penalty) and the **differential-photometry precision** in mmag
against the entered comparison star.

The main-window labels mirror the two planning results live: the stack
plan under CALCULATION and the differential precision / σ(EW) under
PHOTOMETRY.

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
  (gain/sqrt(12)) noise. Sources can be **point**, **extended** or
  **defocus**: extended sources (galaxy/nebula/planet) keep the entered
  integrated magnitude spread uniformly over a stated angular area, with
  area-fraction aperture capture and a surface-brightness peak pixel (valid
  when the source is much larger than the seeing disc); **defocus** models a
  defocused telescope, computing the donut PSF (geometric annulus of outer
  radius `delta*D/2F` with a central hole set by the obstruction — a filled
  disc when the obstruction is 0 — **convolved with the atmospheric seeing/
  guiding PSF**, so the profile has realistic rounded rims and wings, not
  vertical edges) and its encircled-energy curve, from which you pick the
  photometry aperture radius in a dedicated window (profile and
  encircled-energy figures and data saved to the session).
  Spectroscopy supports `slit` and
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

## RNAAS article

`docs/rnaas/` contains a Research Notes of the AAS manuscript presenting the
tool: `spetc_rnaas_aastex.tex` is the submission-ready AASTeX 6.3.1 source
(compile where `aastex631.cls` is available, e.g. the AAS submission system),
and `main.tex`/`docs/SPETC_rnaas_preprint.pdf` is an identical-text preprint
rendering that compiles with the bundled `compile.sh`. Body text is ~590
words, one figure — within the RNAAS limits (1000 words, one figure).
