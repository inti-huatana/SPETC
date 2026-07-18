# SPETC v10.0 — Repository Audit

Static review of the Python sources, data wiring and documentation. No code was
changed. Findings are grouped as **Bugs / errors**, **Dead code that can be
removed safely**, **Smaller issues**, and **Suggested additions**.

Scope reviewed: all 16 top-level modules (`4149` LOC), the two test entry points,
`requirements.txt`, `README.md`, and the JSON/data wiring. The two numeric smoke
tests were executed (see notes).

---

## 1. Bugs and errors

### 1.1 The documented test command fails — `ModuleNotFoundError`
`README.md` (§ *Included numerical smoke test*) tells the user to run:

```bash
python3 tests/test_spectral_pipeline.py
```

Running it exactly as documented fails:

```
ModuleNotFoundError: No module named 'detector'
```

Cause: when a script is run by path, Python puts the **script's own directory**
(`tests/`) on `sys.path`, not the repository root, so `from detector import …`
cannot be resolved. There is no `conftest.py`, no `tests/__init__.py`, and the
test does not insert the parent directory onto `sys.path`. It only works if the
user happens to set `PYTHONPATH=.`:

```bash
PYTHONPATH=. python3 tests/test_spectral_pipeline.py   # PASS
```

`self_check.py` works as documented only because it lives in the repository root.
Fix options: add a two-line `sys.path` insert at the top of the test, add a
`conftest.py`/`pyproject.toml` for `pytest`, or move the test to the root. As
shipped, the second half of the documented verification procedure is broken.

### 1.2 `requirements.txt` is not runnable out of the box in a clean env
The declared minimums (`numpy>=1.20`, `pandas>=1.3`, `astropy>=5.0`, …) are fine,
but nothing pins upper bounds and there is no lock file. More importantly the
repo assumes a system with these already present; a fresh `python3` has none.
Not a code defect, but combined with 1.1 it means "clone → follow README" does
not reproduce a working install cleanly. Consider a pinned/tested set or a note.

### 1.3 `solar_cycle_phase()` can raise `UnboundLocalError` on malformed input
`sky_brightness.py::solar_cycle_phase` assigns `fase_raw` inside a `for` loop
with no `else`/default. For every realistic year the contiguous-interval logic
assigns it, so this does **not** trigger in normal use (years > 2019.96 take the
`elif` branch). It is nonetheless fragile: any future edit to the interval logic,
or a NaN year, leaves `fase_raw` undefined and the function crashes rather than
degrading. Same pattern is duplicated in `get_sky_components`. Low severity but
worth a defensive default.

### 1.4 `Detector.counts_to_adu` truncates and can crash on non-finite peaks
`detector.py::counts_to_adu` does `int(adu_clipped)` / `.astype(int)`. For a
non-finite peak (e.g. an upstream NaN) `int(nan)` raises `ValueError`, and for
large counts the truncation-toward-zero loses a sub-ADU. Peaks are finite in the
current flows, so this is latent rather than active, but the conversion is not
guarded.

---

## 2. Dead code that can be removed safely

These are **not referenced anywhere** in the project (verified by grep across all
`*.py`). Some are explicitly labelled "compatibility wrappers"; those are called
out so you can decide whether to keep them as public API or delete them.

### 2.1 `atmospheric.py` — entire module is dead (≈200 LOC)
Nothing imports `atmospheric`. Every function
(`seeing_young`, `plate_scale_arcsec_per_pixel`, `psf_gaussian`,
`slit_loss_rectangular`, `airmass_extinction`, `zenith_distance_from_altitude`,
`altitude_from_zenith_distance`) is unused — the plate-scale and slit-throughput
logic now lives inline in `photometry.py` / `spectroscopy.py` and in
`etc_physics.slit_throughput`. The module even carries long docstrings narrating
"bugs fixed in this version" for code that is never called. **Safe to delete the
whole file.** Its constants `H_PLANCK` / `C_LIGHT` are also unused (the real code
uses `astropy.constants`).

### 2.2 `solvers.py` — 3 of 4 functions are dead
Only `exposure_time_for_snr` is used (by the GUI). Unused:
- `reverse_texp_for_snr`
- `compute_snr_vs_texp`
- `compute_texp_vs_snr`

The 30-line module docstring documents a bug fix in `reverse_texp_for_snr`, a
function nothing calls. Safe to remove the three unused functions (and trim the
docstring), keeping `exposure_time_for_snr`.

### 2.3 `ephemeris.py` — most of the module is unused
The GUI imports only `compute_target_track`. Unused elsewhere in the project:
`julian_date`, `julian_centuries`, `sun_position`, `moon_position`,
`airmass_simple`, `jd_to_hm_string`, `parse_ra`, `parse_dec`,
`degrees_to_sexagesimal_ra`, `degrees_to_sexagesimal_dec`,
`galactic_to_equatorial`, `equatorial_to_galactic`, `ecliptic_to_equatorial`,
`equatorial_to_ecliptic`, `altitude_azimuth`, `find_altitude_at`, `find_transit`,
`find_altitude_crossing`, `find_rise_set`.

Note: `julian_date/…/airmass_simple` are imported by `sky_brightness.py` but that
import line itself is dead (see 2.4), so those functions have **zero live
callers**. The `find_*` helpers are annotated "Compatibility wrappers retained
for callers outside the GUI" — keep only if you intend a public API; otherwise
they are removable.

### 2.4 `sky_brightness.py` — dead import line and dead function
- Line 12 `from ephemeris import julian_date, julian_centuries, sun_position,
  moon_position, airmass_simple` — **none of these names are used** in the file.
  Remove the import.
- `get_sky_components(...)` is never called. Removable.
- Inside `sky_brightness_total`/`get_sky_components`: local `k` (illumination
  fraction) is computed and never used; parameter `moon_zenith_dist_deg` is
  accepted and never used; the `include_sun_twilight` branch is never exercised
  (the GUI always passes `include_sun_twilight=False`).

### 2.5 `etc_physics.py` — dead wrapper chain
- `ab_f_lambda` — unused.
- `interpolate_quantity` — unused.
- `normalise_template_magnitude` — only called by `normalise_template_ab`.
- `normalise_template_ab` — not called anywhere.

The two `normalise_*` wrappers form a dead chain (one calls the other, nobody
calls either). All four are safe to remove; live code uses
`calibrated_template_magnitude` directly.

### 2.6 `filter_catalog.py` — `magnitude_to_ab` unused
`magnitude_to_ab(magnitude, profile)` is never called. Removable (or wire it in;
see 4.x). `filters_directory` and `generic_zenith_atmosphere_curve` **are** used.

### 2.7 `detector.py` — dead methods
`Detector.check_saturation`, `Detector.read_noise_electrons` (and its internal
`read_noise_table`), and `Detector.info` are never called. Live code uses
`counts_to_adu` and `saturation_flag` only. `read_noise_electrons` is even
docstring-flagged as "a placeholder". Removable.

### 2.8 `star_catalog.py` — partially unused
`search_stars` is used by the GUI. Unused: `find_star_by_name`,
`spectral_type_reference`, and the `SPECTRAL_TYPE_MS` / `SPECTRAL_TYPE_GIANT`
lookup tables (with their `_TS_*` source lists). The module docstring advertises
"search by characteristics", but the GUI only calls `search_stars(name_query=…)`.
Either wire the spectral-type search into the UI or drop the tables. Also
`H_PLANCK`/`C_LIGHT` here are unused.

---

## 3. Smaller issues / code quality

- **Duplicated constants across two sky models.** `SOLAR_MINIMA`,
  `SOLAR_CYCLE(_YR)`, the U–K sky magnitudes and extinction table are defined
  independently in both `sky_background.py` and `sky_brightness.py`. Two copies
  can drift. Consider a single shared constants module.
- **Two overlapping sky sub-systems.** `sky_background.sky_magnitude_vega`
  (ING normalization) and `sky_brightness.sky_brightness_total` (Benn & Ellison
  + Krisciunas–Schaefer colour) are both live and are stitched together in
  `etc_gui._sky_models_for_track`. This is intentional but under-documented in
  code; a short module-level note on who owns normalization vs. colour would help
  future maintenance.
- **Duplicated site defaults.** `observatories.json` (7 sites, `mag_sky`) and
  `etc_sites.json` (richer schema, `sky_ab_mag_arcsec2`, timezone) both define
  Asiago/Paranal/La Silla. `_load_site_records` merges with `etc_sites.json`
  winning. Works, but two sources of truth for the same observatories invite
  inconsistency (e.g. Asiago sky mag is 21.20 in one, 21.2 in the other).
- **No `.gitignore`.** `__pycache__/` exists in the working tree (not currently
  tracked). A one-line `.gitignore` (`__pycache__/`, `*.pyc`) prevents accidental
  commits.
- **No `LICENSE`.** README gives author/years (2007–2026) but there is no license
  file; the redistribution terms of the bundled SVO filter profiles and BPGS
  spectra are also unstated.
- **`self_check.py` tolerance is very wide.** It asserts photometric S/N in
  `(8e3, 3e4)` and spectral S/N/resel in `(150, 800)` — broad enough to pass even
  with sizeable regressions. It caught nothing here (both pass: phot S/N≈17192,
  spec max≈382), but as a guardrail it is loose.
- **Mixed docstring style.** Newer modules (`etc_physics`, `spectral_utils`,
  `filter_catalog`) are terse and current; older ones (`solvers`, `atmospheric`,
  `detector`, `sky_brightness`) carry long "bug fixed in this version" narratives
  that describe history rather than behaviour — noise once the dead code around
  them is removed.

---

## 4. Suggested additions (useful, not currently present)

**Testing / CI**
- A real `pytest` layout (`conftest.py` or `pyproject.toml`) so both test files
  run without `PYTHONPATH` gymnastics, plus a GitHub Actions workflow that runs
  `self_check.py` and the pipeline tests on push. This directly prevents 1.1
  recurring.
- Unit tests for the parts with zero coverage today: `filter_catalog`
  VOTable parsing (MagSys/ZeroPoint/DetectorType validation paths),
  `star_catalog.parse_interpola_db` fixed-width vs. CSV fallback,
  `ephemeris.compute_target_track` geometry, and `solvers.exposure_time_for_snr`
  round-tripping against `compute_photometry_single`.
- A tightened numeric regression: assert S/N against a stored golden value with a
  few-percent tolerance rather than an order-of-magnitude window.

**Packaging / project hygiene**
- `.gitignore`, `LICENSE`, and a `pyproject.toml` (or `setup.cfg`) declaring the
  package and console entry point (`spetc = etc_gui:main`).
- A `data/README` documenting the required data layout in one place
  (`interpola.db.csv`, `filters.list`, `qe.dat`, BPGS, optional atmosphere FITS),
  since that knowledge is currently spread through the main README prose.

**Functionality**
- Wire the already-written but unused capabilities into the UI instead of
  deleting them, if they are wanted: spectral-type search
  (`star_catalog.spectral_type_reference`), and `filter_catalog.magnitude_to_ab`
  for a one-click Vega↔AB readout.
- A headless/CLI mode (`--config foo.json --batch`) so the ETC can be scripted;
  today all computation is reachable only through the Tk GUI, which also blocks
  automated testing of the end-to-end path.
- Persist/export results: a "Save results as CSV/JSON" for the single-time
  Results window (the time-series path already emits CSV).
- The README notes `NIGHT_SKY_EM` is deliberately unused pending a defensible
  flux unit; a documented spectral sky model would remove the current broad-band
  colour approximation and retire one of the two duplicated sky sub-systems.
- Input validation feedback in the GUI for out-of-coverage requests: the physics
  layer raises clear `ValueError`s (good), but surfacing them as inline field
  errors rather than dialogs would improve usability.

---

## 5. Quick reference — safe deletions

| File | Remove | Keep |
|------|--------|------|
| `atmospheric.py` | entire module | — |
| `solvers.py` | `reverse_texp_for_snr`, `compute_snr_vs_texp`, `compute_texp_vs_snr` | `exposure_time_for_snr` |
| `ephemeris.py` | unused `julian_*`, `*_position`, `airmass_simple`, `parse_*`, `*_sexagesimal_*`, coord converters, `find_*` (unless kept as public API) | `compute_target_track` |
| `sky_brightness.py` | dead `ephemeris` import line, `get_sky_components`, unused `k`/`moon_zenith_dist_deg` | `sky_brightness_total`, band tables |
| `etc_physics.py` | `ab_f_lambda`, `interpolate_quantity`, `normalise_template_magnitude`, `normalise_template_ab` | `calibrated_template_magnitude` & core |
| `filter_catalog.py` | `magnitude_to_ab` (or wire in) | rest |
| `detector.py` | `check_saturation`, `read_noise_electrons`, `info` | `counts_to_adu`, `saturation_flag` |
| `star_catalog.py` | `find_star_by_name`, spectral-type tables (or wire in) | `parse_interpola_db`, `search_stars`, `load_star_spectrum` |

*Verify with a project-wide grep before deleting anything intended as an external
API; the `find_*` and spectral-type helpers are explicitly annotated as retained
compatibility surface.*
