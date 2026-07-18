# Scientific review of the usability update (commit `2467bf6`)

Review of "abbellimenti grafici e risoluzioni problemi su spettri" from the
astronomical standpoint, followed by targeted suggestions for the intended
audience — amateur spectroscopists and photometrists, and professionals
without an institutional ETC — and a comparison with existing tools.
Analysis only; no code was changed.

---

## 1. What the update does (as read from the diff)

Usability: coloured status bar and Run/Exit buttons; the Run button is now
**disabled until a pre-flight check passes** (template × reference filter
overlap > 0 and detected signal > 0), with a clear "Cannot run ETC: …"
message; friendlier error dialogs (no traceback for input errors); the
template browser now searches by **spectral type** and displays B−V
(wiring up the previously dead `search_stars(spt_prefix=…)`); scrollbar on
the template list; reference-filter changes refresh the response preview.

Physics policy change (the important one): the **hard wavelength-coverage
rule was relaxed**. Template, QE and atmospheric curves are now
zero-filled outside their coverage (`interpolate_zero_filled`) instead of
raising, in photometry, spectroscopy and `atmospheric_transmission`.

The pre-flight gate is a genuinely good design: it moves the failure from
mid-calculation to before the button, which is exactly what an amateur
user needs. The findings below concern where the zero-fill relaxation is
scientifically safe and where it silently biases results.

---

## 2. Findings, ranked by scientific severity

### 2.1 Silent flux-calibration bias when the template only partly covers the reference band — HIGH
With `require_coverage` disabled, a template that covers a *fraction* of
the **reference (or visual) magnitude band** still calibrates: the
synthetic magnitude inside `calibrated_template_magnitude` is computed on
the truncated SED, comes out too faint, and the whole spectrum is scaled
**up** to compensate. Verified numerically on this code: a flat template
covering half of the reference band yields predicted counts biased high by
a factor **1.92 (0.71 mag), with no warning**. In target-S/N mode this
halves the recommended exposure — the observer comes home with S/N
√2 lower than requested.

The GUI pre-flight only requires the overlap to be *non-zero*; 10%
coverage passes. Recommendation (one of):

* keep zero-fill on the **observing** side (counts) but restore the strict
  coverage requirement for the **reference and visual calibration bands**;
  or
* renormalize the synthetic magnitude by the covered response fraction
  and require it ≥ ~95%; or
* surface the coverage fraction in the pre-flight message
  ("reference band 62% covered — calibration biased by up to 0.5 mag")
  and refuse below a threshold.

The calibration integral is the one place where truncation is a *bias*,
not a conservative loss.

### 2.2 Zero-filled atmosphere = opaque atmosphere, and a source/sky asymmetry — MEDIUM–HIGH
`atmospheric_transmission` now returns **T = 0 outside the loaded curve's
coverage**: the model atmosphere becomes perfectly opaque there. Verified:
a 4000–7000 Å curve on a 3500–8000 Å grid gives T = [0, 0.8, …, 0.8, 0, 0].
Two consequences:

* With the Paranal FITS loaded, any band extending past the file's range
  silently loses that flux — e.g. red/NIR photometry can lose a large
  fraction of its counts with no message.
* The sky is (correctly) *not* re-extinguished (`sky_at_telescope`), so at
  the zero-filled wavelengths the source contributes nothing while the sky
  keeps accumulating — S/N is doubly underestimated and required exposures
  inflated.

An atmosphere is the one curve for which zero is the *least* physical
extrapolation. Better: extend with the edge value (nearest-neighbour), or
fall back to the built-in broad-band table outside the file's coverage,
and note it in the output.

### 2.3 QE zero-fill is good physics — keep it
Unlike the two cases above, zero response outside a measured QE curve is
usually *true* (detector cutoffs) and always conservative. This part of
the relaxation is scientifically sound as the new default.

### 2.4 The regression suite no longer passes — MEDIUM
`test_spectroscopy_rejects_template_outside_requested_range` fails,
because the contract it protects ("out-of-coverage must not silently
return zero counts") was deliberately changed. The test should be updated
to pin the *new* contract (zero-filled bins outside coverage, pre-flight
rejection at zero overlap) rather than left failing — a failing suite
masks future real regressions. `self_check.py` still passes.

### 2.5 Latent crash in the two converter panels — LOW (but a real bug)
The status-label variables `time_convert_status_var` and `dms_status_var`
are now commented out at creation, but both **error paths still call
`.set()` on them** (`etc_gui.py:600`, `etc_gui.py:651`). Entering an
invalid time or coordinate string in the converter panels raises
`AttributeError` instead of showing the message. Either restore the two
`tk.StringVar` creations or drop the `.set()` calls.

### 2.6 `np.trapezoid` requires NumPy ≥ 2.0 — LOW
The new pre-flight uses `np.trapezoid` (`etc_gui.py:1168,1173`), which does
not exist before NumPy 2.0, while `requirements.txt` allows `numpy>=1.20`.
On an older NumPy the preview crashes and the Run button stays disabled.
Use the existing `etc_physics.trapz_quantity` fallback pattern or bump the
requirement.

### 2.7 Name search was replaced, not extended — usability note
The search box now matches spectral type only: typing "Vega" or "Sun"
finds nothing. Since `search_stars` already accepts both criteria,
matching `name_query` **or** `spt_prefix` on the same string would keep
both behaviours (try SpT prefix; fall back to substring name match).

### 2.8 The Assumptions/visibility tab is hidden — transparency note
The tab stating the run's physical assumptions (sky model and its value at
the selected time, PSF model, parallactic angle, noise terms, saturation
policy) is commented out. The parallactic angle and sky brightness remain
in the time-series CSV, but the one-glance provenance is gone — that panel
is what lets a user (or a referee) know *which* sky the number assumed.
Suggest reinstating it, possibly collapsed by default.

---

## 3. What to add for this audience

Ordered by usefulness-per-effort for amateurs and un-equipped
professionals. (Observation-planning features are explicitly out of scope
and none are suggested.)

1. **Instrument preset library.** The profile mechanism already exists —
   ship it populated: Star Analyser 100/200, Shelyak Alpy 600, LISA,
   LHIRES III (with its gratings), UVEX, DADOS, and common cameras
   (ASI533/1600/2600/6200, QHY equivalents, typical DSLRs) with pixel
   size, gain-indexed read noise and full well, and measured QE curves.
   For amateurs this replaces the hardest data-entry step; for pros it
   demonstrates the workflow to copy for their own instrument.
2. **CMOS gain model.** Amateur CMOS cameras change read noise, full well
   and gain together with the gain setting. A per-camera table
   (gain setting → e⁻/ADU, RN, FWC) with a "gain" selector would make the
   saturation and noise predictions match what the camera actually does —
   today a single triple must be re-typed per setting.
3. **One-shot-colour (Bayer) support.** Most amateurs image with OSC
   cameras: per-channel (R/G/B) effective QE curves and the ×0.25/×0.5
   pixel-fraction sampling per channel. Without it, OSC users
   systematically overestimate their counts by 2–4×.
4. **Sky brightness from SQM / Bortle.** Amateurs know their sky as an SQM
   reading (mag/arcsec² in V) or a Bortle class, not as a per-band table.
   An input mode "SQM = 20.4" (or Bortle preset → SQM) feeding the
   `fixed_ab`/colour machinery would map directly onto how this audience
   thinks.
5. **Stack planner.** Given a target total S/N, report the optimal
   sub-exposure (saturation- or sky-noise-limited), the number of frames,
   and the read-noise penalty of the stack:
   S/N² summing with N×RN² is a two-line extension of the existing solver
   and is *the* question amateurs actually ask ("how many 120 s subs?").
6. **Differential-photometry precision mode.** For variable-star and
   exoplanet photometrists (AAVSO/Exoclock style): S/N of target and a
   comparison star of magnitude m_c combined into an error budget in
   millimagnitudes per exposure, scintillation included (it correlates
   only partially between stars). Output "3.2 mmag per 60 s frame" instead
   of raw S/N — this is the unit that community works in.
7. **Emission-line source mode for spectroscopy.** The template library is
   stellar-continuum only. Nebulae, comets, novae and Be stars are
   line-dominated: accept a line list (λ, integrated flux in
   erg s⁻¹ cm⁻² or relative to Hβ) on an optional continuum, and report
   S/N per line — "can my Alpy detect [OIII] 5007 of NGC 6543 in 10
   minutes" is a core amateur-spectroscopy question the current model
   cannot answer.
8. **Slit-spectrograph geometry helper.** Amateurs know grating lines/mm,
   collimator/camera focal lengths and slit width — not R. A small
   calculator (the grating equation, anamorphism, slit-limited vs
   seeing-limited R) feeding the existing R field would mirror what the
   Star Analyser path already does for slitless.
9. **Equivalent-width / line-depth detectability.** Given R, S/N and line
   FWHM, report σ(EW) (Cayrel relation). Turns the spectroscopic S/N into
   a science answer ("EW error 15 mÅ — Hα variability of this Be star is
   detectable").
10. **Bundled amateur filter set.** Ship SVO profiles for
    Johnson-Cousins BVRI (Astrodon/Baader), Sloan g'r'i', narrowband
    Hα/OIII/SII (3.5/7/12 nm), UV/IR-cut and a generic RGB set, so the
    filter selector is useful out of the box without visiting SVO.
11. **Headless/batch mode and a one-page PDF/HTML run summary** — the
    engines are already GUI-free; a small CLI plus a printable summary
    (setup, sky, S/N curve, saturation) would serve pros scripting many
    cases and amateurs documenting sessions.
12. **A validation page.** One documented comparison of predicted vs
    measured counts for a standard star on real amateur hardware
    (e.g. an SA100 + ASI camera) would do more for trust than any model
    refinement, and would let the slitless mode drop its "experimental"
    label.

---

## 4. Comparison with existing tools

No existing tool covers SPETC's combination (instrument-agnostic,
photometry *and* spectroscopy, physical sky/Moon model, saturation and
time-dependence). The landscape:

| Tool | Audience | Scope | Compared with SPETC |
|---|---|---|---|
| **SimSpec** (C. Buil, Excel) | amateur spectroscopists | slit-spectrograph S/N for Shelyak-style setups | The de-facto amateur standard. Rich instrument presets; but spreadsheet-bound, single-epoch, simple sky, no Moon/position dependence, no saturation/time series, no photometry. SPETC exceeds it physically; adopting its *preset* idea (Sect. 3.1) would close the usability gap. |
| **Shelyak documentation calculators** | amateur spectroscopists | per-instrument specs | Static tables, not an ETC. |
| **CCD/SNR calculators** (CCDCalc, astronomy.tools, SharpCap sky-limited exposure, NINA exposure aid) | amateur imagers | sampling/FOV, sky-limited sub-exposure | Imaging-oriented; none propagates a spectral template through filter × QE × atmosphere; no spectroscopy; no Vega/AB rigor. SPETC's photometric mode is strictly more capable; their Bortle/SQM-style inputs (Sect. 3.4) are worth borrowing. |
| **Public professional ETCs** (ESO, Gemini ITC, LCO, NOT/MMT) | professionals | their own instruments only | Full physics but locked to facility instruments — useless for a private 40-cm or a spectrograph they don't host. SPETC's niche is exactly the "no institutional ETC" case. |
| **ETC-42** (CNES/LAM, Java) | professionals | generic configurable ETC | The closest professional analogue: instrument-agnostic and configurable. Aging Java stack, steep configuration, no amateur hardware notion, no Moon/position sky model of SPETC's kind, no slitless-grating mode. SPETC is the lighter, amateur-reachable equivalent. |
| **synphot / pysynphot / ScopeSim** | professionals who code | synthetic-photometry libraries / instrument simulators | Libraries, not tools: everything SPETC does out of the box must be programmed. Useful as validation cross-checks for SPETC's synthetic magnitudes. |

Positioning: with the presets, OSC support and SQM input of Sect. 3, SPETC
would be the only tool an amateur spectroscopist *and* photometrist needs
between "what's my hardware" and "was the exposure long enough" — sitting
above SimSpec in physics and below facility ETCs in ceremony, with the
generic-instrument flexibility that even most professionals' public ETCs
lack.

---

## 5. Summary table

| # | Finding | Severity | Action suggested |
|---|---|---|---|
| 2.1 | Partial reference-band coverage silently biases calibration (demonstrated ×1.9 / 0.7 mag) | High | strict coverage (or covered-fraction renormalization) for calibration bands only |
| 2.2 | Zero-filled atmosphere = opaque; source/sky asymmetry | Med–High | edge-extend or broad-band fallback outside curve coverage |
| 2.4 | Regression suite failing after contract change | Med | update the test to pin the new zero-fill contract |
| 2.5 | AttributeError in converter error paths | Low | restore the two StringVars or drop the `.set()` calls |
| 2.6 | `np.trapezoid` needs NumPy ≥ 2 | Low | fallback or requirement bump |
| 2.7 | Name search no longer works | Low | match name OR spectral type |
| 2.8 | Assumptions tab hidden | Low | reinstate (optional/collapsed) |
| 2.3 | QE zero-fill | — | correct as-is; keep |
