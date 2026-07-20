# SPETC v10.1 — Full physics audit of the branch

Audit of every physics path at commit `13e7c55`, explicitly hunting the three
failure classes: (A) demonstrated wrong physics, (B) missing standard terms,
(C) approximations that are acceptable but must stay visible, (D) shipped
data whose normalization is suspect. Where a claim could be tested
numerically, it was; the numbers quoted below come from runs against this
branch, not from reading alone. My own additions from this session are
audited with the same severity as the inherited code.

---

## A. Demonstrated wrong physics (verified numerically on this branch)

### A1. Slitless sky background is underestimated by ~two orders of magnitude — SEVERE
In slitless spectroscopy the sky is an extended uniform source: dispersing a
spatially uniform source leaves the detector illumination uniform, so **every
pixel receives sky light integrated over the entire grating bandpass**, not
just over the resolution element mapped to that pixel. This is the classical
reason objective-grating spectra are sky-limited. The code books the sky per
resel through the same `dlam` as the source — i.e. it treats the sky as if a
slit restricted it.

Measured on this branch (SA100-like setup, 4000–7300 Å, 28 Å resel): code
sky = 0.96 e⁻/s/resel; the physical pixel sees the full ~3300 Å of sky →
**underestimate factor ≈ 118×** for this configuration (in general
≈ band width / resel width). Every slitless S/N this ETC has produced is
optimistic in its sky term by that factor; the effect is largest exactly
where slitless work happens (bright sky, low resolution). The mode's
"experimental" flag is carrying much more weight than intended: this is not
an unvalidated detail, it is missing physics. **Fix**: slitless sky per
pixel = sky surface brightness × (pixel area on sky) × ∫ R(λ)·QE(λ)·T(λ)
dλ over the full grating bandpass (× grating efficiency), independent of the
resel; per resel multiply by the dispersion-direction pixels.

### A2. Template flux is point-sampled per resel, not integrated — narrow-line flux is not conserved — SEVERE for line sources
`compute_spectroscopy` computes each resel's source as
`F_λ(λ_center) × Δλ`: the template is *sampled* at the resel centre instead
of being *integrated* over the resel (equivalently: the template is never
convolved to the instrument resolution). For smooth continua the two agree;
for a spectral line narrower than the resel they do not, and total line
counts become resolution-dependent, which is unphysical.

Measured: a 2 Å-FWHM emission line (10⁻¹² erg s⁻¹ cm⁻²) on a 200 mm
telescope gives total detected line counts of 6.72 e⁻/s at R=200, 2.69 at
R=500, 18.0 at R=2000 and 22.5 at R=20000 — a spread of **8×** where the
value must be R-invariant (22.5 e⁻/s is the correct converged value,
matching the hand integral). Consequences: σ(EW) and line-S/N for the very
emission-line targets the template importer now supports (PN atlases,
transients, Be stars) are unreliable at low/medium R; even continuum
sources see percent-level aliasing when the template grid beats against
the resel grid. **Fix**: bin-integrate the calibrated template over each
resel (cumulative-integral differencing is O(n) and fast), optionally after
Gaussian convolution to the instrument LSF; same for the photometric band
integral (see B7).

---

## D. Shipped data with suspect normalization

### D1. The Weaver daylight-sky table is ~4–5 magnitudes too faint — SUSPECT, needs source check
`DAYLIGHT_V_MAG` spans 8.50–9.20, and `sky_magnitude_vega` uses these
directly as the daytime V surface brightness in mag/arcsec². Physical
daytime zenith sky luminance is ~3000–8000 cd/m²; against the night sky
(21.9 mag ≈ 2×10⁻⁴ cd/m²) that corresponds to **V ≈ 3.5–4.5 mag/arcsec²**,
not 8.7. If the table values are interpreted as mag/arcsec², the modeled
daytime sky is ~75× too dark, and every daytime/bright-twilight prediction
(a documented feature — the Sun is even drawn in the plots) is wildly
optimistic. The numbers plausibly are Weaver's tabulated quantity in a
different unit (his paper works in log-luminance units), mis-imported as
magnitudes somewhere in the Fortran→Python lineage. This table came with
the original supplied data, so I flag rather than silently change it:
**verify against Weaver (1947) and re-derive the unit conversion**; until
then, daytime results should be considered wrong, and the twilight blend
inherits a too-dark daylight anchor above sun altitude ≈ −10°.

---

## B. Missing standard terms (real omissions, ranked)

**B1. Sky/dark-subtraction noise.** The CCD equation assumes the background
is known perfectly. Real photometry estimates the sky from an annulus of
n_sky pixels, multiplying the background variance by (1 + n_pix/n_sky);
spectroscopy likewise from sky windows along the slit. For small sky
regions this is tens of percent of the background noise. Standard in
photometry texts (e.g. Merline & Howell 1995); absent here in both engines.

**B2. Lunar distance and phase-angle geometry.** The K&S illuminance uses
the phase angle α computed as 180° − elongation (exact only for infinite
Sun distance; fine) but ignores the Moon's distance variation — (385000/d)²
is a ±15% moonlight brightness modulation over the anomalistic month that
K&S themselves include. One multiplicative factor from data already present
in the track (`moon distance` is computed and discarded).

**B3. Zodiacal light has no solar-elongation dependence.** The
Benn & Ellison form used depends on |β| only; the real zodiacal brightness
rises steeply toward the Sun (factor ~3 from elongation 180°→60° at β=0).
For evening/morning targets at low elongation the dark sky is
underestimated. The elongation is trivially available from the track.

**B4. Airglow path extinction.** The van Rhijn factor raises the airglow
along slant paths but the extra extinction along the same path is not
applied; near the horizon the two partially cancel. Net effect: low-altitude
sky brightness somewhat overestimated (the raw van Rhijn 6× at z=90° is
never realized in nature). Standard treatment multiplies van Rhijn by
10^(−0.4·k·(X−1)) or similar.

**B5. Refraction conventions at thresholds.** Twilight boundaries (−18°,
−12°…) are defined on the *geometric* Sun altitude, but the track supplies
refracted altitudes everywhere; near the horizon this shifts twilight/day
switching by a few minutes. Similarly "Sun above horizon" uses the centre,
not the upper limb (−16′ semidiameter): displayed sunrise/sunset differ
from almanac times by ~1–2 min. Cosmetic-to-minor, but a stated convention
should exist.

**B6. Inconsistent QE coverage policy between engines.** Photometry
zero-fills the QE outside its curve (correct physics); spectroscopy still
calls `interpolate_checked`, which *raises* when the requested wavelength
range exceeds the QE table. Same input, different behaviour by mode — one
of them is unintended (the strict one contradicts the v10.1 zero-fill
contract and will crash red-end spectroscopy runs that photometry accepts).

**B7. Photometric band integral on the filter's native grid.** The
integration grid is the SVO table's own sampling; coarse filter tables
alias line-rich templates (same family as A2, milder). A common refined
grid (as `synthetic_magnitude` already builds internally) would fix it.

**B8. Telluric line absorption.** Extinction is a smooth curve everywhere;
the O₂/H₂O band structure beyond ~6500 Å is absent, and beyond the loaded
curve's range the edge value is extended flat. Fine through V/R; wrong in
detail for red/NIR spectroscopy per-resel S/N. (Known, previously
documented; repeated here because it interacts with A2: proper telluric
treatment also requires the binning fix.)

**B9. Slit-mode R is taken on faith.** The engine uses the entered R even
when the slit and seeing imply a coarser resolution; only the geometry
helper (if used) applies min(slit, seeing). An engine-side sanity clamp or
warning would prevent physically impossible R claims in slit mode.

**B10. PSF second-order items.** The Moffat slit×height product treats a
non-separable 2D profile as separable (Gaussian case exact; Moffat a small
systematic, worst around width ≈ FWHM); peak-pixel fractions assume the
star centred on a pixel (stated worst case — fine); no guiding/image-motion
blur term exists for long exposures (adds in quadrature to seeing in
reality); the slit geometry helper uses the unscaled seeing even when
seeing-wavelength scaling is enabled.

**B11. No radial-velocity/redshift shift for templates**, and no
interstellar-reddening adjustment: a template's colours are used as-is,
so a reddened target calibrated in B but observed in I inherits the
unreddened template colour. Worth at least an E(B−V) knob eventually.

---

## C. Approximations that are present and acceptable — but list them once

Uniform-disc extended sources (stated); linear-flux twilight blend
(stated); SQM≈V and Bortle→SQM table (stated); Gaussian instrumental LSF
along dispersion in the peak-pixel estimate; constant pixels-per-resel
across the band; Young scintillation coefficient 0.09 without site
calibration (Osborn correction absorbable); scintillation treated as
uncorrelated between target and comparison; parallactic angle static
within an exposure; sky colour model from nine broad bands (the
`spectral_sky_f_lambda` hook exists for a real spectral sky); no fringing,
cosmic rays, nonlinearity, persistence, or dark-current temperature model
(none claimed); horizon profile computed but not yet masking visibility
(stated in docs).

---

## Verified correct (checked, do not "fix")

Extinction applied once as T_zenith^X with the Paranal mag/airmass
conversion; Krisciunas & Schaefer f(ρ) and illuminance; van Rhijn factor;
position-dependent zodiacal/starlight via astropy frame conversions
(ecliptic symmetry regression-tested); Pickering airmass on refracted
altitude (X(90°)=1, X(5°)=10.3); parallactic angle (zero at meridian,
sign east/west correct); Filippenko refractive index and differential
refraction (1.02″ for 4000→5500 Å at z=45°); Kolmogorov seeing scaling
(exact at V/zenith, 2^0.6 at X=2, blue>red — and correctly *changing the
blue/red slit-loss ratio end-to-end*, 0.360→0.306 in the test case); SVO
photon/energy synthetic-magnitude convention (photon vs energy differ for
coloured SEDs, regression-tested); the mv0/CALSPEC calibration chain and
the same-response zero-point shortcut; the CCD S/N equation and its
closed-form inversion including the time-linear scintillation variance
(solver reproduces target S/N to <2% for scintillation-limited cases;
stack-planner scintillation correctly framing-independent); OSC fill
fractions (rates and pixels scale, peak does not); collecting area
π(D²−d²)/4; hour angle/LST arithmetic; azimuth N=0/E=90 consistent across
engines, plots and horizon CSV; plot azimuth wrap masking and
below-horizon masking; time-series apparent-magnitude column
(m + band-effective extinction).

Small robustness notes (not physics): `psf_encircled_energy` and `snr()`
guards are scalar-only (fine as called today, brittle to future
vectorization); `covered_response_fraction` checks the template envelope
only, so an internal gap in a template would evade the calibration check.

---

## Priority

| # | Finding | Severity | Effort |
|---|---------|----------|--------|
| A1 | Slitless sky misses the full-band term (~118× low) | Severe | moderate |
| A2 | Per-resel point-sampling: line flux not R-invariant (8× spread demonstrated) | Severe for line sources | moderate |
| D1 | Daylight table ~4–5 mag too faint if read as mag/arcsec² | High (daytime only) | verify + convert |
| B1 | Sky-subtraction noise (1+n_pix/n_sky) | Medium | small |
| B6 | QE policy mismatch photometry/spectroscopy | Medium (crash-class) | trivial |
| B2–B4 | Moon distance, zodiacal elongation, airglow extinction | Medium-low | small each |
| B5, B7–B11 | Conventions and second-order terms | Low | small each |

A1 and A2 are the two findings that materially change results users are
already looking at; both are in the slitless/spectroscopy path, both are
fixable without touching the architecture, and both have regression tests
practically writing themselves (sky ∝ full band; line counts R-invariant).
