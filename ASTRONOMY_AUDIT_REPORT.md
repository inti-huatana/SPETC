# SPETC v10.0 — Astronomy / Astrophysics Review

Companion to `CODE_AUDIT_REPORT.md`, but this time judging the code as an
**exposure-time calculator physics engine**: are the radiometric conventions,
photometric zero points, S/N statistics, sky/moon models, extinction and airmass
scientifically correct? No code was changed. Confidence is stated per item.

Overall the *core radiometry is sound* (unit-safe `astropy` throughout, correct
photon-counting rates, correct CCD-equation S/N, correct SVO synthetic-magnitude
convention). The problems are concentrated in the **sky-background / moonlight
model**, where one formula is transcribed incorrectly and the sky brightness is
made independent of where the telescope is pointed.

---

## 1. Scientific bugs

### 1.1 Moonlight scattering function is misparenthesized — **HIGH severity, high confidence**
`sky_brightness.py::sky_brightness_total` (and the copy in `get_sky_components`),
line 178:

```python
fr = (10**(5.36 * (1.06 + np.cos(np.radians(rho))**2)) +
      10**(6.15 - rho / 40))
```

The Krisciunas & Schaefer (1991, PASP 103, 1033) scattering function is

&nbsp;&nbsp;&nbsp;&nbsp;**f(ρ) = 10^5.36 · (1.06 + cos²ρ) + 10^(6.15 − ρ/40)**

i.e. the constant **10^5.36 ≈ 2.29×10⁵ is *multiplied* by (1.06 + cos²ρ)**. The
code instead raises 10 to the power `5.36 · (1.06 + cos²ρ)`, which is a completely
different function. The error grows explosively at small Moon–target separation:

| Moon–target ρ | correct f(ρ) | code f(ρ) | factor wrong |
|---:|---:|---:|---:|
| 90° | ~2.5×10⁵ | ~4.9×10⁵ | ~2× |
| 30° | ~6.6×10⁵ | ~5×10⁹ | ~7000× |
| 10° | ~9×10⁵ | ~4×10¹⁰ | ~4×10⁴× |

The Mie/aureole term `10**(6.15 - rho/40)` is correct.

**Impact:** whenever the Moon is above the horizon the model's lunar sky
contribution is wrong — mildly at large separations, catastrophically within a few
tens of degrees of the Moon. In the delivered ETC this feeds
`etc_gui._sky_models_for_track`, where the Moon term enters as the *relative*
brightening `10^(-0.4(m_total − m_dark)) − 1 = q_moon/q_sky`; because `q_sky`
cancels, the buggy `f(ρ)` propagates directly into the sky brightness used for
source and background S/N. So bright-time and small-Moon-separation exposure
estimates are unreliable. The fix is a one-character class change:
`10**5.36 * (1.06 + cos²ρ)`.

### 1.2 Zodiacal light uses signed `sin`, not `sin|β|` — latent, medium confidence
`sky_brightness.py`, line 138 / 243:

```python
qzod = 140 - 90 * np.sin(np.radians(ecliptic_lat_deg))
```

Zodiacal light is symmetric about the ecliptic, so this must use the *absolute*
ecliptic latitude (`abs_eclat` is even computed on the line above but only used to
pick the `<60°` branch, not in the value). As written, southern ecliptic
latitudes (β<0) give `qzod > 140` (spuriously brighter) and northern ones give
`qzod < 140`. **Dormant** in the current GUI only because `ecliptic_lat_deg`
defaults to 0 (see 1.3); it becomes an active error the moment a real ecliptic
latitude is supplied.

### 1.3 Sky brightness does not depend on the target's position on the sky — modeling gap, high confidence
`etc_gui._sky_models_for_track` builds the per-time `common` dict without
`ecliptic_lat_deg` or `galactic_lat_deg`, so both default to 0 for **every**
target. Consequences:

- **Zodiacal light** is always evaluated at the ecliptic (β=0, its brightest),
  regardless of where the telescope points.
- **Integrated starlight / Milky Way** (`qstar = 100·exp(−|b|/10)`) is always
  evaluated at the galactic plane (b=0, its brightest).

The dark-sky *normalization* actually delivered comes from
`sky_background.sky_magnitude_vega`, which is a single U–K brightness table that
is **also position-independent** (it uses only the filter pivot wavelength). The
net effect: the ETC's dark-sky surface brightness is a fixed function of
band + airmass + solar cycle, with **no zodiacal or Milky Way dependence on the
actual field**. For a high-ecliptic-latitude, high-galactic-latitude target this
over-estimates the sky (pessimistic S/N); near the Galactic centre or ecliptic it
can under-estimate it. `ephemeris.py` already contains
`equatorial_to_ecliptic` / `equatorial_to_galactic` (currently dead code) that
would supply exactly the missing latitudes.

---

## 2. Physics that is correct (verified)

These were checked and are right — recorded so a future editor doesn't "fix" them:

- **Flux-density conversion.** `magnitude_f_lambda = ZP_ν·c/λ²` correctly turns an
  F_ν zero point (Jy) into F_λ; AB uses 3631 Jy. ✔
- **Photon-counting rate.** `electron_rate` integrates
  `F_λ·T·QE·T_atm·T_inst·A·η / (hc/λ)` over λ → e⁻/s. Units and the 1/photon-energy
  weighting are correct. ✔
- **Synthetic magnitudes (SVO convention).** `synthetic_magnitude` uses the
  photon (`∫F_λ R λ dλ`) vs energy (`∫F_λ R dλ`) weighting selected by
  `DetectorType`, referenced to a flat-F_ν SED at the passband `ZeroPoint`. This
  is the correct SVO "effective zero point" definition; a source whose in-band
  flux equals the ZP flux gets m=0. ✔
- **CCD equation.** `snr = S·t / sqrt((S+sky+dark)·t + N_pix·RN²)` in both the
  closed-form solver and the inline spectroscopy path. ✔ The inverse
  `exposure_time_for_snr` solves the correct quadratic including the read-noise
  floor. ✔
- **Aperture photometry.** Gaussian encircled energy `1−exp(−r²/2σ²)` for the
  source, full aperture sky (`μ_sky` × aperture area), `N_pix = area/plate²`,
  dark and read noise scaled by `N_pix`. ✔ Peak/saturation uses the *unextracted*
  PSF central-pixel fraction `[erf(0.5/√2σ_pix)]²`, correctly independent of the
  photometric aperture. ✔
- **Bouguer extinction.** `T(λ) = T_zenith(λ)^X`, with the Paranal FITS
  `EXTINCTION` (mag/airmass) mapped to `T_zenith = 10^(−0.4k)` and the airmass
  applied exactly once. ✔
- **Vega zero points** `BAND_VEGA_ZEROPOINT_JY` (V=3640, J=1594, K=666.7 Jy …)
  match the standard Bessell/Cohen values. ✔
- **K&S lunar illuminance** `I* = 10^(−0.4(3.84 + 0.026|α| + 4×10⁻⁹α⁴))` and the
  extinction term `10^(−0.4kX_moon)(1−10^(−0.4kX))` are transcribed correctly. ✔
  (Only the scattering function `f(ρ)` in the same block is wrong — see 1.1.)
- **Template flux calibration** reduces, in the V-only case, to the original
  Fortran `10^(−0.4(V_target − mv0))`. ✔ (but see caveat 3.1)

---

## 3. Approximations and caveats worth documenting (not bugs)

These are legitimate simplifications, but they set the accuracy floor of the ETC
and should be stated to users:

1. **Template calibration via `mv0`** — *withdrawn after author review.* The
   shipped templates are Bohlin CALSPEC/BPGS absolute spectrophotometry from HST
   data, so the tabulated `mv0` is the file's visual magnitude by construction
   and the Fortran `10^(0.4 mv0)` convention is exact for this catalogue.

2. **Airmass = sec z** (`secz` from AltAz, or `1/sin alt`). No Young/Rozenberg
   curvature term, so airmass is overestimated for X ≳ 3–4. The 5° altitude cutoff
   limits the damage. Refraction is disabled (`pressure=0`), so altitudes/airmass
   are geometric (airless) — fine away from the horizon.

3. **Two independent sky sub-systems with different airmass handling.**
   `sky_magnitude_vega` scales the *entire* dark-sky brightness (airglow +
   zodiacal + starlight, lumped in the U–K table) by airmass, whereas the
   Benn & Ellison code only scales the *airglow* term. Only airglow physically
   follows the (van Rhijn) air-path; scaling the zodiacal + stellar components by
   airmass as well is an over-simplification. The two models also duplicate
   `SOLAR_MINIMA` and the U–K tables (drift risk).

4. **Airglow–airmass law is linear** (`×X`). The van Rhijn function is sub-linear
   and turns over near the horizon; linear scaling over-brightens the sky at high
   airmass.

5. **Gaussian PSF everywhere.** Encircled energy and slit/extraction losses assume
   a Gaussian core with no Moffat wings, so aperture and slit throughputs are
   slightly optimistic for real seeing-limited profiles.

6. **Twilight** is a linear *flux* blend between the dark and daylight models over
   Sun altitude −18°→0°; the Weaver (1947) daylight table is broadband V mapped to
   other bands by a single dark-sky colour. Explicitly a rough bridge.

7. **Noise model is Poisson + read + dark only.** No scintillation, flat-field /
   fringe residual, sky-subtraction penalty, or digitization noise. For bright
   targets scintillation can dominate and is absent here.

8. **`bnl → S10` uses a bare factor 3.8** (line 190) and `MAG_ZERO_S10 = 27.78`.
   The zero point is standard; the 3.8 nanolambert→S10 conversion should be
   double-checked against K&S once 1.1 is fixed, since the two were presumably
   tuned together and the fr bug would have masked a miscalibration.

9. **Solar-cycle activity** (0.8 at min → 2.0 at max, 11.006-yr period from a fixed
   minima list) is an empirical airglow scaler, not a predictive space-weather
   model — acceptable, but it will extrapolate blindly past the last tabulated
   minimum (2019.96).

---

## 4. Suggested scientific additions

- **Fix 1.1, then add a moonlight regression test** against published K&S sky
  brightnesses (e.g. their Table for known α, ρ, X) so the scattering function
  can't silently regress again.
- **Wire target coordinates into the sky model** (fixes 1.3): compute ecliptic and
  galactic latitude from the already-entered RA/Dec via the existing (currently
  dead) `ephemeris.equatorial_to_ecliptic` / `equatorial_to_galactic`, and pass
  them to `sky_brightness_total`. Then fix 1.2 (`sin|β|`).
- **Adopt a Moffat (β≈2.5–4.7) PSF option** alongside the Gaussian for more
  realistic aperture and slit-loss estimates.
- **Add a scintillation term** (Young/Dravins 1.5-power law in D, X, altitude) so
  bright-star S/N is not overestimated.
- **A van Rhijn airglow–airmass law and a Young airmass polynomial** would improve
  behaviour at X > 3.
- **A validation harness against an independent ETC** (e.g. ESO/Paranal or a
  published SNR for a known standard star + instrument) to anchor absolute counts,
  not just the order-of-magnitude self-check currently in `self_check.py`.
- **Expose the sky-model provenance in the Results window** (which components,
  which airmass law, Moon on/off) so an observer can judge the estimate.

---

## 5. Priority summary

| # | Finding | Severity | Reached in GUI? |
|---|---------|----------|-----------------|
| 1.1 | K&S moon scattering `f(ρ)` misparenthesized | **High** | Yes, whenever Moon is up |
| 1.3 | Sky brightness independent of target ecliptic/galactic latitude | Medium–High | Yes, always |
| 1.2 | Zodiacal light uses signed `sin β` | Medium | Dormant (β always 0) |
| 3.2–3.4 | Airmass/airglow simplifications | Low | Yes |

The single highest-value scientific fix is **1.1** — it is a clear transcription
error against a well-known published formula and it corrupts every bright-time
estimate.
