"""
solvers.py
Reverse solvers: t_exp for target SNR, and SNR vs t_exp curves.

Bug fixed in this version
--------------------------
reverse_texp_for_snr() previously used:

    snr(t) = sqrt(source*t) / sqrt(source*t + sky*t)

which algebraically simplifies to sqrt(source/(source+sky)) -- completely
INDEPENDENT of t. The binary search was therefore searching a constant
function, and (depending on floating point comparison) would silently
walk to whichever bound of the search range it started closer to, e.g.
always returning t_exp_max (1e6 s) regardless of the requested SNR. The
correct Poisson SNR (matching the convention used throughout
photometry.py and spectroscopy.py) is:

    snr(t) = source*t / sqrt(source*t + sky*t)

which is directly invertible in closed form:

    t = snr_target**2 * (source + sky) / source**2

so no iterative search is needed at all. The previous "sky-limited,
impossible target" concept has also been removed: under pure Poisson
statistics (no read noise or other noise floor is modeled here), SNR
always grows as sqrt(t) without bound, however slowly, so there is no
truly unreachable target SNR in this noise model -- only impractically
long exposure times, which the returned t_exp value itself already
communicates.
"""

import numpy as np


def exposure_time_for_snr(target_snr, source_rate_e_s, sky_rate_e_s,
                          dark_rate_e_s=0.0, read_noise_e=0.0, n_pixels=1.0):
    """Closed-form exposure solution including sky, dark and read noise.

    For S = source rate and Q = source + sky + dark, the ETC convention is
    ``SNR = S t / sqrt(Q t + Npix RN^2)``.  Solving the resulting quadratic
    gives the positive real exposure time.
    """
    target_snr = float(target_snr)
    source_rate_e_s = float(source_rate_e_s)
    sky_rate_e_s = float(sky_rate_e_s)
    dark_rate_e_s = float(dark_rate_e_s)
    read_noise_e = float(read_noise_e)
    n_pixels = float(n_pixels)
    if target_snr <= 0 or source_rate_e_s <= 0 or sky_rate_e_s < 0 or dark_rate_e_s < 0 or n_pixels <= 0:
        return np.nan
    q_rate = source_rate_e_s + sky_rate_e_s + dark_rate_e_s
    read_variance = n_pixels * read_noise_e**2
    s2 = target_snr**2
    return (s2 * q_rate + np.sqrt((s2 * q_rate)**2 + 4 * source_rate_e_s**2 * s2 * read_variance)) / (2 * source_rate_e_s**2)


def plan_stack(target_snr, source_rate_e_s, sky_rate_e_s, dark_rate_e_s,
               read_noise_e, n_pixels, max_unsaturated_exptime_s,
               user_sub_exposure_s=0.0):
    """Plan a stack of sub-exposures reaching a target *total* S/N.

    The stacked CCD equation for N frames of length t (T = N t) is
    ``SNR = S T / sqrt(Q T + N Npix RN^2)`` with Q the total Poisson rate,
    which is closed-form in T for a fixed t:
    ``T = SNR^2 (Q + Npix RN^2 / t) / S^2``.

    The sub-exposure is the shortest of: the user's preference (0 = none),
    the saturation limit of the brightest pixel, and it is compared with the
    sky-limited exposure (background variance per pixel = 10 RN^2), below
    which read noise starts to matter.  The read-noise penalty is the extra
    total time relative to one ideal unsaturated exposure of length T.
    """
    target_snr = float(target_snr)
    source = float(source_rate_e_s)
    q_rate = source + float(sky_rate_e_s) + float(dark_rate_e_s)
    read_var = float(n_pixels) * float(read_noise_e) ** 2
    if target_snr <= 0 or source <= 0:
        return None
    sky_limited_sub_s = (10.0 * read_var / (float(sky_rate_e_s) + float(dark_rate_e_s))
                         if (sky_rate_e_s + dark_rate_e_s) > 0 else np.inf)
    limits = {"saturation": float(max_unsaturated_exptime_s)}
    if user_sub_exposure_s and user_sub_exposure_s > 0:
        limits["user"] = float(user_sub_exposure_s)
    sub_exposure_s = min(limits.values())
    limited_by = min(limits, key=limits.get)
    if not np.isfinite(sub_exposure_s) or sub_exposure_s <= 0:
        return None
    total_time_s = target_snr**2 * (q_rate + read_var / sub_exposure_s) / source**2
    n_frames = max(int(np.ceil(total_time_s / sub_exposure_s)), 1)
    stacked_time_s = n_frames * sub_exposure_s
    achieved_snr = (source * stacked_time_s
                    / np.sqrt(q_rate * stacked_time_s + n_frames * read_var))
    ideal_time_s = exposure_time_for_snr(target_snr, source, sky_rate_e_s,
                                         dark_rate_e_s, read_noise_e, n_pixels)
    penalty_percent = 100.0 * (total_time_s / ideal_time_s - 1.0) if ideal_time_s > 0 else np.nan
    return {"sub_exposure_s": sub_exposure_s, "n_frames": n_frames,
            "total_time_s": stacked_time_s, "achieved_snr": float(achieved_snr),
            "read_noise_penalty_percent": float(penalty_percent),
            "sky_limited_sub_s": float(sky_limited_sub_s), "limited_by": limited_by}


def reverse_texp_for_snr(
    target_snr,
    source_phot_per_s,
    sky_phot_per_s,
    mode='photometry'
):
    """
    Find the exposure time needed to reach a target SNR, in closed form.

    snr(t) = source*t / sqrt(source*t + sky*t)
    =>  t = target_snr**2 * (source + sky) / source**2

    Parameters
    ----------
    target_snr : float
        Desired SNR (> 0)
    source_phot_per_s : float
        Source photon rate [photons/s] (must be > 0)
    sky_phot_per_s : float
        Sky photon rate [photons/s] (>= 0)
    mode : {'photometry', 'spectroscopy'}
        Unused, kept for API compatibility with callers that pass it.

    Returns
    -------
    result : dict
        {'status': 'ok' | 'error',
         'texp_required_s': float or None,
         'min_snr_achievable': None,
         'reason': str or None,
         'snr_achieved': float}
    """
    if target_snr <= 0:
        return {
            'status': 'error', 'texp_required_s': None, 'min_snr_achievable': None,
            'reason': 'Target SNR must be positive', 'snr_achieved': None
        }

    if source_phot_per_s <= 0:
        return {
            'status': 'error', 'texp_required_s': None, 'min_snr_achievable': None,
            'reason': 'Source photon rate is zero or negative -- check target magnitude, '
                      'filter band, spectral template, and QE curve wavelength coverage.',
            'snr_achieved': None
        }

    if sky_phot_per_s < 0:
        return {
            'status': 'error', 'texp_required_s': None, 'min_snr_achievable': None,
            'reason': 'Sky photon rate is negative', 'snr_achieved': None
        }

    t_result = target_snr ** 2 * (source_phot_per_s + sky_phot_per_s) / source_phot_per_s ** 2

    source_total = source_phot_per_s * t_result
    sky_total = sky_phot_per_s * t_result
    snr_achieved = source_total / np.sqrt(max(source_total + sky_total, 1e-30))

    return {
        'status': 'ok',
        'texp_required_s': t_result,
        'min_snr_achievable': None,
        'reason': None,
        'snr_achieved': snr_achieved
    }


def compute_snr_vs_texp(source_phot_per_s, sky_phot_per_s, t_exp_array_s):
    """
    Compute SNR for an array of exposure times.

    snr(t) = source*t / sqrt(source*t + sky*t)

    Parameters
    ----------
    source_phot_per_s : float
    sky_phot_per_s : float
    t_exp_array_s : ndarray

    Returns
    -------
    ndarray
    """
    t_exp_array_s = np.asarray(t_exp_array_s, dtype=float)
    source_total = source_phot_per_s * t_exp_array_s
    sky_total = sky_phot_per_s * t_exp_array_s

    return source_total / np.sqrt(np.maximum(source_total + sky_total, 1e-30))


def compute_texp_vs_snr(source_phot_per_s, sky_phot_per_s, snr_array):
    """
    Compute the exposure time needed for each of an array of target SNRs,
    using the same closed-form relation as reverse_texp_for_snr().

    Parameters
    ----------
    source_phot_per_s : float
    sky_phot_per_s : float
    snr_array : ndarray

    Returns
    -------
    ndarray
    """
    snr_array = np.asarray(snr_array, dtype=float)
    if source_phot_per_s <= 0:
        return np.full_like(snr_array, np.nan)

    return snr_array ** 2 * (source_phot_per_s + sky_phot_per_s) / source_phot_per_s ** 2
