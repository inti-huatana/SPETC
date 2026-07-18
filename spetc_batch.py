"""Headless SPETC: run the ETC engines from a JSON configuration, no GUI.

Usage:
    python3 spetc_batch.py <run_config.json> [output_directory]

Reads a run description (see ``examples/batch_photometry.json`` and
``examples/batch_spectroscopy.json``), computes the requested photometric or
spectroscopic case with exactly the same engines the GUI uses, and writes:

* ``<name>_result.csv``   - the numerical result table;
* ``<name>_summary.html`` - a self-contained one-page run summary with the
  configuration, sky, key numbers, stack plan and an embedded S/N figure.

Sky modes supported headless: ``fixed_ab`` (observed AB surface brightness)
and ``sqm`` (zenith SQM V reading + built-in band colours).  The Moon and
twilight terms need the time-dependent track machinery of the GUI and are
not applied here; batch results assume the entered sky.
"""

import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import filter_catalog as fcat
import star_catalog as scat
from detector import Detector, load_qe_curve
from photometry import PhotometryETC
from spectroscopy import SpectroscopyETC
from observing_conditions import scintillation_variance_rate_e2_s
from sky_background import SKY_WAVELENGTH_AA, SKY_MAG_VEGA
from solvers import exposure_time_for_snr, plan_stack


def _sky_model(config, observing_vega_profile):
    sky = config["sky"]
    mode = str(sky.get("mode", "fixed_ab")).strip().lower()
    aperture = float(config.get("photometry", {}).get("aperture_radius_arcsec", 1.0))
    if mode == "fixed_ab":
        return {"sky_mag": float(sky["sky_mag"]), "sky_zero_point_jy": 3631.0,
                "sky_at_telescope": True, "aperture_radius_arcsec": aperture}
    if mode == "sqm":
        pivot = observing_vega_profile.pivot_wavelength_aa
        if not np.isfinite(pivot):
            pivot = float(np.average(observing_vega_profile.transmission[:, 0],
                                     weights=observing_vega_profile.transmission[:, 1]))
        v_table = float(np.interp(5510.0, SKY_WAVELENGTH_AA, SKY_MAG_VEGA))
        colour = float(np.interp(pivot, SKY_WAVELENGTH_AA, SKY_MAG_VEGA)) - v_table
        return {"sky_mag": float(sky["sqm_v_mag_arcsec2"]) + colour,
                "sky_zero_point_jy": observing_vega_profile.zero_point_jy,
                "sky_at_telescope": True, "aperture_radius_arcsec": aperture}
    raise ValueError("Batch sky mode must be 'fixed_ab' or 'sqm'.")


def _load_context(config):
    data_dir = Path(config.get("data_directory", "data"))
    catalog = scat.parse_interpola_db(data_dir / "interpola.db.csv")
    template_name = config["target"]["template"]
    record = scat.find_star_by_name(catalog, template_name)
    if record is None:
        raise ValueError(f"Template {template_name!r} not found in the catalogue.")
    star_spec = scat.load_star_spectrum(record, data_dir)
    system = config["target"].get("magnitude_system", "Vega")
    reference_profile = fcat.load_filter_profile(data_dir, config["target"]["reference_filter"], system)
    observing_name = config.get("observing_filter", "BLANK")
    observing_profile = fcat.load_filter_profile(data_dir, observing_name, system)
    observing_vega = fcat.load_filter_profile(data_dir, observing_name, "Vega")
    visual_profile = fcat.load_filter_profile(data_dir, "Bessell.V", "Vega")
    qe_curve = load_qe_curve(data_dir / "qe.dat", config.get("qe_wavelength_unit", "Angstrom"))
    atmosphere = dict(config.get("atmosphere", {}))
    atmosphere.setdefault("airmass", 1.2)
    atmosphere.setdefault("seeing_arcsec", 2.0)
    atmosphere["transmission_curve"] = fcat.generic_zenith_atmosphere_curve()
    detector = Detector(**config["detector"])
    return {"record": record, "star_spec": star_spec, "reference": reference_profile,
            "observing": observing_profile, "observing_vega": observing_vega,
            "visual": visual_profile, "qe": qe_curve, "atmosphere": atmosphere,
            "detector": detector, "telescope": config["telescope"]}


def _calibration_kwargs(context):
    return dict(target_zero_point_jy=context["reference"].zero_point_jy,
                reference_filter=context["reference"].transmission,
                template_mv0=context["record"].mv0,
                visual_band=context["visual"].transmission,
                visual_zero_point_jy=context["visual"].zero_point_jy,
                reference_detector_type=context["reference"].detector_type,
                visual_detector_type=context["visual"].detector_type)


def _figure_base64(figure):
    stream = io.BytesIO()
    figure.tight_layout(pad=0.4)
    figure.savefig(stream, format="png", dpi=130, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _run_photometry(config, context):
    photometry = config.get("photometry", {})
    etc = PhotometryETC(context["telescope"], context["detector"], context["atmosphere"],
                        _sky_model(config, context["observing_vega"]))
    magnitude = float(config["target"]["magnitude"])
    t_exp = float(config.get("exposure_time_s", 60.0))
    kwargs = _calibration_kwargs(context)
    kwargs.update(observing_zero_point_jy=context["observing"].zero_point_jy,
                  observing_detector_type=context["observing"].detector_type,
                  source_geometry=photometry.get("source_geometry", "point"),
                  source_area_arcsec2=photometry.get("source_area_arcsec2"))
    target_snr = config.get("target_snr")
    result = etc.compute_photometry_single(context["star_spec"], context["observing"].transmission,
                                           context["qe"], magnitude, t_exp, **kwargs)
    scint_rate = scintillation_variance_rate_e2_s(
        result["source_rate_per_s"], float(context["telescope"]["diameter_mm"]),
        float(context["atmosphere"]["airmass"]), float(context["atmosphere"].get("elevation_m", 0.0)))
    if target_snr:
        t_exp = exposure_time_for_snr(float(target_snr), result["source_rate_per_s"],
                                      result["sky_rate_per_s"],
                                      context["detector"].dark_current_e_s_pix * result["n_pixels"],
                                      context["detector"].read_noise_e, result["n_pixels"],
                                      extra_variance_rate_e2_s=scint_rate)
        if not np.isfinite(t_exp):
            raise ValueError("The requested target S/N is not reachable for these rates.")
        result = etc.compute_photometry_single(context["star_spec"], context["observing"].transmission,
                                               context["qe"], magnitude, t_exp, **kwargs)
    plan = None
    if target_snr:
        plan = plan_stack(float(target_snr), result["source_rate_per_s"], result["sky_rate_per_s"],
                          context["detector"].dark_current_e_s_pix * result["n_pixels"],
                          context["detector"].read_noise_e, result["n_pixels"],
                          result["max_unsaturated_exptime_s"],
                          float(config.get("stack_sub_exposure_s", 0.0)),
                          extra_variance_rate_e2_s=scint_rate)
    # S/N versus exposure-time curve for the summary figure.
    times = np.logspace(0, np.log10(max(4.0 * t_exp, 600.0)), 30)
    curve = [etc.compute_photometry_single(context["star_spec"], context["observing"].transmission,
                                           context["qe"], magnitude, float(value), **kwargs)["snr"]
             for value in times]
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    axis.loglog(times, curve, color="#0072B2")
    axis.axvline(t_exp, color="#D55E00", linestyle="--", linewidth=1,
                 label=f"selected {t_exp:.0f} s")
    axis.axvline(result["max_unsaturated_exptime_s"], color="#777777", linestyle=":",
                 linewidth=1, label="saturation limit")
    axis.set_xlabel("exposure time [s]"); axis.set_ylabel("S/N")
    axis.grid(True, which="both", color="#eeeeee"); axis.legend(fontsize=8)
    import pandas as pd
    table = pd.DataFrame([result])
    key_rows = [("Exposure [s]", f"{t_exp:.1f}"), ("S/N", f"{result['snr']:.1f}"),
                ("Source rate [e-/s]", f"{result['photons_source_es']:.4g}"),
                ("Sky rate [e-/s]", f"{result['photons_sky_es']:.4g}"),
                ("Peak pixel [e-]", f"{result['peak_e_unclipped']:.4g}"),
                ("Saturation", result["saturation_flag"]),
                ("Max unsaturated exposure [s]", f"{result['max_unsaturated_exptime_s']:.1f}"),
                ("Scintillation noise [e-]", f"{result['scintillation_noise_e']:.4g}"),
                ("Estimated observing magnitude", f"{result['estimated_observing_magnitude']:.3f}")]
    return table, key_rows, plan, _figure_base64(figure)


def _run_spectroscopy(config, context):
    spectroscopy = config.get("spectroscopy", {})
    etc = SpectroscopyETC(context["telescope"], context["detector"], context["atmosphere"],
                          _sky_model(config, context["observing_vega"]))
    kwargs = _calibration_kwargs(context)
    result = etc.compute_spectroscopy(
        context["star_spec"], float(spectroscopy.get("resolution_R", 1000.0)),
        float(spectroscopy.get("slit_width_arcsec", 25.0)),
        float(config.get("exposure_time_s", 300.0)),
        (float(spectroscopy.get("wavelength_min_aa", 4000.0)),
         float(spectroscopy.get("wavelength_max_aa", 7500.0))),
        float(config["target"]["magnitude"]), context["qe"],
        context["reference"].transmission,
        pixels_per_resel=float(spectroscopy.get("pixels_per_resel", 2.0)),
        extraction_height_arcsec=spectroscopy.get("extraction_height_arcsec"),
        spectroscopy_mode=spectroscopy.get("mode", "slit"),
        slitless_extraction_width_arcsec=spectroscopy.get("slitless_extraction_width_arcsec"),
        slitless_dispersion_aa_pix=spectroscopy.get("slitless_dispersion_aa_pix", 10.0),
        slitless_intrinsic_fwhm_pix=float(spectroscopy.get("slitless_intrinsic_fwhm_pix", 1.0)),
        observing_filter=context["observing"].transmission,
        slitless_grating_lines_mm=spectroscopy.get("grating_lines_mm"),
        slitless_grating_distance_mm=spectroscopy.get("grating_distance_mm"),
        grating_efficiency=float(spectroscopy.get("grating_efficiency", 1.0)),
        slit_at_parallactic=bool(spectroscopy.get("slit_at_parallactic", True)),
        **kwargs)
    reference_wl = float(spectroscopy.get("reference_wavelength_aa", 5500.0))
    ref_row = result.iloc[int(np.argmin(np.abs(result["wavelength_aa"].to_numpy() - reference_wl)))]
    plan = None
    if config.get("target_snr"):
        scint_rate = scintillation_variance_rate_e2_s(
            float(ref_row["photons_source_es"]), float(context["telescope"]["diameter_mm"]),
            float(context["atmosphere"]["airmass"]), float(context["atmosphere"].get("elevation_m", 0.0)))
        plan = plan_stack(float(config["target_snr"]), float(ref_row["photons_source_es"]),
                          float(ref_row["photons_sky_es"]),
                          context["detector"].dark_current_e_s_pix * result.attrs["n_pixels_per_resel"],
                          context["detector"].read_noise_e, result.attrs["n_pixels_per_resel"],
                          float(ref_row["max_unsaturated_exptime_s"]),
                          float(config.get("stack_sub_exposure_s", 0.0)),
                          extra_variance_rate_e2_s=scint_rate)
    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True)
    ax1.plot(result["wavelength_aa"], result["photons_source_es"], color="#0072B2")
    ax1.set_ylabel("source [e-/s/resel]")
    ax2.plot(result["wavelength_aa"], result["snr"], color="#009E73")
    ax2.set_ylabel("S/N per resel"); ax2.set_xlabel("wavelength [A]")
    for axis in (ax1, ax2):
        axis.grid(True, color="#eeeeee")
    key_rows = [("Mode", result.attrs["spectroscopy_mode"]),
                ("Median resolving power R", f"{result.attrs['effective_resolution_R']:.0f}"),
                (f"S/N at {reference_wl:.0f} A", f"{float(ref_row['snr']):.1f}"),
                (f"sigma(EW) at {reference_wl:.0f} A [mA]", f"{float(ref_row['sigma_ew_mangstrom']):.1f}"),
                ("Saturation at reference", str(ref_row["saturation_flag"])),
                ("Max unsaturated exposure [s]", f"{float(ref_row['max_unsaturated_exptime_s']):.1f}")]
    return result, key_rows, plan, _figure_base64(figure)


def _summary_html(config, key_rows, plan, figure_b64, csv_name):
    def rows(pairs):
        return "\n".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in pairs)
    setup = [("Mode", config.get("mode", "photometry")),
             ("Template", config["target"]["template"]),
             ("Magnitude", f'{config["target"]["magnitude"]} {config["target"].get("magnitude_system", "Vega")} '
                           f'in {config["target"]["reference_filter"]}'),
             ("Observing filter", config.get("observing_filter", "BLANK")),
             ("Telescope", f'{config["telescope"]["diameter_mm"]} mm, f={config["telescope"]["focal_length_mm"]} mm'),
             ("Airmass / seeing", f'{config.get("atmosphere", {}).get("airmass", 1.2)} / '
                                  f'{config.get("atmosphere", {}).get("seeing_arcsec", 2.0)}"'),
             ("Sky", json.dumps(config["sky"]))]
    plan_html = ""
    if plan:
        plan_html = ("<h2>Stack plan</h2><table>"
                     + rows([("Frames", f"{plan['n_frames']} × {plan['sub_exposure_s']:.0f} s "
                                        f"(limited by {plan['limited_by']})"),
                             ("Total time [s]", f"{plan['total_time_s']:.0f}"),
                             ("Achieved S/N", f"{plan['achieved_snr']:.1f}"),
                             ("Read-noise penalty", f"+{plan['read_noise_penalty_percent']:.1f}%"),
                             ("Background beats read noise above", f"{plan['sky_limited_sub_s']:.0f} s/frame")])
                     + "</table>")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SPETC run: {config.get('name', 'run')}</title>
<style>
body {{ font-family: Georgia, serif; margin: 2em auto; max-width: 52em; color: #111; }}
h1 {{ font-size: 1.4em; border-bottom: 2px solid #444; }}
h2 {{ font-size: 1.1em; margin-top: 1.2em; }}
table {{ border-collapse: collapse; margin: 0.5em 0; }}
th, td {{ border: 1px solid #bbb; padding: 3px 10px; text-align: left; font-size: 0.92em; }}
th {{ background: #f2f2f2; font-weight: normal; }}
img {{ max-width: 100%; }}
.footer {{ color: #666; font-size: 0.8em; margin-top: 1.5em; }}
</style></head><body>
<h1>SPETC run summary — {config.get('name', 'run')}</h1>
<h2>Configuration</h2><table>{rows(setup)}</table>
<h2>Key results</h2><table>{rows(key_rows)}</table>
{plan_html}
<h2>S/N</h2><img src="data:image/png;base64,{figure_b64}" alt="S/N figure"/>
<p class="footer">Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by spetc_batch.py;
full table in <code>{csv_name}</code>. Batch sky assumes the entered brightness
(no Moon/twilight terms).</p>
</body></html>
"""


def run_batch(config_path, output_dir="."):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    context = _load_context(config)
    name = config.get("name", Path(config_path).stem)
    mode = str(config.get("mode", "photometry")).strip().lower()
    if mode == "photometry":
        table, key_rows, plan, figure_b64 = _run_photometry(config, context)
    elif mode == "spectroscopy":
        table, key_rows, plan, figure_b64 = _run_spectroscopy(config, context)
    else:
        raise ValueError("Batch mode must be 'photometry' or 'spectroscopy'.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}_result.csv"
    html_path = output_dir / f"{name}_summary.html"
    table.to_csv(csv_path, index=False)
    html_path.write_text(_summary_html(config, key_rows, plan, figure_b64, csv_path.name),
                         encoding="utf-8")
    return {"csv": csv_path, "html": html_path, "key_rows": key_rows, "plan": plan}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    outcome = run_batch(sys.argv[1], output_dir)
    for label, value in outcome["key_rows"]:
        print(f"{label}: {value}")
    if outcome["plan"]:
        plan = outcome["plan"]
        print(f"Stack plan: {plan['n_frames']} x {plan['sub_exposure_s']:.0f} s "
              f"= {plan['total_time_s']:.0f} s (S/N {plan['achieved_snr']:.1f})")
    print(f"wrote {outcome['csv']}\nwrote {outcome['html']}")


if __name__ == "__main__":
    main()
