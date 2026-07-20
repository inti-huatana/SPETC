"""Local horizon profile from the Copernicus GLO-30 digital elevation model.

Computes the apparent elevation of the terrain horizon as a function of
azimuth for an observing site, from the public Copernicus DSM 30 m tiles on
AWS (no credentials required; tiles are cached locally).  The apparent
angle includes the Earth-curvature drop and the standard 34-arcminute
horizontal refraction.

The heavy geospatial dependencies (rasterio, pyproj) are imported lazily:
the pure horizon mathematics (``apparent_elevation_angle``,
``compute_horizon_profile``) and the CSV input/output work without them,
so the rest of SPETC never requires them.  Generating a new profile does:

    pip install rasterio pyproj

Output CSV format (one header block, then one row per azimuth):

    # SPETC horizon profile
    # latitude_deg: 45.848600
    # longitude_deg: 11.569600
    # radius_km: 10.0
    # center_elevation_m: 1366.0
    azimuth_deg,horizon_elevation_deg
    0.0,1.234
    1.0,1.198

Azimuth is degrees from North through East (the SPETC convention);
horizon elevation is degrees above the astronomical horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
DEM_BUCKET_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
HORIZON_REFRACTION_DEG = 34.0 / 60.0  # standard refraction at the horizon
DEM_CACHE_DIR = Path(__file__).resolve().parent / "dem_cache"
HORIZON_DIR = Path(__file__).resolve().parent / "data" / "horizons"

MIN_RADIUS_KM = 1.0
MAX_RADIUS_KM = 100.0
KM_PER_MILE = 1.609344

# Physical bounds on the reported horizon elevation: from the top of a very
# high mountain looking down into a valley (-20 deg) up to straight overhead
# (+90 deg).  The computed profile is clamped to this range.
HORIZON_MIN_DEG = -20.0
HORIZON_MAX_DEG = 90.0


def _require_geo_stack():
    try:
        import rasterio  # noqa: F401
        import pyproj    # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Generating a horizon profile needs the geospatial libraries: "
            "pip install rasterio pyproj") from exc


@dataclass
class LocalGrid:
    """Elevation grid on a local East/North metre frame centred on the site."""
    elevation_m: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    center_elevation_m: float


# ---------------------------------------------------------------------------
# Copernicus GLO-30 tile access (public AWS bucket, cached locally)
# ---------------------------------------------------------------------------
def _tile_name(lat_deg, lon_deg):
    lat_floor, lon_floor = math.floor(lat_deg), math.floor(lon_deg)
    ns, ew = ("N" if lat_floor >= 0 else "S"), ("E" if lon_floor >= 0 else "W")
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def _download_tile(tile_name, cache_dir):
    """Return the cached tile path, downloading once; None for ocean (404)."""
    import urllib.error
    import urllib.request
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{tile_name}.tif"
    if local_path.exists():
        return local_path
    url = f"{DEM_BUCKET_URL}/{tile_name}/{tile_name}.tif"
    try:
        urllib.request.urlretrieve(url, local_path)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"Failed to download {tile_name}: HTTP {exc.code}") from exc
    except Exception as exc:
        local_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {tile_name}: {exc}") from exc
    return local_path


def _tiles_covering_bbox(lat_min, lat_max, lon_min, lon_max):
    return [_tile_name(lat + 0.5, lon + 0.5)
            for lat in range(math.floor(lat_min), math.floor(lat_max) + 1)
            for lon in range(math.floor(lon_min), math.floor(lon_max) + 1)]


def _synthetic_nan_tile(tile_name, crs, resolution):
    """In-memory NaN tile standing in for missing (ocean) coverage."""
    import re
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    ns, lat_txt, ew, lon_txt = re.search(r"_([NS])(\d{2})_00_([EW])(\d{3})_00_DEM", tile_name).groups()
    lat_origin = int(lat_txt) * (1 if ns == "N" else -1)
    lon_origin = int(lon_txt) * (1 if ew == "E" else -1)
    px_x, px_y = resolution
    n_rows, n_cols = int(round(1.0 / px_y)), int(round(1.0 / px_x))
    memfile = MemoryFile()
    with memfile.open(driver="GTiff", height=n_rows, width=n_cols, count=1, dtype="float32",
                      crs=crs, transform=from_origin(lon_origin, lat_origin + 1.0, px_x, px_y),
                      nodata=np.nan) as dst:
        dst.write(np.full((n_rows, n_cols), np.nan, dtype="float32"), 1)
    return memfile.open()


def fetch_dem_window(center_lat, center_lon, radius_m, cache_dir=DEM_CACHE_DIR):
    """Elevation array, transform and CRS covering the site circle; ocean = NaN."""
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds
    deg_lat = (radius_m / 111_320.0) * 1.05
    deg_lon = deg_lat / max(math.cos(math.radians(center_lat)), 0.1)
    lat_min, lat_max = center_lat - deg_lat, center_lat + deg_lat
    lon_min, lon_max = center_lon - deg_lon, center_lon + deg_lon
    tile_names = _tiles_covering_bbox(lat_min, lat_max, lon_min, lon_max)
    tile_paths = [_download_tile(name, cache_dir) for name in tile_names]
    if all(path is None for path in tile_paths):
        raise RuntimeError("All DEM tiles are missing here (open ocean?).")
    template = next(path for path in tile_paths if path is not None)
    with rasterio.open(template) as src:
        crs, resolution = src.crs, src.res
    sources = [rasterio.open(path) if path else _synthetic_nan_tile(name, crs, resolution)
               for name, path in zip(tile_names, tile_paths)]
    if len(sources) == 1:
        src = sources[0]
        window = window_from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        elevation = src.read(1, window=window)
        transform = src.window_transform(window)
        src.close()
        return elevation, transform, str(crs)
    from rasterio.merge import merge
    mosaic, transform = merge(sources, bounds=(lon_min, lat_min, lon_max, lat_max))
    for src in sources:
        src.close()
    return mosaic[0], transform, str(crs)


def build_local_grid(center_lat, center_lon, radius_m, target_resolution_m=30.0,
                     cache_dir=DEM_CACHE_DIR):
    """Resample the DEM onto a local East/North metre grid centred on the site."""
    _require_geo_stack()
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    elevation_geo, transform_geo, crs_geo = fetch_dem_window(center_lat, center_lon, radius_m, cache_dir)
    zone = int((center_lon + 180.0) // 6) + 1
    utm_crs = f"EPSG:32{6 if center_lat >= 0 else 7}{zone:02d}"
    cx, cy = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform(center_lon, center_lat)
    n_grid = int(2 * radius_m / target_resolution_m) + 1
    axis = np.linspace(-radius_m, radius_m, n_grid)
    destination = np.full((n_grid, n_grid), np.nan, dtype=np.float64)
    reproject(source=elevation_geo, destination=destination,
              src_transform=transform_geo, src_crs=crs_geo,
              dst_transform=from_origin(cx + axis[0], cy + axis[-1],
                                        target_resolution_m, target_resolution_m),
              dst_crs=utm_crs, resampling=Resampling.bilinear)
    destination = np.flipud(destination)  # row 0 = south
    centre = n_grid // 2
    return LocalGrid(destination, axis, axis, float(destination[centre, centre]))


# ---------------------------------------------------------------------------
# Horizon mathematics (pure; no geospatial dependencies)
# ---------------------------------------------------------------------------
def apparent_elevation_angle(delta_h_m, distance_m):
    """Apparent elevation [deg] of a terrain point: geometry - curvature + refraction."""
    if distance_m <= 0.0:
        return 90.0 if delta_h_m > 0 else -90.0
    effective_height = delta_h_m - distance_m**2 / (2.0 * EARTH_RADIUS_M)
    return math.degrees(math.atan2(effective_height, distance_m)) + HORIZON_REFRACTION_DEG


def compute_horizon_profile(grid, n_azimuths=360):
    """Horizon elevation versus azimuth (deg from North through East)."""
    from scipy.interpolate import RegularGridInterpolator
    azimuths = np.linspace(0.0, 360.0, int(n_azimuths), endpoint=False)
    horizon = np.zeros(azimuths.size, dtype=np.float64)
    max_radius = min(abs(grid.east_m[0]), grid.east_m[-1], abs(grid.north_m[0]), grid.north_m[-1])
    pixel = grid.east_m[1] - grid.east_m[0]
    distances = np.linspace(pixel, max_radius, max(int(max_radius / pixel), 2))
    interpolator = RegularGridInterpolator((grid.north_m, grid.east_m), grid.elevation_m,
                                           bounds_error=False, fill_value=np.nan)
    curvature = distances**2 / (2.0 * EARTH_RADIUS_M)
    for i, azimuth in enumerate(np.radians(azimuths)):
        samples = interpolator(np.column_stack((distances * np.cos(azimuth),
                                                distances * np.sin(azimuth))))
        valid = np.isfinite(samples)
        if not np.any(valid):
            continue
        delta_h = samples[valid] - grid.center_elevation_m - curvature[valid]
        angles = np.degrees(np.arctan2(delta_h, distances[valid])) + HORIZON_REFRACTION_DEG
        horizon[i] = float(np.max(angles))
    # Constrain to the physical range: never below a deep valley view
    # (-20 deg) nor above the zenith (+90 deg).
    horizon = np.clip(horizon, HORIZON_MIN_DEG, HORIZON_MAX_DEG)
    return azimuths, horizon


# ---------------------------------------------------------------------------
# CSV input/output
# ---------------------------------------------------------------------------
def horizon_csv_path(lat_deg, lon_deg, radius_km, directory=HORIZON_DIR):
    return Path(directory) / f"horizon_lat{lat_deg:+.4f}_lon{lon_deg:+.4f}_r{radius_km:g}km.csv"


def save_horizon_csv(path, azimuths_deg, horizon_deg, lat_deg, lon_deg, radius_km,
                     center_elevation_m):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SPETC horizon profile",
             f"# latitude_deg: {lat_deg:.6f}",
             f"# longitude_deg: {lon_deg:.6f}",
             f"# radius_km: {radius_km:g}",
             f"# center_elevation_m: {center_elevation_m:.1f}",
             "azimuth_deg,horizon_elevation_deg"]
    lines += [f"{az:.1f},{el:.3f}" for az, el in zip(azimuths_deg, horizon_deg)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_horizon_csv(path):
    """Return (azimuths_deg, horizon_deg, metadata dict)."""
    path = Path(path)
    metadata, rows = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            if ":" in line:
                key, _, value = line.lstrip("# ").partition(":")
                try:
                    metadata[key.strip()] = float(value)
                except ValueError:
                    metadata[key.strip()] = value.strip()
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No horizon rows found in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1], metadata


def horizon_png_path(lat_deg, lon_deg, radius_km, directory=HORIZON_DIR):
    return Path(directory) / f"horizon_lat{lat_deg:+.4f}_lon{lon_deg:+.4f}_r{radius_km:g}km.png"


def save_horizon_png(path, azimuths_deg, horizon_deg, lat_deg, lon_deg, radius_km,
                     center_elevation_m):
    """Write a standalone PNG of the horizon profile (no GUI needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=120)
    ax.fill_between(azimuths_deg, horizon_deg, HORIZON_MIN_DEG,
                    where=horizon_deg > HORIZON_MIN_DEG, color="#b0c4de", alpha=0.7)
    ax.plot(azimuths_deg, horizon_deg, color="#1f4f82", linewidth=1.3)
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set(xlim=(0, 360), ylim=(HORIZON_MIN_DEG, max(10.0, float(np.max(horizon_deg)) + 3.0)),
           xlabel="Azimuth [deg]  (N=0, E=90, S=180, W=270)",
           ylabel="Horizon elevation [deg]",
           title=f"SPETC horizon  lat {lat_deg:+.4f}  lon {lon_deg:+.4f}  "
                 f"r = {radius_km:g} km  (site {center_elevation_m:.0f} m)")
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.grid(True, color="#dddddd")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_horizon(lat_deg, lon_deg, radius_km, directory=HORIZON_DIR,
                     cache_dir=DEM_CACHE_DIR, n_azimuths=360):
    """Full pipeline: DEM download, horizon computation, automatic CSV+PNG save."""
    radius_km = float(radius_km)
    if not MIN_RADIUS_KM <= radius_km <= MAX_RADIUS_KM:
        raise ValueError(f"Horizon radius must be between {MIN_RADIUS_KM:g} and "
                         f"{MAX_RADIUS_KM:g} km (got {radius_km:g}).")
    grid = build_local_grid(float(lat_deg), float(lon_deg), radius_km * 1000.0,
                            cache_dir=cache_dir)
    azimuths, horizon = compute_horizon_profile(grid, n_azimuths)
    path = save_horizon_csv(horizon_csv_path(lat_deg, lon_deg, radius_km, directory),
                            azimuths, horizon, lat_deg, lon_deg, radius_km,
                            grid.center_elevation_m)
    png_path = save_horizon_png(horizon_png_path(lat_deg, lon_deg, radius_km, directory),
                                azimuths, horizon, lat_deg, lon_deg, radius_km,
                                grid.center_elevation_m)
    return {"path": path, "png_path": png_path, "azimuths_deg": azimuths, "horizon_deg": horizon,
            "center_elevation_m": grid.center_elevation_m,
            "max_horizon_deg": float(np.max(horizon)),
            "max_horizon_azimuth_deg": float(azimuths[int(np.argmax(horizon))])}


def horizon_elevation_at_azimuth(azimuths_deg, horizon_deg, query_azimuth_deg):
    """Interpolate the horizon elevation at one or more azimuths (deg).

    Used to couple the terrain horizon with target visibility: a target is
    blocked when its altitude is below the horizon at its azimuth.
    """
    az = np.asarray(azimuths_deg, dtype=float)
    order = np.argsort(az)
    az_sorted, hor_sorted = az[order], np.asarray(horizon_deg, dtype=float)[order]
    # Wrap the profile so interpolation is periodic in azimuth.
    az_wrapped = np.concatenate((az_sorted - 360.0, az_sorted, az_sorted + 360.0))
    hor_wrapped = np.concatenate((hor_sorted, hor_sorted, hor_sorted))
    return np.interp(np.asarray(query_azimuth_deg, dtype=float) % 360.0, az_wrapped, hor_wrapped)
