"""
Module vent ERA5 — Champ de vent réel interpolé spatiotemporellement
depuis les réanalyses ERA5 (ECMWF) à 850 hPa.

Remplace le modèle simplifié (wind.py) : chaque particule reçoit
un vent LOCAL interpolé bilinéairement à sa position (lon, lat)
et linéairement dans le temps.

Conversion : m/s → degrés/heure, avec correction cos(lat) pour
la composante zonale (longitude).
"""
import os
import datetime
import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from config import GRID


# ── Singleton : données ERA5 chargées une seule fois ──────────────────
_era5 = None
_interp_cache = {}

# Date de l'accident : 26 avril 1986, 01:00 UTC (arrondi)
_ACCIDENT_UTC = np.datetime64("1986-04-26T01:00:00")


def _load(era5_path):
    """Charge le NetCDF ERA5 en mémoire (une seule fois)."""
    global _era5
    if _era5 is not None:
        return _era5

    if not os.path.isfile(era5_path):
        raise FileNotFoundError(
            f"Fichier ERA5 introuvable : {era5_path}\n"
            "→ Exécuter d'abord :  python download_era5.py"
        )

    print("  Chargement ERA5...", end=" ", flush=True)
    ds = xr.open_dataset(era5_path)

    # Squeeze la dimension pression si présente
    for dim_name in ("level", "pressure_level"):
        if dim_name in ds.dims:
            ds = ds.squeeze(dim_name, drop=True)

    # Déterminer les noms de variables (u/v)
    if "u" in ds:
        u_name, v_name = "u", "v"
    elif "u_component_of_wind" in ds:
        u_name, v_name = "u_component_of_wind", "v_component_of_wind"
    else:
        raise KeyError(f"Variables u/v non trouvées. Disponibles : {list(ds.data_vars)}")

    lats = ds["latitude"].values.astype(np.float64)
    lons = ds["longitude"].values.astype(np.float64)
    times = ds["time"].values  # numpy datetime64

    # Heures depuis l'accident
    t_hours = ((times - _ACCIDENT_UTC) / np.timedelta64(1, "h")).astype(np.float64)

    # S'assurer que les latitudes sont croissantes
    # (ERA5 les fournit souvent décroissantes : 72→33)
    lat_ascending = lats[0] < lats[-1]
    if not lat_ascending:
        lats = lats[::-1]
        u_data = ds[u_name].values[:, ::-1, :]
        v_data = ds[v_name].values[:, ::-1, :]
    else:
        u_data = ds[u_name].values
        v_data = ds[v_name].values

    _era5 = {
        "lats": lats,            # croissant
        "lons": lons,            # croissant
        "t_hours": t_hours,      # heures depuis l'accident
        "u": u_data,             # (n_times, n_lats, n_lons) en m/s
        "v": v_data,             # idem
    }

    ds.close()

    if (lats.min() > GRID["lat_min"] or lats.max() < GRID["lat_max"] or
            lons.min() > GRID["lon_min"] or lons.max() < GRID["lon_max"]):
        warnings.warn(
            "Le domaine ERA5 est plus petit que la grille de simulation ; "
            "certaines particules pourront être hors domaine.",
            RuntimeWarning,
            stacklevel=2,
        )

    print(f"OK — {len(t_hours)} pas de temps, "
          f"lat [{lats[0]:.1f}→{lats[-1]:.1f}], "
          f"lon [{lons[0]:.1f}→{lons[-1]:.1f}], "
          f"t [{t_hours[0]:.0f}→{t_hours[-1]:.0f}]h")

    return _era5


def _interpolator(t_idx, component):
    """Crée (et cache) un interpolateur pour un instant + composante."""
    key = (t_idx, component)
    if key in _interp_cache:
        return _interp_cache[key]

    data = _era5
    values = data["u"][t_idx] if component == "u" else data["v"][t_idx]

    interp = RegularGridInterpolator(
        (data["lats"], data["lons"]),
        values,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Cache limité pour ne pas exploser la mémoire
    if len(_interp_cache) > 30:
        _interp_cache.clear()
    _interp_cache[key] = interp
    return interp


def get_wind(lons, lats, t_hours, era5_path="data/era5_chernobyl_1986.nc",
             turbulence=0.25, rng=None):
    """
    Vent ERA5 interpolé à chaque position de particule.

    Parameters
    ----------
    lons : ndarray (n,) — longitudes des particules (degrés)
    lats : ndarray (n,) — latitudes des particules (degrés)
    t_hours : float — heures depuis l'accident
    era5_path : str — chemin vers le fichier NetCDF ERA5
    turbulence : float — amplitude du bruit turbulent (fraction du vent)

    Returns
    -------
    u_deg, v_deg : ndarray (n,) — vent en degrés/heure
    """
    data = _load(era5_path)
    t_arr = data["t_hours"]
    n = len(lons)
    outside_domain = (
        (lats < data["lats"].min()) | (lats > data["lats"].max()) |
        (lons < data["lons"].min()) | (lons > data["lons"].max())
    )
    if np.any(outside_domain):
        warnings.warn(
            f"{int(np.sum(outside_domain))} particule(s) hors domaine ERA5 ; "
            "fill_value=0 appliqué.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Interpolation temporelle ──────────────────────────────────────
    # Trouver les deux pas de temps encadrants
    idx = np.searchsorted(t_arr, t_hours) - 1
    idx = max(0, min(idx, len(t_arr) - 2))

    t0, t1 = t_arr[idx], t_arr[idx + 1]
    dt = max(t1 - t0, 1e-10)
    alpha = np.clip((t_hours - t0) / dt, 0.0, 1.0)

    # ── Interpolation spatiale aux deux instants ──────────────────────
    points = np.column_stack([lats, lons])  # (n, 2) — [lat, lon]

    u0 = _interpolator(idx, "u")(points)
    u1 = _interpolator(idx + 1, "u")(points)
    v0 = _interpolator(idx, "v")(points)
    v1 = _interpolator(idx + 1, "v")(points)

    # Interpolation temporelle linéaire
    u_ms = (1 - alpha) * u0 + alpha * u1   # m/s
    v_ms = (1 - alpha) * v0 + alpha * v1   # m/s
    if np.any(np.isnan(u_ms)) or np.any(np.isnan(v_ms)):
        warnings.warn(
            "NaN détectés dans les vents ERA5 interpolés ; remplacement par 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        u_ms = np.nan_to_num(u_ms, nan=0.0)
        v_ms = np.nan_to_num(v_ms, nan=0.0)

    # ── Conversion m/s → degrés/heure ─────────────────────────────────
    # 1° latitude  ≈ 111 km  (constant)
    # 1° longitude ≈ 111 km × cos(lat)
    cos_lat = np.cos(np.radians(lats))
    cos_lat = np.maximum(cos_lat, 0.1)     # sécurité pôles

    u_deg = u_ms * 3600.0 / (111_000.0 * cos_lat)  # deg/h longitude
    v_deg = v_ms * 3600.0 / 111_000.0               # deg/h latitude

    # ── Bruit turbulent ───────────────────────────────────────────────
    # Proportionnel à la vitesse locale du vent
    if turbulence > 0:
        random_source = rng if rng is not None else np.random
        wind_mag = np.sqrt(u_deg**2 + v_deg**2)
        sigma = turbulence * np.maximum(wind_mag, 0.005)
        u_deg += random_source.normal(0, sigma, n)
        v_deg += random_source.normal(0, sigma, n)

    return u_deg, v_deg
