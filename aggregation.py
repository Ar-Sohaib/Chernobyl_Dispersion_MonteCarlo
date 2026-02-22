"""
Module d'agrégation spatiale — Grille géographique (lon/lat).
Calcule les cartes de densité, probabilité de présence,
concentration moyenne et dépassement de seuil.
"""
import numpy as np
from config import GRID, SIMULATION


def make_grid():
    """Crée les bords et centres de la grille géographique."""
    lon_edges = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)
    lat_edges = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    return lon_edges, lat_edges, lon_centers, lat_centers


def compute_time_density_maps(traj_lon, traj_lat, active):
    """
    Carte de densité pour chaque pas de temps.

    Returns
    -------
    maps : ndarray (n_steps+1, nlat, nlon)
    """
    lon_edges, lat_edges, _, _ = make_grid()
    n_steps = traj_lon.shape[0]
    nlon = GRID["nlon"]
    nlat = GRID["nlat"]

    maps = np.zeros((n_steps, nlat, nlon))

    for t in range(n_steps):
        alive = active[t]
        if np.sum(alive) == 0:
            continue
        h, _, _ = np.histogram2d(
            traj_lon[t, alive], traj_lat[t, alive],
            bins=[lon_edges, lat_edges]
        )
        maps[t] = h.T  # (nlat, nlon)

    return maps


def compute_density_map(traj_lon, traj_lat, active):
    """
    Densité spatiale cumulée (toutes positions actives).

    Returns
    -------
    density : ndarray (nlat, nlon) — normalisé 0-1
    """
    lon_edges, lat_edges, _, _ = make_grid()
    lon_all = traj_lon[active]
    lat_all = traj_lat[active]
    h, _, _ = np.histogram2d(lon_all, lat_all, bins=[lon_edges, lat_edges])
    density = h.T
    if density.max() > 0:
        density = density / density.max()
    return density


def compute_probability_map(all_runs):
    """
    Probabilité de présence P(lon, lat) :
    fraction des runs où chaque cellule a été visitée.
    """
    lon_edges, lat_edges, _, _ = make_grid()
    nlon = GRID["nlon"]
    nlat = GRID["nlat"]
    n_runs = len(all_runs)

    visit_count = np.zeros((nlat, nlon))

    for traj_lon, traj_lat, active in all_runs:
        lon_all = traj_lon[active]
        lat_all = traj_lat[active]
        h, _, _ = np.histogram2d(lon_all, lat_all, bins=[lon_edges, lat_edges])
        visit_count += (h.T > 0).astype(float)

    prob_map = visit_count / n_runs
    return prob_map


def compute_mean_concentration(all_runs):
    """
    Concentration moyenne normalisée sur tous les runs.
    Représente la « dose intégrée » moyenne par cellule.

    Returns
    -------
    mean_conc : ndarray (nlat, nlon)
    """
    lon_edges, lat_edges, _, _ = make_grid()
    nlon = GRID["nlon"]
    nlat = GRID["nlat"]
    n_runs = len(all_runs)

    total = np.zeros((nlat, nlon))

    for traj_lon, traj_lat, active in all_runs:
        lon_all = traj_lon[active]
        lat_all = traj_lat[active]
        h, _, _ = np.histogram2d(lon_all, lat_all, bins=[lon_edges, lat_edges])
        total += h.T

    mean_conc = total / n_runs
    if mean_conc.max() > 0:
        mean_conc = mean_conc / mean_conc.max()
    return mean_conc


def compute_threshold_map(all_runs, threshold=0.05):
    """
    Probabilité de dépassement P(C > seuil).
    La concentration par run est normalisée puis comparée au seuil.

    Returns
    -------
    threshold_map : ndarray (nlat, nlon)
    """
    lon_edges, lat_edges, _, _ = make_grid()
    nlon = GRID["nlon"]
    nlat = GRID["nlat"]
    n_runs = len(all_runs)

    exceed_count = np.zeros((nlat, nlon))

    for traj_lon, traj_lat, active in all_runs:
        lon_all = traj_lon[active]
        lat_all = traj_lat[active]
        h, _, _ = np.histogram2d(lon_all, lat_all, bins=[lon_edges, lat_edges])
        density = h.T
        if density.max() > 0:
            density = density / density.max()
        exceed_count += (density > threshold).astype(float)

    threshold_map = exceed_count / n_runs
    return threshold_map


def compute_time_probability_maps(all_runs):
    """
    Carte de probabilité de présence à chaque instant t.
    Pour chaque pas de temps, on calcule la fraction des runs MC
    où chaque cellule de la grille contient ≥ 1 particule.

    Returns
    -------
    prob_maps : ndarray (n_steps+1, nlat, nlon) — valeurs 0..1
    """
    lon_edges, lat_edges, _, _ = make_grid()
    nlon = GRID["nlon"]
    nlat = GRID["nlat"]
    n_runs = len(all_runs)

    # Déterminer n_steps depuis le premier run
    n_steps = all_runs[0][0].shape[0]  # traj_lon.shape[0]

    prob_maps = np.zeros((n_steps, nlat, nlon), dtype=np.float32)

    for run_idx, (traj_lon, traj_lat, active) in enumerate(all_runs):
        print(f"    Probabilité temporelle : run {run_idx+1}/{n_runs}...", end="\r")
        for t in range(n_steps):
            alive = active[t]
            if np.sum(alive) == 0:
                continue
            h, _, _ = np.histogram2d(
                traj_lon[t, alive], traj_lat[t, alive],
                bins=[lon_edges, lat_edges]
            )
            prob_maps[t] += (h.T > 0).astype(np.float32)

    prob_maps /= n_runs
    print(f"    Probabilité temporelle : {n_runs}/{n_runs} runs — OK     ")
    return prob_maps
