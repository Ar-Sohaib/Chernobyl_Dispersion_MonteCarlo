"""
Module météorologique — Vents historiques simplifiés de Tchernobyl.
Reproduit les phases de vent qui ont transporté le nuage radioactif
à travers l'Europe entre le 26 avril et mi-mai 1986.

MODE SIMPLIFIÉ : le vent est uniforme spatialement (même valeur pour
toutes les particules) mais varie dans le temps par phases.
Pour les données réelles → voir wind_era5.py
"""
import numpy as np
from config import WIND_PHASES


def get_wind(lons, lats, t_hours, rng=None):
    """
    Retourne les composantes (u, v) du vent en degrés/heure
    pour un instant donné (en heures depuis l'accident).

    Le vent est UNIFORME : les positions lons/lats sont ignorées
    (modèle simplifié). Utilisé quand WIND["mode"] = "simplified".

    Parameters
    ----------
    lons : ndarray (n,) — longitudes (ignorées)
    lats : ndarray (n,) — latitudes (ignorées)
    t_hours : float — heures depuis l'accident

    Returns
    -------
    u, v : ndarray (n,)
        Composantes du vent en degrés/heure (lon, lat).
    """
    n_particles = len(lons)
    # Trouver la phase correspondante
    phase = WIND_PHASES[-1]  # Fallback : dernière phase
    for p in WIND_PHASES:
        if p["t_start"] <= t_hours < p["t_end"]:
            phase = p
            break

    # Interpolation douce entre phases (transition sur 6h)
    u_mean = phase["u"]
    v_mean = phase["v"]
    u_std = phase["u_std"]
    v_std = phase["v_std"]

    # Transition douce : si on est dans les 6 premières heures d'une phase,
    # interpoler avec la phase précédente
    t_in_phase = t_hours - phase["t_start"]
    if t_in_phase < 6.0 and phase["t_start"] > 0:
        # Trouver la phase précédente
        prev_phase = WIND_PHASES[0]
        for p in WIND_PHASES:
            if p["t_end"] == phase["t_start"]:
                prev_phase = p
                break
        alpha = t_in_phase / 6.0  # 0 → 1 sur 6h
        u_mean = (1 - alpha) * prev_phase["u"] + alpha * phase["u"]
        v_mean = (1 - alpha) * prev_phase["v"] + alpha * phase["v"]
        u_std = (1 - alpha) * prev_phase["u_std"] + alpha * phase["u_std"]
        v_std = (1 - alpha) * prev_phase["v_std"] + alpha * phase["v_std"]

    # Vent moyen + fluctuations turbulentes
    random_source = rng if rng is not None else np.random
    u = u_mean + random_source.normal(0, u_std, n_particles)
    v = v_mean + random_source.normal(0, v_std, n_particles)

    return u, v
