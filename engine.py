"""
Moteur Monte Carlo — Simulation lagrangienne en coordonnées géographiques.
Advection par le vent + diffusion turbulente sur la grille lat/lon.
Gère l'émission continue de la source pendant la durée d'émission.
"""
import numpy as np
from config import SOURCE, DIFFUSION, SIMULATION, GRID, WIND

# ── Sélection du modèle de vent ───────────────────────────────────────
try:
    from config import WIND as _WIND_CFG
    _wind_mode = _WIND_CFG.get("mode", "simplified")
except ImportError:
    _wind_mode = "simplified"

if _wind_mode == "era5":
    from wind_era5 import get_wind
    print(f"  Vent : ERA5 reanalyse (850 hPa)")
else:
    from wind import get_wind
    print(f"  Vent : modele simplifie (phases historiques)")


def _is_global_lon_grid():
    """Retourne True si la grille longitude couvre le globe complet."""
    span = GRID["lon_max"] - GRID["lon_min"]
    return GRID["lon_min"] <= -180.0 and GRID["lon_max"] >= 180.0 and span >= 359.0


def _wrap_lon_to_grid(lon):
    """Rabats les longitudes dans l'intervalle [lon_min, lon_max)."""
    span = GRID["lon_max"] - GRID["lon_min"]
    return ((lon - GRID["lon_min"]) % span) + GRID["lon_min"]


def run_single_simulation(seed=None):
    """
    Exécute une simulation Monte Carlo complète.

    Returns
    -------
    traj_lon : ndarray (n_steps+1, n_particles)
    traj_lat : ndarray (n_steps+1, n_particles)
    active   : ndarray (n_steps+1, n_particles) — bool
    """
    rng = np.random.default_rng(seed)

    n_particles = SIMULATION["n_particles"]
    dt = SIMULATION["dt"]
    n_steps = SIMULATION["n_steps"]

    Klon = DIFFUSION["Klon"]
    Klat = DIFFUSION["Klat"]
    apply_diffusion = WIND.get("apply_diffusion", True)

    lon0 = SOURCE["lon"]
    lat0 = SOURCE["lat"]
    emission_dur = SOURCE.get("emission_duration_h")

    # Écart-type de diffusion par pas de temps (en degrés)
    sigma_lon = np.sqrt(2 * Klon * dt)
    sigma_lat = np.sqrt(2 * Klat * dt)

    # Initialisation
    traj_lon = np.zeros((n_steps + 1, n_particles))
    traj_lat = np.zeros((n_steps + 1, n_particles))
    active = np.zeros((n_steps + 1, n_particles), dtype=bool)

    # Émission progressive : les particules sont libérées au fil du temps
    if emission_dur is not None and emission_dur > 0:
        emission_steps = min(int(emission_dur / dt), n_steps)
    else:
        emission_steps = 1  # Tout d'un coup

    # Répartition exacte des émissions (évite de "perdre" des particules)
    release_schedule = np.full(emission_steps, n_particles // emission_steps, dtype=int)
    release_schedule[: n_particles % emission_steps] += 1

    # Position initiale de toutes les particules (source, petite dispersion)
    traj_lon[0] = lon0 + rng.normal(0, 0.05, n_particles)
    traj_lat[0] = lat0 + rng.normal(0, 0.03, n_particles)

    # Activation progressive
    released = 0
    for step in range(n_steps):
        t_hours = step * dt

        # Libérer de nouvelles particules
        if step < emission_steps:
            new_end = min(released + int(release_schedule[step]), n_particles)
            active[step, released:new_end] = True
            released = new_end

        alive = active[step]
        n_alive = np.sum(alive)

        if n_alive == 0:
            active[step + 1] = active[step]
            traj_lon[step + 1] = traj_lon[step]
            traj_lat[step + 1] = traj_lat[step]
            continue

        # Récupérer le vent pour cet instant (ERA5 ou simplifié)
        if _wind_mode == "era5":
            u, v = get_wind(
                traj_lon[step, alive],
                traj_lat[step, alive],
                t_hours,
                era5_path=WIND.get("era5_file", "data/era5_chernobyl_1986.nc"),
                turbulence=WIND.get("turbulence", 0.25),
                rng=rng,
            )
        else:
            u, v = get_wind(
                traj_lon[step, alive],
                traj_lat[step, alive],
                t_hours,
                rng=rng,
            )

        # Bruit de diffusion
        if apply_diffusion:
            noise_lon = rng.normal(0, sigma_lon, n_alive)
            noise_lat = rng.normal(0, sigma_lat, n_alive)
        else:
            noise_lon = 0.0
            noise_lat = 0.0

        # Advection + diffusion
        traj_lon[step + 1, alive] = traj_lon[step, alive] + u * dt + noise_lon
        traj_lat[step + 1, alive] = traj_lat[step, alive] + v * dt + noise_lat

        # Domaine global : longitude périodique
        if _is_global_lon_grid():
            traj_lon[step + 1, alive] = _wrap_lon_to_grid(traj_lon[step + 1, alive])
            # Evite les singularités numériques aux pôles exacts
            traj_lat[step + 1, alive] = np.clip(
                traj_lat[step + 1, alive],
                GRID["lat_min"] + 1e-6,
                GRID["lat_max"] - 1e-6,
            )

        # Particules inactives
        traj_lon[step + 1, ~alive] = traj_lon[step, ~alive]
        traj_lat[step + 1, ~alive] = traj_lat[step, ~alive]

        # Domaine global : pas de "mort" par frontière cartographique
        if _is_global_lon_grid() and GRID["lat_min"] <= -90.0 and GRID["lat_max"] >= 90.0:
            active[step + 1] = alive.copy()
        else:
            # Domaine régional : désactiver hors emprise
            out_of_bounds = (
                (traj_lon[step + 1] < GRID["lon_min"]) |
                (traj_lon[step + 1] > GRID["lon_max"]) |
                (traj_lat[step + 1] < GRID["lat_min"]) |
                (traj_lat[step + 1] > GRID["lat_max"])
            )
            active[step + 1] = alive.copy() & ~out_of_bounds

    return traj_lon, traj_lat, active


def run_monte_carlo():
    """
    Exécute N runs Monte Carlo.

    Returns
    -------
    all_runs : list of (traj_lon, traj_lat, active)
    """
    n_runs = SIMULATION["n_runs"]
    base_seed = SIMULATION["seed"]

    all_runs = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx if base_seed is not None else None
        print(f"  Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)
        traj_lon, traj_lat, active = run_single_simulation(seed=seed)
        all_runs.append((traj_lon, traj_lat, active))
        print("OK")

    return all_runs
