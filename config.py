"""
Configuration — Simulation Tchernobyl (26 avril 1986).
Modélisation de la propagation du nuage radioactif sur l'Eurasie.
Coordonnées en degrés (longitude, latitude).
"""

# ─── Source : Réacteur n°4 de Tchernobyl ────────────────────────────
SOURCE = {
    "lon": 30.0996,       # Longitude de la centrale (°E)
    "lat": 51.3917,       # Latitude de la centrale (°N)
    "name": "Tchernobyl",
    "emission_rate": 200,  # Particules émises par pas de temps
    "emission_duration_h": 240,  # Émission intense ~10 jours
    "half_life_h": None,         # Pas de décroissance dans la visu
}

# ─── Vent historique simplifié (phases post-accident) ───────────────
# Les vents ont tourné sur ~14 jours, créant le schéma de contamination
# caractéristique touchant la Scandinavie, puis l'Europe de l'Est/Ouest.
#
# Chaque phase : (heure_début, heure_fin, u_deg/h, v_deg/h, u_std, v_std)
#   u = composante longitude (positif = vers l'est)
#   v = composante latitude  (positif = vers le nord)
#
# Conversion : ~30 km/h ≈ 0.27 °/h en latitude, variable en longitude
WIND_PHASES = [
    # Phase 1 (26-27 avril) : vent vers le N-NW → Biélorussie, Scandinavie
    {"t_start": 0,   "t_end": 48,  "u": -0.04, "v": 0.18,
     "u_std": 0.06, "v_std": 0.04},

    # Phase 2 (28-29 avril) : rotation, vent vers l'Ouest → Pologne, Baltique
    {"t_start": 48,  "t_end": 96,  "u": -0.15, "v": 0.06,
     "u_std": 0.05, "v_std": 0.05},

    # Phase 3 (30 avril - 2 mai) : vent vers le S-SW → Ukraine, Roumanie, Turquie
    {"t_start": 96,  "t_end": 168, "u": -0.06, "v": -0.12,
     "u_std": 0.06, "v_std": 0.05},

    # Phase 4 (3-5 mai) : vent vers l'Ouest → Europe centrale, Allemagne, France
    {"t_start": 168, "t_end": 264, "u": -0.18, "v": 0.02,
     "u_std": 0.07, "v_std": 0.06},

    # Phase 5 (6-10 mai) : dispersion large, vents faibles et variables
    {"t_start": 264, "t_end": 480, "u": -0.05, "v": 0.03,
     "u_std": 0.10, "v_std": 0.08},
]

# ─── Mode vent ──────────────────────────────────────────────────────
# "simplified" : modèle historique simplifié (WIND_PHASES ci-dessus)
# "era5"       : données réelles ERA5 ECMWF à 850 hPa
#                → lancer d'abord : python download_era5.py
WIND = {
    "mode": "era5",
    "era5_file": "data/era5_chernobyl_1986.nc",
    "turbulence": 0.25,   # bruit turbulent (fraction du vent local)
    # Évite la double turbulence en mode ERA5 (bruit vent + diffusion explicite)
    "apply_diffusion": False,
}

# ─── Diffusion turbulente (en degrés²/h) ───────────────────────────
DIFFUSION = {
    "Klon": 0.008,   # Diffusion en longitude
    "Klat": 0.006,   # Diffusion en latitude
}

# ─── Grille géographique — Eurasie ─────────────────────────────────
GRID = {
    "lon_min": -12.0,    # Ouest : Irlande / Portugal
    "lon_max": 75.0,     # Est  : Oural / Asie centrale
    "lat_min": 33.0,     # Sud  : Afrique du Nord / Turquie
    "lat_max": 72.0,     # Nord : Scandinavie / Arctique
    "nlon": 350,         # Résolution longitude
    "nlat": 200,         # Résolution latitude
}

# ─── Paramètres de simulation ───────────────────────────────────────
SIMULATION = {
    "n_particles": 8000,     # Nombre total de particules
    "dt": 1.0,               # Pas de temps en heures
    "n_steps": 480,           # 20 jours
    "n_runs": 12,             # Répétitions Monte Carlo
    "seed": 1986,             # Graine (année de l'accident)
}

# ─── Visualisation ──────────────────────────────────────────────────
VISUALIZATION = {
    # Source de vérité résolution/rendu:
    #   taille finale (px) = resolution
    #   figsize = (resolution[0] / dpi, resolution[1] / dpi)
    "resolution": (2560, 1440),   # 2K QHD par défaut (CLI peut surcharger)
    "dpi": 120,
    # Compat legacy (mis à jour dynamiquement par visualization.py)
    "figsize": (2560 / 120, 1440 / 120),
    "cmap_cloud": "YlOrRd",
    "cmap_cumulative": "inferno",
    "save_dir": "output",
    "video_fps": 120,
    "video_duration_s": 20,   # Durée cible de la vidéo (~20 sec)
    "video_bitrate": 30000,   # 30 Mbps pour qualité 4K
    "bg_image": "assets/europe_hypsometric.jpg",  # Fond de carte raster
    "scale_bar_km": 500,

    # Mode de rendu :
    #   "all"         → vidéo 4 panneaux (par défaut)
    #   "instant"     → image unique du nuage instantané
    #   "cumulative"  → image unique densité cumulée
    #   "probability" → image unique carte de probabilité (grand format)
    #   "threshold"   → image unique carte de dépassement de seuil
    "render_mode": "probability",
}
