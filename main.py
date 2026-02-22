"""
Script principal — Simulation Tchernobyl.
Propagation du nuage radioactif sur la carte d'Eurasie.
Vidéo MP4 avec 4 panneaux sur fond satellite.
"""
import time
import argparse
import numpy as np

from config import SIMULATION, VISUALIZATION
from engine import run_monte_carlo
from aggregation import (
    make_grid,
    compute_time_density_maps,
    compute_density_map,
    compute_probability_map,
    compute_mean_concentration,
    compute_threshold_map,
    compute_time_probability_maps,
)
from visualization import create_video, create_single_map, create_proba_video


THRESHOLD = 0.05  # Seuil de concentration pour la carte de dépassement

# Alias courts → noms internes
_MODE_ALIASES = {
    "all":         "all",
    "video":       "all",
    "4":           "all",
    "proba":       "probability",
    "probability": "probability",
    "prob":        "probability",
    "proba_video": "proba_video",
    "probavideo":  "proba_video",
    "pv":          "proba_video",
    "heatmap":     "proba_video",
    "cumul":       "cumulative",
    "cumulative":  "cumulative",
    "instant":     "instant",
    "cloud":       "instant",
    "threshold":   "threshold",
    "seuil":       "threshold",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Simulation Monte Carlo -- Tchernobyl 1986",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default=None,
        metavar="MODE",
        help=(
            "Mode de rendu (par défaut : config.py).\n"
            "  all / video / 4    → vidéo 4 panneaux\n"
            "  proba / prob       → image carte de probabilité\n"
            "  proba_video / pv   → vidéo heatmap probabiliste\n"
            "  cumul / cumulative → densité cumulée\n"
            "  instant / cloud    → nuage instantané\n"
            "  seuil / threshold  → dépassement de seuil"
        ),
    )
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        default=None,
        metavar="RES",
        help=(
            "Résolution de sortie (par défaut : 2k).\n"
            "  720p  / hd         →  1280×720\n"
            "  1080p / fullhd     →  1920×1080\n"
            "  2k    / qhd        →  2560×1440\n"
            "  4k    / uhd        →  3840×2160"
        ),
    )
    parser.add_argument(
        "--particles", "-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Nombre de particules à simuler (par défaut : 8000).\n"
            "  Exemples : 1000, 5000, 10000, 50000"
        ),
    )
    return parser.parse_args()


# Résolutions prédéfinies : (largeur_px, hauteur_px)
_RESOLUTIONS = {
    "720p":   (1280,  720),
    "hd":     (1280,  720),
    "1080p":  (1920, 1080),
    "fullhd": (1920, 1080),
    "fhd":    (1920, 1080),
    "2k":     (2560, 1440),
    "qhd":    (2560, 1440),
    "1440p":  (2560, 1440),
    "4k":     (3840, 2160),
    "uhd":    (3840, 2160),
    "2160p":  (3840, 2160),
}


def _apply_resolution(res_key):
    """Met à jour VISUALIZATION avec la résolution choisie."""
    key = res_key.lower().strip()
    if key not in _RESOLUTIONS:
        raise SystemExit(
            f"ERREUR: Resolution inconnue : '{res_key}'\n"
            f"   Valides : {', '.join(sorted(set(f'{v[0]}×{v[1]} ({k})' for k, v in _RESOLUTIONS.items())))}"
        )
    w, h = _RESOLUTIONS[key]
    dpi = VISUALIZATION["dpi"]
    # H.264 exige des dimensions paires
    w = w if w % 2 == 0 else w + 1
    h = h if h % 2 == 0 else h + 1
    VISUALIZATION["figsize"] = (w / dpi, h / dpi)
    return w, h


def main():
    args = _parse_args()

    # ── Résolution : CLI > config.py ──────────────────────────────────
    if args.resolution is not None:
        res_w, res_h = _apply_resolution(args.resolution)
    else:
        dpi = VISUALIZATION["dpi"]
        fw, fh = VISUALIZATION["figsize"]
        res_w, res_h = int(fw * dpi), int(fh * dpi)
        # H.264 exige des dimensions paires
        if res_w % 2 != 0:
            res_w += 1
            VISUALIZATION["figsize"] = (res_w / dpi, fh)
        if res_h % 2 != 0:
            res_h += 1
            VISUALIZATION["figsize"] = (VISUALIZATION["figsize"][0], res_h / dpi)

    # ── Nombre de particules : CLI > config.py ────────────────────────
    if args.particles is not None:
        if args.particles < 100:
            raise SystemExit("ERREUR: Minimum 100 particules.")
        SIMULATION["n_particles"] = args.particles

    # Priorité : CLI > config.py > défaut
    if args.mode is not None:
        key = args.mode.lower().strip()
        if key not in _MODE_ALIASES:
            raise SystemExit(
                f"ERREUR: Mode inconnu : '{args.mode}'\n"
                f"   Modes valides : {', '.join(sorted(set(_MODE_ALIASES.values())))}"
            )
        render_mode = _MODE_ALIASES[key]
    else:
        render_mode = VISUALIZATION.get("render_mode", "probability")

    print("=" * 65)
    print("  SIMULATION MONTE CARLO -- TCHERNOBYL 1986")
    print("=" * 65)
    print(f"  Particules   : {SIMULATION['n_particles']}")
    print(f"  Durée sim.   : {SIMULATION['n_steps']} h ({SIMULATION['n_steps']//24} jours)")
    print(f"  Runs MC      : {SIMULATION['n_runs']}")
    print(f"  Pas de temps : {SIMULATION['dt']} h")
    print(f"  Mode rendu   : {render_mode}")
    print(f"  Résolution   : {res_w}×{res_h}")
    print("-" * 65)

    t0 = time.time()

    # ── 1. Simulation Monte Carlo ─────────────────────────────────────
    print("\n[1/3] Simulation Monte Carlo...")
    all_runs = run_monte_carlo()

    # ── 2. Agrégation spatiale ────────────────────────────────────────
    print("\n[2/3] Agrégation spatiale...")

    traj_lon, traj_lat, active = all_runs[0]
    lon_edges, lat_edges, _, _ = make_grid()

    time_maps = compute_time_density_maps(traj_lon, traj_lat, active)
    print("  [OK] Cartes temporelles calculees")

    prob_map = compute_probability_map(all_runs)
    print("  [OK] Probabilite de presence calculee")

    mean_conc = compute_mean_concentration(all_runs)
    print("  [OK] Concentration moyenne calculee")

    threshold_map = compute_threshold_map(all_runs, threshold=THRESHOLD)
    print(f"  [OK] Depassement de seuil (C > {THRESHOLD}) calcule")

    # ── 3. Génération sortie ──────────────────────────────────────────
    if render_mode == "all":
        print("\n[3/3] Génération de la vidéo MP4 (4 panneaux)...")
        output_path = create_video(
            traj_lon, traj_lat, active,
            time_maps, lon_edges, lat_edges,
            prob_map, mean_conc, threshold_map, THRESHOLD,
        )
    elif render_mode == "proba_video":
        print("\n[3/3] Calcul probabilité temporelle (tous runs × tous pas)...")
        time_prob_maps = compute_time_probability_maps(all_runs)
        print("\n[3/3] Génération vidéo heatmap probabiliste...")
        output_path = create_proba_video(time_prob_maps)
    else:
        print(f"\n[3/3] Génération image unique — mode '{render_mode}'...")
        output_path = create_single_map(
            mode=render_mode,
            prob_map=prob_map,
            mean_conc=mean_conc,
            threshold_map=threshold_map,
            threshold=THRESHOLD,
            traj_lon=traj_lon,
            traj_lat=traj_lat,
            active=active,
            time_density_maps=time_maps,
        )

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"  TERMINE en {elapsed:.1f}s")
    print(f"  Sortie : {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
