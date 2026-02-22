#!/usr/bin/env python3
"""
Téléchargement des données de vent ERA5 (réanalyses ECMWF)
pour la période Tchernobyl : 26 avril — 16 mai 1986.

Composantes u/v à 850 hPa (transport atmosphérique), horaires, Europe.

━━━ Prérequis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Créer un compte gratuit sur :
       https://cds.climate.copernicus.eu

  2. Aller dans ton profil → « Personal Access Token »
     et copier le token.

  3. Créer le fichier  ~/.cdsapirc  avec ce contenu :
       url: https://cds.climate.copernicus.eu/api
       key: <TON_TOKEN>

  4. Accepter les conditions d'utilisation de ERA5 :
       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels

  5. Lancer :  python download_era5.py
"""
import os
import sys

import cdsapi
import xarray as xr

OUTPUT_DIR = "data"
FINAL_FILE = os.path.join(OUTPUT_DIR, "era5_chernobyl_1986.nc")

# Zone géographique — doit couvrir GRID dans config.py
# Format : [North, West, South, East]
AREA = [72, -12, 33, 75]

# Toutes les heures de la journée
HOURS = [f"{h:02d}:00" for h in range(24)]


def _check_cdsapirc():
    """Vérifie que le fichier ~/.cdsapirc existe."""
    rc = os.path.expanduser("~/.cdsapirc")
    if not os.path.isfile(rc):
        print("━" * 65)
        print("  ERREUR: Fichier ~/.cdsapirc introuvable !")
        print()
        print("  Pour le créer :")
        print("    1. Crée un compte : https://cds.climate.copernicus.eu")
        print("    2. Profil → Personal Access Token")
        print("    3. Créer le fichier ~/.cdsapirc :")
        print()
        print("       url: https://cds.climate.copernicus.eu/api")
        print("       key: <TON_TOKEN>")
        print("━" * 65)
        sys.exit(1)
    print("  [OK] Fichier ~/.cdsapirc trouve")


def download():
    """Télécharge et fusionne les données ERA5 pour Tchernobyl."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Déjà téléchargé ?
    if os.path.isfile(FINAL_FILE):
        size_mb = os.path.getsize(FINAL_FILE) / (1024 * 1024)
        print(f"  [OK] Donnees deja presentes : {FINAL_FILE} ({size_mb:.0f} Mo)")
        print("    (Supprimer le fichier pour re-télécharger)")
        return FINAL_FILE

    _check_cdsapirc()

    client = cdsapi.Client()
    temp_files = []

    # ── Requête 1 : Avril 26-30, 1986 ────────────────────────────────
    f_apr = os.path.join(OUTPUT_DIR, "_era5_apr_1986.nc")
    print("\n  [DL] ERA5 -- Avril 1986 (jours 26-30) -- 850 hPa...")
    print("     Variables : u, v (vent)")
    print(f"     Zone : {AREA}")
    print("     Résolution : 0.25° × 0.25°, horaire")
    print("     Telechargement en cours (peut prendre 5-15 min)...\n")

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": ["850"],
            "year": ["1986"],
            "month": ["04"],
            "day": ["26", "27", "28", "29", "30"],
            "time": HOURS,
            "data_format": "netcdf",
            "area": AREA,
        },
        f_apr,
    )
    temp_files.append(f_apr)
    print(f"  [OK] Avril termine : {f_apr}")

    # ── Requête 2 : Mai 1-16, 1986 ───────────────────────────────────
    f_may = os.path.join(OUTPUT_DIR, "_era5_may_1986.nc")
    print("\n  [DL] ERA5 -- Mai 1986 (jours 1-16) -- 850 hPa...")
    print("     Telechargement en cours...\n")

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": ["850"],
            "year": ["1986"],
            "month": ["05"],
            "day": [f"{d:02d}" for d in range(1, 17)],
            "time": HOURS,
            "data_format": "netcdf",
            "area": AREA,
        },
        f_may,
    )
    temp_files.append(f_may)
    print(f"  [OK] Mai termine : {f_may}")

    # ── Fusion ────────────────────────────────────────────────────────
    print("\n  Fusion des fichiers...")
    datasets = [xr.open_dataset(f) for f in temp_files]

    # Chaque fichier GRIB peut avoir une dimension « time » parasite +
    # « valid_time » comme vrai axe temporel.  On normalise.
    cleaned = []
    for d in datasets:
        # Si « valid_time » et « time » coexistent, la dim « time » est
        # souvent un artefact (ex: 2 expver). On aplatit en prenant la
        # première tranche NON-NaN le long de cet axe.
        if "valid_time" in d.dims and "time" in d.dims:
            slices = []
            for i in range(d.sizes["time"]):
                slices.append(d.isel(time=i, drop=True))
            d = slices[0]
            for s in slices[1:]:
                d = d.fillna(s)
        # Renommer valid_time → time
        if "valid_time" in d.dims:
            d = d.rename({"valid_time": "time"})
        # Supprimer les coordonnées parasites
        d = d.drop_vars(["number", "expver"], errors="ignore")
        # Squeeze la dimension pression si présente
        for pname in ("level", "pressure_level"):
            if pname in d.dims:
                d = d.squeeze(pname, drop=True)
        cleaned.append(d)

    ds = xr.concat(cleaned, dim="time")
    for d in datasets:
        d.close()

    # Trier par temps
    ds = ds.sortby("time")

    # Sauvegarder
    ds.to_netcdf(FINAL_FILE)
    ds.close()

    # Nettoyage fichiers temporaires
    for f in temp_files:
        if os.path.isfile(f):
            os.remove(f)

    size_mb = os.path.getsize(FINAL_FILE) / (1024 * 1024)
    print(f"\n  [OK] ERA5 pret : {FINAL_FILE} ({size_mb:.0f} Mo)")
    print(f"     Période : 26 avril — 16 mai 1986")
    print(f"     Variables : u, v à 850 hPa (horaire)")
    print(f"     Zone : {AREA[1]}°W — {AREA[3]}°E, {AREA[2]}°N — {AREA[0]}°N")
    return FINAL_FILE


if __name__ == "__main__":
    print("=" * 65)
    print("  Telechargement ERA5 -- Vents Tchernobyl 1986")
    print("=" * 65)
    download()
