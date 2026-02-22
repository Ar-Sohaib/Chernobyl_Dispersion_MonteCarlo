# Recommandations a implementer

Liste detaillee des ameliorations classees par priorite.
Chaque item est actionnable avec fichiers concernes et description technique.

---

## PRIORITE HAUTE — Fondamentaux

### 1. Creer `requirements.txt`

- **Fichier** : `requirements.txt` (a creer)
- **Probleme** : Aucun fichier de dependances. Impossible de reproduire l'environnement.
- **Action** :
  - Generer `pip freeze > requirements.txt` ou creer manuellement avec versions minimales
  - Dependances principales : numpy, matplotlib, cartopy, scipy, xarray, netCDF4, cdsapi
  - Optionnel : ajouter un `pyproject.toml` pour un packaging moderne

---

### 2. Corriger la double turbulence

- **Fichiers** : `engine.py` (L96-97), `wind_era5.py` (L166-170), `config.py`
- **Probleme** : Le bruit est applique deux fois :
  1. `engine.py` : diffusion gaussienne via `DIFFUSION["Klon"]` et `DIFFUSION["Klat"]`
  2. `wind_era5.py` : bruit turbulent proportionnel au vent (25%)
  - Ces deux sources se cumulent sans justification physique → surestime la dispersion
- **Action** :
  - Option A : Desactiver `DIFFUSION` quand le mode ERA5 est actif (le bruit ERA5 suffit)
  - Option B : Reduire le coefficient turbulent ERA5 et garder une diffusion sous-maille reduite
  - Option C : Ajouter un flag dans `config.py` : `"apply_diffusion": True/False` selon le mode vent
  - Documenter le choix physique dans un commentaire

---

### 3. Remplacer `np.random.seed()` par `np.random.default_rng()`

- **Fichier** : `engine.py` (L37)
- **Probleme** : `np.random.seed(seed)` modifie l'etat global du generateur.
  Non thread-safe, empeche la parallelisation future des runs.
- **Action** :
  - Creer `rng = np.random.default_rng(seed)` au debut de `run_single_simulation`
  - Remplacer tous les `np.random.normal(...)` par `rng.normal(...)`
  - Passer le `rng` a `get_wind()` pour le bruit turbulent (ou le desactiver, cf. point 2)
  - Idem dans `wind_era5.py` L169 : `np.random.normal` → `rng.normal`

---

### 4. Ajouter des tests unitaires

- **Fichier** : `tests/` (repertoire a creer)
- **Probleme** : Zero test. Aucune garantie de non-regression.
- **Action** :
  - Installer pytest : `pip install pytest`
  - Creer `tests/test_engine.py` :
    - Test reproductibilite : meme seed → meme trajectoire
    - Test emission progressive : verifier le nombre de particules actives par step
    - Test hors domaine : particules au-dela de la grille → desactivees
    - Test 0 particules : pas de crash
  - Creer `tests/test_aggregation.py` :
    - Test `make_grid()` : dimensions correctes
    - Test `compute_probability_map()` : valeurs entre 0 et 1
    - Test `compute_density_map()` : normalisation correcte
    - Test avec donnees synthetiques connues → resultat attendu
  - Creer `tests/test_wind.py` :
    - Test conversion m/s → deg/h (valeur connue)
    - Test interpolation temporelle (alpha = 0 → valeur t0, alpha = 1 → valeur t1)
    - Test transition de phase dans `wind.py` (smooth blending)
  - Creer `tests/test_config.py` :
    - Test coherence des parametres (n_steps > 0, dt > 0, grille valide)

---

### 5. Ajouter la validation des donnees ERA5

- **Fichier** : `wind_era5.py` (dans `_load()` et `get_wind()`)
- **Probleme** :
  - Aucune verification que le domaine ERA5 couvre le domaine de simulation
  - Pas de detection des NaN dans les donnees interpolees
  - `fill_value=0.0` renvoie un vent nul sans avertissement si une particule est hors domaine
- **Action** :
  - Dans `_load()` : comparer `_era5["lats"]` et `_era5["lons"]` avec `GRID` et avertir si le domaine ERA5 est plus petit
  - Dans `get_wind()` : detecter les particules recevant `fill_value=0` et logger un warning
  - Ajouter un check des NaN : `np.any(np.isnan(u_ms))` → warning ou fallback

---

## PRIORITE MOYENNE — Enrichissement scientifique

### 6. Implementer le depot sec

- **Fichier** : `engine.py` (dans la boucle temporelle)
- **Probleme** : Les particules ne se deposent jamais → surestime la dispersion a longue distance
- **Action** :
  - Ajouter dans `config.py` :
    ```
    DEPOSITION = {
        "dry_velocity": 0.002,    # m/s (typique Cs-137)
        "mixing_height": 1000.0,  # m (hauteur de melange)
    }
    ```
  - A chaque pas de temps, probabilite de depot : `P_dep = v_d * dt * 3600 / H`
  - Tirage aleatoire : si `random < P_dep` → particule desactivee (deposee)
  - Stocker les positions de depot pour une carte de contamination au sol

---

### 7. Implementer la decroissance radioactive

- **Fichier** : `engine.py`, `config.py`
- **Probleme** : `half_life_h: None` dans config mais jamais utilise dans le code
- **Action** :
  - Ajouter les isotopes principaux dans config :
    ```
    SOURCE["half_life_h"] = {
        "I-131":  192.5,     # 8.02 jours
        "Cs-137": 264_720,   # ~30.2 ans (negligeable sur 20j)
    }
    ```
  - Implementer la decroissance : `P_decay = 1 - exp(-lambda * dt)` ou lambda = ln(2) / T_1/2
  - Pour simplifier : utiliser I-131 (demi-vie 8j, visible sur 20 jours de simulation)
  - Ponderer les particules au lieu de les desactiver (poids decroissant)

---

### 8. Paralleliser les runs Monte Carlo

- **Fichier** : `engine.py` (`run_monte_carlo()`)
- **Probleme** : Les 12 runs s'executent sequentiellement. Sur 8+ coeurs c'est du gaspillage.
- **Action** :
  - Prerequis : point 3 (rng non global) doit etre fait avant
  - Utiliser `multiprocessing.Pool` ou `concurrent.futures.ProcessPoolExecutor`
  - Chaque worker recoit son seed et retourne `(traj_lon, traj_lat, active)`
  - Attention a la memoire : 12 runs × 481 × 8000 × 8 bytes × 3 arrays = ~1.1 GB
  - Ajouter un flag dans config : `"parallel": True, "n_workers": 4`

---

### 9. Factoriser `visualization.py`

- **Fichier** : `visualization.py` (873 lignes)
- **Probleme** :
  - Liste des villes dupliquee 3 fois (L109-125, L247-263, via `_add_cities`)
  - `create_single_map()` re-code tout le fond de carte au lieu d'utiliser `_setup_map_ax()`
  - Constantes de style dupliquees (fontes, tailles) entre les fonctions
- **Action** :
  - Extraire `CITIES` comme constante de module
  - Refactoriser `create_single_map()` pour utiliser `_setup_map_ax()`
  - Creer un dict `STYLE_SINGLE` / `STYLE_VIDEO` pour les tailles de fonte
  - Objectif : passer de 873 lignes a ~600 lignes sans perte de fonctionnalite

---

### 10. Ajouter le depot humide (pluie)

- **Fichier** : `engine.py`, `config.py`, potentiellement `wind_era5.py`
- **Probleme** : Le lessivage par la pluie est un mecanisme critique pour Tchernobyl
  (contamination de Kiev et de la Suede liee aux precipitations)
- **Action** :
  - Telecharger les precipitations ERA5 (variable `total_precipitation`) via `download_era5.py`
  - Ajouter l'interpolation des precipitations dans `wind_era5.py`
  - Coefficient de lessivage : `Lambda = a * P^b` (P = intensite pluie mm/h, a ~ 5e-5, b ~ 0.8)
  - Probabilite de capture : `P_wet = 1 - exp(-Lambda * dt * 3600)`

---

## PRIORITE BASSE — Polish et bonnes pratiques

### 11. Ajouter le logging

- **Fichiers** : Tous les modules
- **Probleme** : Tout le feedback via `print()`. Pas de niveaux, pas de fichier log.
- **Action** :
  - `import logging` dans chaque module
  - `logger = logging.getLogger(__name__)`
  - Remplacer `print("  [OK] ...")` par `logger.info(...)`
  - Remplacer `print("ERREUR: ...")` par `logger.error(...)`
  - Configurer dans `main.py` : `logging.basicConfig(level=logging.INFO, format=...)`
  - Ajouter option CLI `--verbose` / `-v` pour passer en DEBUG

---

### 12. Ajouter les type hints

- **Fichiers** : Tous les modules Python
- **Probleme** : Aucune annotation de type. Les signatures sont vagues.
- **Action** :
  - `engine.py` :
    ```python
    def run_single_simulation(seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ```
  - `aggregation.py` :
    ```python
    def compute_probability_map(all_runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    ```
  - `wind_era5.py` :
    ```python
    def get_wind(lons: np.ndarray, lats: np.ndarray, t_hours: float, ...) -> tuple[np.ndarray, np.ndarray]:
    ```
  - Ajouter `from __future__ import annotations` pour compatibilite Python 3.9

---

### 13. Optimiser `compute_time_probability_maps()`

- **Fichier** : `aggregation.py` (L148-180)
- **Probleme** : Boucle Python O(n_runs x n_steps) avec `np.histogram2d` a chaque iteration.
  C'est le goulot d'etranglement principal.
- **Action** :
  - Vectoriser avec `np.digitize` + indexation avancee au lieu de `histogram2d` dans la boucle
  - Pre-calculer les indices de bin pour toutes les particules d'un run
  - Utiliser `np.add.at` pour l'accumulation sparse
  - Alternative : utiliser `numba.jit` sur la boucle interne

---

### 14. Mettre en place CI/CD (GitHub Actions)

- **Fichier** : `.github/workflows/ci.yml` (a creer)
- **Probleme** : Pas d'integration continue. Les regressions passent inapercues.
- **Action** :
  - Creer un workflow GitHub Actions :
    - Python 3.9 / 3.11 / 3.12
    - `pip install -r requirements.txt && pip install pytest`
    - `pytest tests/ -v`
  - Ajouter un badge dans le README
  - Optionnel : ajouter `flake8` ou `ruff` comme linter

---

### 15. Validation contre HYSPLIT / donnees historiques

- **Fichier** : `docs/validation.md` (a creer)
- **Probleme** : Aucune validation quantitative. Pas de comparaison avec un modele de reference.
- **Action** :
  - Telecharger une simulation HYSPLIT de reference pour Tchernobyl 1986
  - Comparer les contours de probabilite (P > 0.1, P > 0.5) visuellement
  - Calculer des metriques : RMSE spatial, distance de Wasserstein
  - Comparer avec les cartes IAEA de contamination au Cs-137
  - Documenter les ecarts et leurs causes (depot non modelise, etc.)

---

### 16. Ajouter un mode `--dry-run`

- **Fichier** : `main.py`
- **Probleme** : Impossible de verifier la config sans lancer toute la simulation.
- **Action** :
  - Ajouter `--dry-run` dans `_parse_args()`
  - Afficher la config, verifier les fichiers (ERA5 present, fond de carte present, ffmpeg disponible)
  - Estimer la memoire requise : `n_runs * n_steps * n_particles * 8 * 3 / 1e9` GB
  - Quitter sans simuler

---

### 17. Exporter les resultats en NetCDF / GeoTIFF

- **Fichier** : `export.py` (a creer)
- **Probleme** : Les resultats ne sont disponibles qu'en image/video. Pas de donnees reutilisables.
- **Action** :
  - Sauvegarder `prob_map`, `mean_conc`, `threshold_map` en NetCDF avec coordonnees lon/lat
  - Ajouter les metadonnees (CRS, date, parametres de simulation)
  - Optionnel : export GeoTIFF pour utilisation dans QGIS / ArcGIS
  - Ajouter option CLI `--export` / `--save-data`

---

### 18. Supporter la simulation 3D (couches verticales)

- **Fichier** : `engine.py`, `config.py`, `wind_era5.py`
- **Probleme** : Modele purement 2D. Ignore la structure verticale de l'atmosphere.
- **Action** (complexe) :
  - Telecharger ERA5 sur plusieurs niveaux de pression (1000, 925, 850, 700, 500 hPa)
  - Ajouter une coordonnee `z` (altitude ou niveau de pression) aux particules
  - Advection verticale : composante `w` du vent ERA5
  - Diffusion verticale avec coefficient Kz
  - Impact : ameliore le realisme mais multiplie la complexite et le temps de calcul

---

### 19. Mettre a jour `chart.md`

- **Fichier** : `chart.md`
- **Probleme** : Decalage entre la theorie documentee et le code reel.
  Mentionne 3D, depot, chimie qui ne sont pas implementes.
- **Action** :
  - Ajouter une section "Etat d'implementation" indiquant ce qui est fait vs. prevu
  - Supprimer ou marquer comme "extension future" les fonctionnalites non implementees
  - Ajouter les details specifiques a Tchernobyl (source, isotopes, chronologie)

---

## Recapitulatif

| #  | Tache                                  | Priorite | Effort  | Fichiers                    |
|----|----------------------------------------|----------|---------|-----------------------------|
| 1  | requirements.txt                       | Haute    | 5 min   | requirements.txt            |
| 2  | Corriger double turbulence             | Haute    | 30 min  | engine.py, wind_era5.py     |
| 3  | np.random.default_rng                  | Haute    | 20 min  | engine.py, wind_era5.py     |
| 4  | Tests unitaires                        | Haute    | 2-3h    | tests/                      |
| 5  | Validation donnees ERA5                | Haute    | 30 min  | wind_era5.py                |
| 6  | Depot sec                              | Moyenne  | 1h      | engine.py, config.py        |
| 7  | Decroissance radioactive               | Moyenne  | 1h      | engine.py, config.py        |
| 8  | Paralleliser les runs                  | Moyenne  | 1h      | engine.py                   |
| 9  | Factoriser visualization               | Moyenne  | 1-2h    | visualization.py            |
| 10 | Depot humide                           | Moyenne  | 2-3h    | engine.py, wind_era5.py     |
| 11 | Logging                                | Basse    | 1h      | Tous                        |
| 12 | Type hints                             | Basse    | 1h      | Tous                        |
| 13 | Optimiser time_probability_maps        | Basse    | 1h      | aggregation.py              |
| 14 | CI/CD GitHub Actions                   | Basse    | 30 min  | .github/workflows/          |
| 15 | Validation HYSPLIT                     | Basse    | 3-5h    | docs/                       |
| 16 | Mode --dry-run                         | Basse    | 20 min  | main.py                     |
| 17 | Export NetCDF / GeoTIFF                | Basse    | 1-2h    | export.py                   |
| 18 | Simulation 3D                          | Basse    | 5-10h   | engine.py, wind_era5.py     |
| 19 | Mettre a jour chart.md                 | Basse    | 30 min  | chart.md                    |
