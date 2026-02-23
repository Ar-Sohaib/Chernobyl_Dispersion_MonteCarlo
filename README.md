# Chernobyl Dispersion — Monte Carlo Simulation

Modèle lagrangien particulaire simulant la dispersion atmosphérique du nuage radioactif de Tchernobyl (26 avril – 16 mai 1986) sur l'Eurasie. La simulation utilise des méthodes Monte Carlo avec les données de vent réelles ERA5 (ECMWF) pour produire des cartes de probabilité et des visualisations animées.

> **Version** : 1.0 · **Date** : Février 2026 · **Licence** : MIT

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Modèle mathématique et physique](#modèle-mathématique-et-physique)
4. [Prérequis et installation](#prérequis-et-installation)
5. [Données ERA5](#données-era5)
6. [Utilisation (CLI)](#utilisation-cli)
7. [Sorties générées](#sorties-générées)
8. [Configuration](#configuration)
9. [Tests](#tests)

---

## Vue d'ensemble

Le modèle suit des milliers de particules virtuelles émises depuis le Réacteur n°4 (51.39°N, 30.10°E) et advectées par les champs de vent sur une période de 20 jours. La diffusion turbulente est modélisée par une marche aléatoire gaussienne. Plusieurs runs Monte Carlo sont agrégés pour calculer des distributions de probabilité spatiales.

**Fonctionnalités principales :**
- Suivi lagrangien de particules avec émission progressive configurable
- Champs de vent ERA5 réels à 850 hPa (0.25°, horaires) de l'ECMWF
- Modèle de vent simplifié de repli basé sur les phases historiques (5 phases)
- Cartes de probabilité, de seuil, de densité cumulée et de nuage instantané
- Vidéo heatmap animée (H.264, jusqu'en 4K 120fps)
- Projection plate (Lambert/PlateCarrée) et sphérique (Orthographique)
- CLI avec options résolution, nombre de particules, mode de rendu

---

## Structure du projet

```
config.py               Configuration (source, vent, grille, simulation, rendu)
engine.py               Moteur Monte Carlo lagrangien (advection + diffusion)
wind.py                 Modèle de vent simplifié (5 phases historiques uniformes)
wind_era5.py            Interpolation vent ERA5 spatiotemporelle (bilinéaire)
aggregation.py          Agrégation spatiale (densité, probabilité, seuil)
visualization.py        Rendu 2D (cartes uniques, vidéo 4 panneaux, vidéo proba)
visualization_globe.py  Rendu sphérique (projection orthographique Cartopy)
main.py                 Point d'entrée CLI (argparse)
download_era5.py        Téléchargement ERA5 depuis Copernicus CDS
chart.md                Contexte théorique et méthodologie détaillée
assets/                 Image de fond (Hypsometric Natural Earth)
data/                   Fichiers NetCDF ERA5 (.nc)
output/                 Sorties générées (*.png, *.mp4)
tests/                  Tests unitaires (pytest)
```

---

## Modèle mathématique et physique

### 1. Équation d'advection–diffusion (approche lagrangienne)

La dispersion atmosphérique est régie par l'**équation d'advection–diffusion** :

$$\frac{\partial C}{\partial t} + \vec{u} \cdot \nabla C = \nabla \cdot (K \nabla C) + S$$

où $C$ est la concentration, $\vec{u}$ le vecteur vent, $K$ le tenseur de diffusivité turbulente et $S$ le terme source.

Dans l'approche **lagrangienne particulaire**, cette EDP est résolue stochastiquement en intégrant individuellement les trajectoires :

$$\vec{x}(t + \Delta t) = \vec{x}(t) + \underbrace{\vec{u}(\vec{x},\,t)\,\Delta t}_{\text{advection}} + \underbrace{\boldsymbol{\xi}(t)}_{\text{diffusion turbulente}}$$

avec :
- $\vec{x}(t) = (\lambda, \phi)$ — position géographique (longitude, latitude) en degrés
- $\vec{u}(\vec{x}, t) = (u, v)$ — vent local interpolé en degrés/heure
- $\boldsymbol{\xi} \sim \mathcal{N}\!\left(0,\,\sqrt{2K\,\Delta t}\right)$ — bruit gaussien de diffusion turbulente

### 2. Discrétisation temporelle (schéma d'Euler explicite)

Le moteur (`engine.py`) intègre les trajectoires par le **schéma d'Euler explicite** à chaque pas $\Delta t = 1\,\text{h}$ :

$$\lambda_{k+1} = \lambda_k + u(\lambda_k, \phi_k, t_k)\,\Delta t + \xi_\lambda$$

$$\phi_{k+1} = \phi_k + v(\lambda_k, \phi_k, t_k)\,\Delta t + \xi_\phi$$

avec les bruits indépendants :

$$\xi_\lambda \sim \mathcal{N}(0,\;\sigma_\lambda), \quad \sigma_\lambda = \sqrt{2\,K_{\text{lon}}\,\Delta t}$$

$$\xi_\phi \sim \mathcal{N}(0,\;\sigma_\phi), \quad \sigma_\phi = \sqrt{2\,K_{\text{lat}}\,\Delta t}$$

Valeurs par défaut : $K_{\text{lon}} = 0.008\,\text{deg}^2/\text{h}$, $K_{\text{lat}} = 0.006\,\text{deg}^2/\text{h}$.

### 3. Conversion des vents ERA5 (m/s → deg/h)

Les données ERA5 fournissent les composantes $(u, v)$ en **m/s**. La conversion en degrés/heure tient compte de la courbure terrestre :

$$u_{\deg} = \frac{u_{\text{m/s}} \times 3600}{111\,000 \times \cos(\phi)}$$

$$v_{\deg} = \frac{v_{\text{m/s}} \times 3600}{111\,000}$$

où $111\,000\,\text{m/deg}$ est la longueur d'un degré de latitude, et $\cos(\phi)$ corrige la convergence des méridiens. Une borne inférieure $\cos(\phi) \geq 0.1$ évite les singularités polaires.

### 4. Interpolation bilinéaire ERA5

À chaque pas de temps, le vent est interpolé à la position exacte de chaque particule par **interpolation bilinéaire spatiale** (via `scipy.interpolate.RegularGridInterpolator`) et **linéaire temporelle** entre les deux instants ERA5 encadrants $t_k$ et $t_{k+1}$ :

$$\vec{u}(\vec{x}, t) = (1-\alpha)\,\vec{u}(\vec{x}, t_k) + \alpha\,\vec{u}(\vec{x}, t_{k+1}), \quad \alpha = \frac{t - t_k}{t_{k+1} - t_k}$$

### 5. Turbulence sous-maille (mode ERA5)

En mode ERA5, la turbulence sous-maille est ajoutée proportionnellement à la vitesse locale du vent :

$$\sigma_{\text{turb}} = \tau \cdot \|\vec{u}_{\deg}\|, \quad \tau = 0.25$$

$$u_{\deg} \mathrel{+}= \mathcal{N}(0,\,\sigma_{\text{turb}}), \quad v_{\deg} \mathrel{+}= \mathcal{N}(0,\,\sigma_{\text{turb}})$$

En mode ERA5, la diffusion explicite ($\boldsymbol{\xi}$ du §2) est désactivée (`apply_diffusion: False`) pour éviter le double comptage.

### 6. Modèle de vent simplifié (5 phases historiques)

Lorsque `WIND["mode"] = "simplified"`, le vent est **uniforme spatialement** mais évolue par phases discrètes avec transitions douces sur 6 heures :

| Phase | Période | Direction dominante |
|-------|---------|---------------------|
| 1 | 26–27 avril (0–48 h) | N-NW → Biélorussie, Scandinavie |
| 2 | 28–29 avril (48–96 h) | Ouest → Pologne, Baltique |
| 3 | 30 avril–2 mai (96–168 h) | S-SW → Ukraine, Roumanie, Turquie |
| 4 | 3–5 mai (168–264 h) | Ouest → Europe centrale, France |
| 5 | 6–16 mai (264–480 h) | Variable, vents faibles |

### 7. Émission progressive de la source

Les $N$ particules ne sont pas toutes libérées à $t=0$ : elles sont émises de façon progressive sur une durée $T_{\text{émission}} = 240\,\text{h}$ (10 jours), répartie uniformément en $\lfloor T_{\text{émission}} / \Delta t \rfloor$ pas, avec une légère dispersion initiale autour de la source :

$$\lambda_0^{(i)} \sim \mathcal{N}(\lambda_{\text{src}},\;0.05°), \quad \phi_0^{(i)} \sim \mathcal{N}(\phi_{\text{src}},\;0.03°)$$

### 8. Agrégation Monte Carlo

Pour $M$ runs indépendants, chaque run produit une trajectoire $(N \times T_{\text{steps}})$. L'agrégation spatiale sur une grille $(n_{\text{lon}} \times n_{\text{lat}})$ produit trois types de cartes :

**Densité spatiale cumulée** (normalisée) :

$$D(i,j) = \frac{1}{\max(\cdot)} \sum_{r=1}^{M} \sum_{t=0}^{T} \sum_{p=1}^{N} \mathbf{1}\!\left[(\lambda_t^{(r,p)}, \phi_t^{(r,p)}) \in \text{cell}_{i,j}\right] \cdot a_t^{(r,p)}$$

**Carte de probabilité de présence** :

$$P(i,j) = \frac{1}{M} \sum_{r=1}^{M} \mathbf{1}\!\left[\exists\,t,p : (\lambda_t^{(r,p)}, \phi_t^{(r,p)}) \in \text{cell}_{i,j} \land a_t^{(r,p)}=1 \right]$$

**Carte de dépassement de seuil** (fraction des runs dépassant $C_s$) :

$$P_s(i,j) = \frac{1}{M}\sum_{r=1}^{M} \mathbf{1}\!\left[ \bar{C}^{(r)}(i,j) > C_s \right]$$

---

## Prérequis et installation

**Dépendances Python** (voir `requirements.txt`) :

| Package | Version | Rôle |
|---------|---------|------|
| `numpy` | ≥1.26 | Calcul vectoriel, RNG |
| `matplotlib` | ≥3.8 | Rendu des cartes et vidéos |
| `cartopy` | ≥0.22 | Projections géographiques |
| `scipy` | ≥1.11 | Interpolation bilinéaire ERA5 |
| `xarray` | ≥2024.1 | Lecture fichiers NetCDF |
| `netCDF4` | ≥1.6 | Backend NetCDF |
| `cdsapi` | ≥0.7 | Téléchargement ERA5 (optionnel) |
| `pytest` | ≥8 | Tests unitaires |

```bash
python -m venv .venv
source .venv/bin/activate       # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

**FFmpeg** (nécessaire pour l'export vidéo) :

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## Données ERA5

La simulation utilise les réanalyses ERA5 de l'ECMWF (composantes $u$ et $v$ du vent à 850 hPa, résolution 0.25°, pas horaire). Pour télécharger :

1. Créer un compte sur https://cds.climate.copernicus.eu
2. Récupérer votre *Personal Access Token* depuis votre profil
3. Créer `~/.cdsapirc` :
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <VOTRE_TOKEN>
   ```
4. Lancer le téléchargement :
   ```bash
   python download_era5.py
   ```

Cela télécharge ~27 Mo de données horaires couvrant le domaine (-12°O à 75°E, 33°N à 72°N) pour le 26 avril – 16 mai 1986.

Pour utiliser le modèle de vent simplifié à la place, définir `WIND["mode"] = "simplified"` dans `config.py`.

---

## Utilisation (CLI)

```bash
# Carte de probabilité (défaut)
python main.py

# Vidéo heatmap de probabilité animée
python main.py -m pv

# Vidéo 4 panneaux (nuage, densité, probabilité, seuil)
python main.py -m all

# Modes de rendu spécifiques
python main.py -m instant       # Nuage instantané
python main.py -m cumul         # Densité cumulée
python main.py -m threshold     # Dépassement de seuil
python main.py -m proba         # Carte de probabilité statique

# Options
python main.py -m pv -r 4k            # Résolution 4K
python main.py -m pv -r 1080p         # Résolution 1080p
python main.py -m pv -n 20000         # 20 000 particules
python main.py --help                  # Aide complète
```

### Options CLI

| Option | Description | Défaut |
|--------|-------------|--------|
| `-m, --mode` | Mode de rendu : `pv`, `all`, `proba`, `instant`, `cumul`, `threshold` | `probability` |
| `-r, --resolution` | Résolution de sortie : `720p`, `1080p`, `2k`, `4k` | `2k` |
| `-n, --particles` | Nombre de particules simulées | `8000` |

---

## Sorties générées

Les fichiers sont sauvegardés dans `output/` :

| Fichier | Description |
|---------|-------------|
| `probability_map.png` | Carte statique de probabilité |
| `chernobyl_proba_video.mp4` | Heatmap de probabilité animée |
| `chernobyl_simulation.mp4` | Vidéo 4 panneaux animée |

---

## Configuration

Tous les paramètres sont centralisés dans `config.py` :

| Bloc | Paramètres clés |
|------|----------------|
| `SOURCE` | Position de la source (lon/lat), débit d'émission, durée |
| `WIND_PHASES` | 5 phases de vent historiques (u, v, σ, plages horaires) |
| `WIND` | Mode vent (`era5` / `simplified`), fichier NetCDF, turbulence |
| `DIFFUSION` | Coefficients $K_{\text{lon}}$, $K_{\text{lat}}$ en deg²/h |
| `GRID` | Emprise géographique et résolution de la grille |
| `SIMULATION` | Nombre de particules, $\Delta t$, $T_{\text{steps}}$, $M$ runs, graine |
| `VISUALIZATION` | DPI, résolution, colormaps, durée vidéo, bitrate |

---

## Tests

Les tests unitaires couvrent les modules principaux :

```bash
pytest tests/ -v
```

| Fichier de test | Module testé |
|----------------|--------------|
| `tests/test_config.py` | Cohérence des paramètres `config.py` |
| `tests/test_engine.py` | Moteur Monte Carlo, formes des tableaux |
| `tests/test_wind.py` | Modèle de vent simplifié |
| `tests/test_aggregation.py` | Agrégation spatiale, grilles |
| `tests/test_visualization.py` | Fonctions de rendu |

---

## Licence

MIT
