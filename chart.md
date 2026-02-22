# Simulation Monte Carlo de Dispersion Atmosphérique  
## Génération d’une Probability Map Spatiale

---

## 1. Objectif du projet

L’objectif de ce projet est de développer une simulation numérique de dispersion atmosphérique basée sur une approche Monte Carlo (modèle lagrangien particulaire) afin de générer :

- Une carte de densité spatiale
- Une carte de probabilité de présence
- Une carte de probabilité de dépassement d’un seuil

Le modèle est générique et peut être appliqué à tout polluant atmosphérique (gaz ou particules fines).

---

## 2. Cadre théorique

La dispersion atmosphérique est modélisée par l’équation d’advection–diffusion :

Transport = Advection (vent moyen) + Diffusion turbulente + Dépôt + Décroissance

Dans l’approche lagrangienne :

- Le fluide (air) est décrit en champ eulérien (grille météo)
- Les particules sont suivies individuellement
- La diffusion est modélisée par une marche aléatoire gaussienne

Position à chaque pas de temps Δt :

x(t+Δt) = x(t) + u(x,t)Δt + bruit_x  
y(t+Δt) = y(t) + v(x,t)Δt + bruit_y  

où :
- u, v : composantes du vent
- bruit : tirage gaussien calibré selon un coefficient de diffusion turbulente

---

## 3. Architecture du système

### 3.1 Modules principaux

1. Configuration source
   - Position (x0, y0, z0)
   - Débit d’émission
   - Durée d’émission
   - Demi-vie (si applicable)

2. Météorologie
   - Champ de vent 2D ou 3D
   - Stabilité atmosphérique
   - Coefficients de diffusion horizontale et verticale

3. Moteur Monte Carlo
   - Génération des particules
   - Intégration temporelle
   - Gestion dépôt / disparition
   - Pondération masse / activité

4. Agrégation spatiale
   - Discrétisation en grille
   - Comptage des occurrences
   - Estimation concentration
   - Calcul probabilités

5. Visualisation
   - Heatmap 2D
   - Carte de dépassement seuil
   - Animation temporelle

---

## 4. Méthodologie Monte Carlo

### 4.1 Simulation

- N particules simulées
- Pas de temps Δt
- Horizon temporel T
- M répétitions si incertitudes météo

### 4.2 Types de cartes générées

1. Densité spatiale :
   D(x,y) = nombre total de visites normalisé

2. Probabilité de présence :
   P(x,y) = fraction des runs où la cellule a été visitée

3. Probabilité de dépassement seuil :
   P(C(x,y,t) > C_seuil)

---

## 5. Sources de données

### 5.1 Données météorologiques

1. ERA5 – ECMWF  
   Source : Copernicus Climate Data Store  
   Variables :
   - Vent (u, v, w)
   - Température
   - Pression
   - Précipitations

   https://cds.climate.copernicus.eu

2. NOAA – Global Forecast System (GFS)  
   https://www.ncei.noaa.gov  
   Résolution horaire disponible

3. Météo-France (France)  
   Données ARPEGE / AROME  
   https://donneespubliques.meteofrance.fr

---

### 5.2 Données géographiques

1. OpenStreetMap  
   Relief, bâtiments  
   https://www.openstreetmap.org

2. SRTM – NASA  
   Modèle numérique de terrain  
   https://srtm.csi.cgiar.org

3. IGN (France)  
   https://www.ign.fr

---

### 5.3 Paramètres physiques

1. Coefficients de diffusion turbulente  
   Références :
   - Pasquill-Gifford stability classes
   - Turner, D.B. (Workbook of Atmospheric Dispersion Estimates)

2. Demi-vies isotopiques (si nécessaire)  
   IAEA – International Atomic Energy Agency  
   https://www.iaea.org

---

## 6. Hypothèses simplificatrices

- Terrain plat (option simplifiée)
- Vent horizontal dominant
- Diffusion isotrope
- Pas de chimie atmosphérique secondaire
- Pas d’interaction particule-particule

---

## 7. Validation

Comparaison possible avec :

- Modèle du panache gaussien analytique
- HYSPLIT (NOAA)
- Données historiques de dispersion atmosphérique

Métriques :
- RMSE spatial
- Divergence KL
- Distance de Wasserstein

---

## 8. Stack technique proposée

Langage :
- Python

Librairies :
- NumPy
- SciPy
- xarray (données météo)
- pandas
- matplotlib / seaborn
- cartopy / folium
- rasterio
- netCDF4

Option performance :
- Numba
- CuPy (GPU)
- Multiprocessing

---

## 9. Extensions possibles

- Simulation 3D complète
- Intégration dépôt humide dynamique
- Modélisation dose inhalée
- Assimilation de données réelles
- Couplage avec modèle CFD

---

## 10. Résultat attendu

- Carte de densité spatiale
- Carte de probabilité de présence
- Carte de probabilité de dépassement seuil
- Animation temporelle
- Rapport quantitatif des incertitudes

---

## 11. Références scientifiques

- Turner, D.B. (1970). Workbook of Atmospheric Dispersion Estimates.
- Hanna, S.R., Briggs, G.A., Hosker, R.P. (1982). Handbook on Atmospheric Diffusion.
- Stohl, A. et al. (1998). Lagrangian particle dispersion models.
- NOAA HYSPLIT Documentation.

---

Fin du document.
