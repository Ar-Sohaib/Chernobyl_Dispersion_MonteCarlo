"""
Module de visualisation Globe — Projection orthographique spherique.
Rendu cinematographique minimaliste : heatmap inferno lissee sur globe desature.

Specifique Cartopy : la projection Orthographic ne supporte pas imshow,
on utilise pcolormesh (flat, edgecolors='none') + lissage gaussien
+ colormap avec alpha progressif integre.
"""
import os
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, FFMpegWriter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import GRID, VISUALIZATION, SOURCE, SIMULATION


# ═══════════════════════════════════════════════════════════════════════
#  Constantes visuelles
# ═══════════════════════════════════════════════════════════════════════

_GLOBE_LON = 25.0           # Centre du globe (longitude)
_GLOBE_LAT = 50.0           # Centre du globe (latitude)
_BG        = "#06090f"      # Fond bleu nuit profond
_SIGMA     = 1.5            # Lissage gaussien (en cellules)
_FONT      = "sans-serif"   # Police uniforme

# ── Typographie — hierarchie stricte ─────────────────────────────────
_F_TITLE  = 17              # Titre principal
_F_SUB    = 9.5             # Sous-titre
_F_INFO   = 9.5             # Panneau info
_F_DATE   = 10.5            # Date / heure
_F_CB     = 10              # Label colorbar
_F_CB_T   = 8.5             # Ticks colorbar
_F_CITY   = 6.0             # Villes secondaires
_F_CITY_P = 7.5             # Villes principales
_F_SRC    = 9.5             # Source (Tchernobyl)

# ── Villes — hierarchie visuelle ─────────────────────────────────────
_CITIES = [
    # nom             lon    lat    size     color      bold   dx    dy
    ("Tchernobyl",   30.10, 51.39, _F_SRC,  "#e04040", True,  0.8,  0.6),
    ("Kiev",         30.52, 50.45, _F_CITY_P,"#a0a0b8",False, 0.8, -1.0),
    ("Moscou",       37.62, 55.75, _F_CITY_P,"#a0a0b8",False, 0.8,  0.5),
    ("Paris",         2.35, 48.86, _F_CITY_P,"#8888a0",False, 0.8,  0.5),
    ("Stockholm",    18.07, 59.33, _F_CITY,  "#606878",False, 0.8,  0.5),
    ("Helsinki",     24.94, 60.17, _F_CITY,  "#606878",False,-2.8,  0.3),
    ("Berlin",       13.40, 52.52, _F_CITY,  "#606878",False, 0.8,  0.5),
    ("Londres",      -0.12, 51.51, _F_CITY,  "#606878",False, 0.8,  0.5),
    ("Rome",         12.50, 41.90, _F_CITY,  "#606878",False, 0.8,  0.5),
    ("Varsovie",     21.01, 52.23, _F_CITY,  "#606878",False, 0.8,  0.5),
    ("Vienne",       16.37, 48.21, _F_CITY,  "#606878",False, 0.8,  0.5),
    ("Minsk",        27.56, 53.90, _F_CITY,  "#606878",False, 0.8,  0.5),
    ("Bucarest",     26.10, 44.43, _F_CITY,  "#606878",False, 0.8,  0.5),
]


# ═══════════════════════════════════════════════════════════════════════
#  Colormap — inferno + alpha progressif integre
# ═══════════════════════════════════════════════════════════════════════

def _build_cmap():
    """
    Inferno (range 0.08-0.92) avec transparence progressive :
      - valeurs ~0   : invisible (alpha = 0)
      - valeurs basses : halo leger
      - valeurs hautes : riche mais pas opaque (max ~0.88)
    Pas besoin de parametre alpha global sur pcolormesh.
    """
    n = 256
    rgba = plt.cm.inferno(np.linspace(0.08, 0.92, n))

    a = np.zeros(n)
    a[:8]     = 0.0
    a[8:35]   = np.linspace(0.00, 0.28, 27)
    a[35:80]  = np.linspace(0.28, 0.52, 45)
    a[80:160] = np.linspace(0.52, 0.74, 80)
    a[160:]   = np.linspace(0.74, 0.88, 96)

    rgba[:, 3] = a
    cm = mcolors.ListedColormap(rgba)
    cm.set_under(alpha=0)
    cm.set_bad(alpha=0)
    return cm


_CMAP = _build_cmap()
_NORM = mcolors.PowerNorm(gamma=0.5, vmin=0.008, vmax=1.0)

# ── Echelle adaptative (reference 2K = 2560 px) ─────────────────────
def _s(base):
    """Multiplie une dimension de base par le ratio resolution / 2K."""
    return base * (VISUALIZATION["figsize"][0] * VISUALIZATION["dpi"]) / 2560.0

# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _outline(w=2.2):
    return [pe.withStroke(linewidth=w, foreground="black")]


def _smooth(data):
    """Lissage gaussien + clamp >= 0."""
    return np.clip(gaussian_filter(data.astype(np.float64), sigma=_SIGMA),
                   0, None)


def _wind_phase(t):
    for t0, t1, name in [
        (0,   48,  "N-NW \u2192 Scandinavie"),
        (48,  96,  "W \u2192 Pologne, Baltique"),
        (96,  168, "S-SW \u2192 Ukraine, Roumanie"),
        (168, 264, "W \u2192 Europe centrale"),
        (264, 480, "Dispersion large"),
    ]:
        if t0 <= t < t1:
            return name
    return "Dispersion large"


def _edges_mesh():
    """Grille d'aretes (nlat+1, nlon+1) pour pcolormesh flat."""
    lon_e = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)
    lat_e = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
    return np.meshgrid(lon_e, lat_e)


def _setup_globe(ax, crs_data):
    """
    Globe desature cinematographique :
    Blue Marble + overlays sombres + cotes subtiles + villes attenuees.
    """
    ax.set_facecolor(_BG)
    ax.set_global()

    # Fond Blue Marble
    ax.stock_img()

    # Desaturation : overlays semi-transparents sombres sur ocean et terre
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "ocean", "50m",
            facecolor=_BG, edgecolor="none"),
        alpha=0.42, zorder=1)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "50m",
            facecolor="#0a0d16", edgecolor="none"),
        alpha=0.36, zorder=1)

    # Cotes — subtiles
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m",
        edgecolor="#505068", facecolor="none",
        linewidth=_s(0.35)), zorder=3)

    # Frontieres — quasi-invisibles
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "50m",
        edgecolor="#35354a", facecolor="none",
        linewidth=_s(0.2), linestyle=":"), zorder=3)

    # Source — Tchernobyl
    ax.plot(SOURCE["lon"], SOURCE["lat"], marker="*",
            markersize=_s(12), color="#e04040",
            markeredgecolor="#ffaa44", markeredgewidth=_s(0.5),
            transform=crs_data, zorder=15)

    # Villes (hierarchie : source > principales > secondaires)
    for name, lon, lat, sz, col, bold, dx, dy in _CITIES:
        ms = _s(2.8) if bold else _s(1.8)
        ax.plot(lon, lat, "o", markersize=ms, color=col,
                markeredgecolor="none", transform=crs_data, zorder=12)
        ax.text(lon + dx, lat + dy, name,
                fontsize=_s(sz), color=col, fontfamily=_FONT,
                fontweight="bold" if bold else "light",
                transform=crs_data, zorder=12,
                path_effects=_outline(_s(2.0) if bold else _s(1.4)))

    # Bord du globe — presque invisible, fond dans le noir
    for sp in ax.spines.values():
        sp.set_edgecolor("#101020")
        sp.set_linewidth(_s(0.8))


# ═══════════════════════════════════════════════════════════════════════
#  Image statique Globe
# ═══════════════════════════════════════════════════════════════════════

def create_globe_map(mode, prob_map=None, mean_conc=None,
                     threshold_map=None, threshold=0.05,
                     traj_lon=None, traj_lat=None, active=None,
                     time_density_maps=None, filename=None):
    """
    Image PNG sur globe orthographique.
    Modes : probability, threshold, cumulative, instant.
    """
    os.makedirs(VISUALIZATION["save_dir"], exist_ok=True)

    proj = ccrs.Orthographic(_GLOBE_LON, _GLOBE_LAT)
    dc = ccrs.PlateCarree()
    lm, am = _edges_mesh()

    fig = plt.figure(figsize=VISUALIZATION["figsize"],
                     facecolor=_BG, dpi=VISUALIZATION["dpi"])

    # Layout cinematic : globe centré, bandes noires latérales
    ax  = fig.add_axes([0.22, 0.02, 0.56, 0.96], projection=proj)
    cax = fig.add_axes([0.82, 0.18, 0.012, 0.55])
    cax.set_facecolor(_BG)

    _setup_globe(ax, dc)

    # ── Donnees par mode ──────────────────────────────────────────────
    if mode == "probability":
        title = "Probabilite de presence"
        cb_label = "P(presence)"
        norm = _NORM
        data = prob_map
        default_fn = "globe_probability.png"

    elif mode == "threshold":
        title = f"Depassement de seuil  P(C > {threshold})"
        cb_label = f"P(C > {threshold})"
        norm = _NORM
        data = threshold_map
        default_fn = "globe_threshold.png"

    elif mode == "cumulative":
        title = "Densite cumulee"
        cb_label = "Concentration moy."
        norm = mcolors.PowerNorm(
            gamma=0.4, vmin=0.005,
            vmax=max(mean_conc.max(), 1) * 0.6)
        data = mean_conc
        default_fn = "globe_cumulative.png"

    elif mode == "instant":
        title = "Nuage instantane  t = 480h"
        cb_label = "Densite"
        t_last = traj_lon.shape[0] - 1
        data = time_density_maps[t_last]
        norm = mcolors.PowerNorm(
            gamma=0.35, vmin=0.5,
            vmax=max(data.max(), 1) * 0.6)
        default_fn = "globe_instant.png"

        alive = active[t_last]
        if np.sum(alive) > 0:
            ax.scatter(traj_lon[t_last, alive], traj_lat[t_last, alive],
                       s=1.5, c="#ffcc00", alpha=0.4,
                       transform=dc, zorder=6, edgecolors="none")
    else:
        raise ValueError(f"Mode inconnu : {mode}")

    data_s = _smooth(data)
    fn = filename or default_fn

    # pcolormesh — pas de maillage, pas d'artefacts
    pcm = ax.pcolormesh(
        lm, am, data_s,
        cmap=_CMAP, norm=norm,
        transform=dc, zorder=5,
        shading="flat",
        edgecolors="none",
        antialiased=True,
    )

    # ── Colorbar fine ─────────────────────────────────────────────────
    cb = fig.colorbar(pcm, cax=cax, orientation="vertical")
    cb.set_label(cb_label, color="#9090a8", fontsize=_s(_F_CB),
                 fontfamily=_FONT, labelpad=_s(8))
    cb.ax.tick_params(colors="#707088", labelsize=_s(_F_CB_T),
                      width=_s(0.4), length=_s(2.5))
    cb.outline.set_edgecolor("#202030")
    cb.outline.set_linewidth(_s(0.4))

    # ── Légende — bande noire gauche ──────────────────────────────────
    fig.text(0.03, 0.92, title,
             fontsize=_s(_F_INFO), color="#c0c0d4",
             va="top", ha="left", fontfamily="monospace",
             path_effects=_outline(_s(2.0)))

    # ── Sauvegarde ────────────────────────────────────────────────────
    path = os.path.join(VISUALIZATION["save_dir"], fn)
    fig.savefig(path, dpi=VISUALIZATION["dpi"],
                facecolor=fig.get_facecolor())
    plt.close(fig)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] Image globe sauvegardee : {path}  ({size_mb:.1f} Mo)")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Video Globe — Heatmap probabiliste animee
# ═══════════════════════════════════════════════════════════════════════

def create_globe_video(time_prob_maps,
                       filename="chernobyl_globe_video.mp4"):
    """
    Video sur globe orthographique — heatmap de probabilite animee.
    Inferno + lissage gaussien + panneau info unifie.
    """
    os.makedirs(VISUALIZATION["save_dir"], exist_ok=True)

    dt_sim = SIMULATION["dt"]
    n_steps = time_prob_maps.shape[0]
    dpi = VISUALIZATION["dpi"]
    fw, fh = VISUALIZATION["figsize"]

    fps = VISUALIZATION["video_fps"]
    dur = VISUALIZATION["video_duration_s"]
    total = fps * dur
    f2s = np.linspace(0, n_steps - 1, total).astype(int)

    proj = ccrs.Orthographic(_GLOBE_LON, _GLOBE_LAT)
    dc = ccrs.PlateCarree()
    lm, am = _edges_mesh()
    acc = datetime.datetime(1986, 4, 26, 1, 23)

    print(f"  Encodage globe ({total} frames a {fps} fps)...")
    print(f"     Resolution : {int(fw * dpi)}x{int(fh * dpi)} px")

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(fw, fh), facecolor=_BG, dpi=dpi)

    # Layout cinematic : globe centré, bandes noires latérales
    ax  = fig.add_axes([0.22, 0.02, 0.56, 0.96], projection=proj)
    cax = fig.add_axes([0.82, 0.18, 0.010, 0.55])
    cax.set_facecolor(_BG)

    _setup_globe(ax, dc)

    # ── Heatmap (mise a jour via set_array) ───────────────────────────
    pcm = ax.pcolormesh(
        lm, am,
        np.zeros((GRID["nlat"], GRID["nlon"])),
        cmap=_CMAP, norm=_NORM,
        transform=dc, zorder=5,
        shading="flat",
        edgecolors="none",
        antialiased=True,
    )

    # ── Colorbar fine ─────────────────────────────────────────────────
    cb = fig.colorbar(pcm, cax=cax, orientation="vertical")
    cb.set_label("P(presence)", color="#9090a8", fontsize=_s(_F_CB),
                 fontfamily=_FONT, labelpad=_s(6))
    cb.ax.tick_params(colors="#707088", labelsize=_s(_F_CB_T),
                      width=_s(0.4), length=_s(2.5))
    cb.outline.set_edgecolor("#202030")
    cb.outline.set_linewidth(_s(0.4))

    # ── Info — stack vertical, bande noire gauche ─────────────────────
    info = fig.text(
        0.03, 0.92, "",
        fontsize=_s(_F_INFO), color="#c0c0d4", va="top", ha="left",
        fontfamily="monospace", linespacing=1.6,
        path_effects=_outline(_s(2.2)),
    )

    # ── Update ────────────────────────────────────────────────────────
    def update(frame_idx):
        t = f2s[frame_idx]
        th = t * dt_sim

        cur = acc + datetime.timedelta(hours=th)
        ds = cur.strftime("%d %b %Y  %Hh00")

        pt = time_prob_maps[t]
        ps = _smooth(pt)
        pcm.set_array(ps.ravel())

        pm = pt.max()

        info.set_text(
            f"{ds}\n"
            f"t + {th:.0f}h\n"
            f"{_wind_phase(th)}\n"
            f"P max  {pm:.2f}"
        )

        return [pcm, info]

    # ── Encodage H.264 ────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=total,
                         interval=1000 // fps, blit=False)

    writer = FFMpegWriter(
        fps=fps,
        bitrate=VISUALIZATION["video_bitrate"],
        codec="libx264",
        extra_args=[
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
        metadata={"title": "Chernobyl Globe \u2014 Monte Carlo",
                  "artist": "MonteCarloSimu"},
    )

    path = os.path.join(VISUALIZATION["save_dir"], filename)
    anim.save(path, writer=writer)
    plt.close(fig)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] Video globe sauvegardee : {path}  ({size_mb:.1f} Mo)")
    return path
