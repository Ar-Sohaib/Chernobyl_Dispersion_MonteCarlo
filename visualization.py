"""
Module de visualisation — Vidéo 4K 120fps Tchernobyl sur carte satellite.
4 panneaux symétriques sur fond de carte réel (Blue Marble + Natural Earth 50m) :
  ┌──────────────────┬──────────────────┐
  │ 1. Nuage instant │ 2. Densité cum.  │
  ├──────────────────┼──────────────────┤
  │ 3. P(présence)   │ 4. P(C > seuil)  │
  └──────────────────┴──────────────────┘
Résolution : 3840×2160 (4K UHD) — 120 fps
"""
import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from config import GRID, VISUALIZATION, SOURCE, SIMULATION


# ═══════════════════════════════════════════════════════════════════════
#  Constantes style — adaptées 4K
# ═══════════════════════════════════════════════════════════════════════

_FONT_TITLE   = 22      # Titre de panneau
_FONT_SUPTITLE = 28     # Titre global
_FONT_BAR     = 20      # Barre de temps
_FONT_CITY    = 13      # Noms de villes
_FONT_CITY_BIG = 15     # Villes principales
_FONT_CB      = 14      # Labels colorbar
_FONT_CB_TICK = 11      # Ticks colorbar
_FONT_INFO    = 15      # Texte info overlay
_MARKER_SOURCE = 22     # Étoile source
_MARKER_CITY   = 4.5    # Point ville
_OUTLINE_W     = 3.5    # Épaisseur contour texte
_SCATTER_SIZE  = 2.5    # Taille particules scatter
_COAST_LW      = 0.9    # Épaisseur côtes
_BORDER_LW     = 0.6    # Épaisseur frontières


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _setup_output_dir():
    os.makedirs(VISUALIZATION["save_dir"], exist_ok=True)


def _extent():
    return [GRID["lon_min"], GRID["lon_max"],
            GRID["lat_min"], GRID["lat_max"]]


def _text_outline():
    return [pe.withStroke(linewidth=_OUTLINE_W, foreground="black")]


def _setup_map_ax(ax, data_crs, add_cities=True, title=None):
    """
    Fond de carte raster haute résolution + features 50m.
    """
    ax.set_extent(_extent(), crs=data_crs)

    # Image raster Europe (Natural Earth Hypsometric)
    bg_path = VISUALIZATION.get("bg_image")
    if bg_path and os.path.isfile(bg_path):
        bg_img = plt.imread(bg_path)
        ax.imshow(bg_img, origin="upper", extent=_extent(),
                  transform=data_crs, zorder=0, interpolation="bilinear")
    else:
        ax.stock_img()  # Fallback Blue Marble

    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m",
        edgecolor="#cccccc", facecolor="none", linewidth=_COAST_LW), zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "50m",
        edgecolor="#999999", facecolor="none", linewidth=_BORDER_LW,
        linestyle="--"), zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "lakes", "50m",
        edgecolor="#666666", facecolor="#1a3050", linewidth=0.4), zorder=2)

    # Source
    ax.plot(SOURCE["lon"], SOURCE["lat"], marker="*",
            markersize=_MARKER_SOURCE, color="#ff0000",
            markeredgecolor="#ffff00", markeredgewidth=1.0,
            transform=data_crs, zorder=15)

    if add_cities:
        _add_cities(ax, data_crs)

    if title:
        ax.set_title(title, color="white", fontsize=_FONT_TITLE,
                     fontweight="bold", pad=14)

    # Cadre subtil
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
        sp.set_linewidth(1.2)


def _add_cities(ax, transform):
    # (nom, lon, lat, taille, couleur, offset_lon, offset_lat)
    cities = [
        ("Tchernobyl",      30.10, 51.39, _FONT_CITY_BIG, "#ff4444",  0.7,  0.4),
        ("Moscou",          37.62, 55.75, _FONT_CITY_BIG, "white",    0.7,  0.4),
        ("Kiev",            30.52, 50.45, _FONT_CITY_BIG, "white",    0.7, -0.9),
        ("Minsk",           27.56, 53.90, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("Stockholm",       18.07, 59.33, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("Helsinki",        24.94, 60.17, _FONT_CITY, "#dddddd",    -3.5,  0.1),
        ("Berlin",          13.40, 52.52, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("Paris",            2.35, 48.86, _FONT_CITY_BIG, "white",    0.7,  0.4),
        ("Londres",         -0.12, 51.51, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("Rome",            12.50, 41.90, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("Ankara",          32.87, 39.93, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("Varsovie",        21.01, 52.23, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("Vienne",          16.37, 48.21, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("Bucarest",        26.10, 44.43, _FONT_CITY, "#dddddd",     0.7,  0.4),
        ("St-Pétersbourg",  30.32, 59.93, _FONT_CITY, "#dddddd",     0.7,  0.4),
    ]
    for name, lon, lat, size, color, dx, dy in cities:
        ax.plot(lon, lat, "o", markersize=_MARKER_CITY, color=color,
                transform=transform, zorder=12)
        ax.text(lon + dx, lat + dy, name, fontsize=size, color=color,
                transform=transform, zorder=12,
                fontweight="bold" if name == "Tchernobyl" else "normal",
                path_effects=_text_outline())


def _get_wind_phase(t_hours):
    phases = [
        (0,   48,  "N-NW → Scandinavie"),
        (48,  96,  "W → Pologne, Baltique"),
        (96,  168, "S-SW → Ukraine, Roumanie"),
        (168, 264, "W → Europe centrale"),
        (264, 480, "Dispersion large"),
    ]
    for t0, t1, name in phases:
        if t0 <= t_hours < t1:
            return name
    return "Dispersion large"


def _make_colorbar(fig, cax, mappable, label):
    """Colorbar dans un axes dédié (cax) pour un alignement parfait."""
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cb.set_label(label, color="white", fontsize=_FONT_CB)
    cb.ax.tick_params(colors="white", labelsize=_FONT_CB_TICK)
    cb.outline.set_edgecolor("#555555")
    cb.outline.set_linewidth(0.8)
    return cb


# ═══════════════════════════════════════════════════════════════════════
#  Image unique — un seul panneau plein écran 4K
# ═══════════════════════════════════════════════════════════════════════

def create_single_map(mode, prob_map=None, mean_conc=None,
                      threshold_map=None, threshold=0.05,
                      traj_lon=None, traj_lat=None, active=None,
                      time_density_maps=None,
                      filename=None):
    """
    Produit une image PNG 4K unique pour un seul type de carte.
    En mode 'probability', la carte est agrandie pour voir les détails.
    """
    _setup_output_dir()

    proj = ccrs.LambertConformal(
        central_longitude=30.0, central_latitude=52.0,
        standard_parallels=(40, 60)
    )
    data_crs = ccrs.PlateCarree()
    data_ext = _extent()

    # Tailles adaptées single map (fontes plus grandes)
    font_title = 36
    font_cb = 22
    font_cb_tick = 16
    font_city = 18
    font_city_big = 22
    marker_source = 30
    marker_city = 7
    outline_w = 5
    font_info = 22

    fig = plt.figure(
        figsize=VISUALIZATION["figsize"],
        facecolor="#08080f",
        dpi=VISUALIZATION["dpi"],
    )

    # Layout plein écran : carte + colorbar
    margin = 0.04
    cb_w = 0.015
    cb_gap = 0.012
    title_h = 0.06

    map_left = margin
    map_bottom = margin
    map_w = 1.0 - 2 * margin - cb_w - cb_gap
    map_h = 1.0 - title_h - margin

    ax = fig.add_axes([map_left, map_bottom, map_w, map_h], projection=proj)
    cax = fig.add_axes([map_left + map_w + cb_gap,
                        map_bottom + map_h * 0.05,
                        cb_w, map_h * 0.9])
    cax.set_facecolor("#08080f")

    # Fond de carte
    ax.set_extent(data_ext, crs=data_crs)
    bg_path = VISUALIZATION.get("bg_image")
    if bg_path and os.path.isfile(bg_path):
        bg_img = plt.imread(bg_path)
        ax.imshow(bg_img, origin="upper", extent=data_ext,
                  transform=data_crs, zorder=0, interpolation="bilinear")
    else:
        ax.stock_img()

    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m",
        edgecolor="#cccccc", facecolor="none", linewidth=1.2), zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "50m",
        edgecolor="#999999", facecolor="none", linewidth=0.8,
        linestyle="--"), zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "lakes", "50m",
        edgecolor="#666666", facecolor="#1a3050", linewidth=0.5), zorder=2)

    # Source
    ax.plot(SOURCE["lon"], SOURCE["lat"], marker="*",
            markersize=marker_source, color="#ff0000",
            markeredgecolor="#ffff00", markeredgewidth=1.2,
            transform=data_crs, zorder=15)

    # Villes (fontes agrandies)
    cities = [
        ("Tchernobyl",      30.10, 51.39, font_city_big, "#ff4444",  0.7,  0.4),
        ("Moscou",          37.62, 55.75, font_city_big, "white",    0.7,  0.4),
        ("Kiev",            30.52, 50.45, font_city_big, "white",    0.7, -0.9),
        ("Minsk",           27.56, 53.90, font_city, "#dddddd",     0.7,  0.4),
        ("Stockholm",       18.07, 59.33, font_city, "#dddddd",     0.7,  0.4),
        ("Helsinki",        24.94, 60.17, font_city, "#dddddd",    -3.5,  0.1),
        ("Berlin",          13.40, 52.52, font_city, "#dddddd",     0.7,  0.4),
        ("Paris",            2.35, 48.86, font_city_big, "white",    0.7,  0.4),
        ("Londres",         -0.12, 51.51, font_city, "#dddddd",     0.7,  0.4),
        ("Rome",            12.50, 41.90, font_city, "#dddddd",     0.7,  0.4),
        ("Ankara",          32.87, 39.93, font_city, "#dddddd",     0.7,  0.4),
        ("Varsovie",        21.01, 52.23, font_city, "#dddddd",     0.7,  0.4),
        ("Vienne",          16.37, 48.21, font_city, "#dddddd",     0.7,  0.4),
        ("Bucarest",        26.10, 44.43, font_city, "#dddddd",     0.7,  0.4),
        ("St-Pétersbourg",  30.32, 59.93, font_city, "#dddddd",     0.7,  0.4),
    ]
    outline = [pe.withStroke(linewidth=outline_w, foreground="black")]
    for name, lon, lat, size, color, dx, dy in cities:
        ax.plot(lon, lat, "o", markersize=marker_city, color=color,
                transform=data_crs, zorder=12)
        ax.text(lon + dx, lat + dy, name, fontsize=size, color=color,
                transform=data_crs, zorder=12,
                fontweight="bold" if name == "Tchernobyl" else "normal",
                path_effects=outline)

    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
        sp.set_linewidth(1.5)

    # ── Overlay de données ────────────────────────────────────────────
    if mode == "probability":
        title = "Probabilite de presence -- Monte Carlo (12 runs)"
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_under(alpha=0)
        norm = mcolors.PowerNorm(gamma=0.5, vmin=0.01, vmax=1.0)
        im = ax.imshow(prob_map, extent=data_ext, origin="lower",
                       cmap=cmap, norm=norm, alpha=0.75,
                       transform=data_crs, zorder=5,
                       interpolation="bilinear")
        cb_label = "P(présence)"
        if filename is None:
            filename = "probability_map.png"

    elif mode == "threshold":
        title = f"Depassement de seuil -- P(C > {threshold})"
        cmap = plt.cm.RdYlGn_r.copy()
        cmap.set_under(alpha=0)
        norm = mcolors.PowerNorm(gamma=0.5, vmin=0.01, vmax=1.0)
        im = ax.imshow(threshold_map, extent=data_ext, origin="lower",
                       cmap=cmap, norm=norm, alpha=0.75,
                       transform=data_crs, zorder=5,
                       interpolation="bilinear")
        cb_label = f"P(C > {threshold})"
        if filename is None:
            filename = "threshold_map.png"

    elif mode == "cumulative":
        title = "Densite cumulee -- integrale temporelle"
        cmap = plt.cm.inferno.copy()
        cmap.set_under(alpha=0)
        norm = mcolors.PowerNorm(gamma=0.4, vmin=0.5,
                                 vmax=max(mean_conc.max(), 1) * 0.6)
        im = ax.imshow(mean_conc, extent=data_ext, origin="lower",
                       cmap=cmap, norm=norm, alpha=0.8,
                       transform=data_crs, zorder=5,
                       interpolation="bilinear")
        cb_label = "Concentration moyenne"
        if filename is None:
            filename = "cumulative_map.png"

    elif mode == "instant":
        title = "Nuage instantane -- etat final (t = 480h)"
        cmap = plt.cm.hot.copy()
        cmap.set_under(alpha=0)
        t_last = traj_lon.shape[0] - 1
        data = time_density_maps[t_last]
        norm = mcolors.PowerNorm(gamma=0.35, vmin=0.5,
                                 vmax=max(data.max(), 1) * 0.6)
        im = ax.imshow(data, extent=data_ext, origin="lower",
                       cmap=cmap, norm=norm, alpha=0.7,
                       transform=data_crs, zorder=5,
                       interpolation="bilinear")
        # Particules
        alive = active[t_last]
        if np.sum(alive) > 0:
            ax.scatter(traj_lon[t_last, alive], traj_lat[t_last, alive],
                       s=4, c="#ffcc00", alpha=0.6,
                       transform=data_crs, zorder=6, edgecolors="none")
        cb_label = "Densité instantanée"
        if filename is None:
            filename = "instant_map.png"
    else:
        raise ValueError(f"Mode inconnu : {mode}")

    # Colorbar
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label(cb_label, color="white", fontsize=font_cb)
    cb.ax.tick_params(colors="white", labelsize=font_cb_tick)
    cb.outline.set_edgecolor("#555555")
    cb.outline.set_linewidth(1.0)

    # Titre
    fig.suptitle(title, fontsize=font_title, fontweight="bold",
                 color="white", y=1.0 - title_h * 0.4,
                 fontfamily="sans-serif")

    path = os.path.join(VISUALIZATION["save_dir"], filename)
    fig.savefig(path, dpi=VISUALIZATION["dpi"], facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] Image 4K sauvegardee : {path}  ({size_mb:.1f} Mo)")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Vidéo probabilité dynamique — heatmap animée plein écran 4K
# ═══════════════════════════════════════════════════════════════════════

def _render_basemap_image():
    """
    Pré-rend le fond de carte (raster + côtes + frontières + villes)
    en un seul array RGBA numpy. Appelé une seule fois.
    """
    dpi = VISUALIZATION["dpi"]
    figw, figh = VISUALIZATION["figsize"]
    data_ext = _extent()
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree()

    fig_tmp = plt.figure(figsize=(figw, figh), dpi=dpi, facecolor="#08080f")
    ax = fig_tmp.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_extent(data_ext, crs=data_crs)
    ax.set_facecolor("#08080f")

    # Image raster
    bg_path = VISUALIZATION.get("bg_image")
    if bg_path and os.path.isfile(bg_path):
        bg_img = plt.imread(bg_path)
        ax.imshow(bg_img, origin="upper", extent=data_ext,
                  transform=data_crs, zorder=0, interpolation="bilinear")
    else:
        ax.stock_img()

    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m",
        edgecolor="#cccccc", facecolor="none", linewidth=_COAST_LW), zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "50m",
        edgecolor="#999999", facecolor="none", linewidth=_BORDER_LW,
        linestyle="--"), zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "lakes", "50m",
        edgecolor="#666666", facecolor="#1a3050", linewidth=0.4), zorder=2)

    ax.plot(SOURCE["lon"], SOURCE["lat"], marker="*",
            markersize=_MARKER_SOURCE, color="#ff0000",
            markeredgecolor="#ffff00", markeredgewidth=1.0,
            transform=data_crs, zorder=15)
    _add_cities(ax, data_crs)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig_tmp.canvas.draw()
    buf = np.frombuffer(fig_tmp.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig_tmp.canvas.get_width_height()
    basemap = buf.reshape(h, w, 4).copy()
    plt.close(fig_tmp)

    print("  [OK] Fond de carte pre-rendu en cache")
    return basemap


def create_proba_video(time_prob_maps,
                       filename="chernobyl_proba_video.mp4"):
    """
    Vidéo 4K plein écran — carte plate (PlateCarree), pas de blanc.
    Le fond de carte est pré-rendu une seule fois puis réutilisé
    comme image statique → rendu ultra-rapide.
    """
    _setup_output_dir()

    dt = SIMULATION["dt"]
    n_steps = time_prob_maps.shape[0]
    data_ext = _extent()
    dpi = VISUALIZATION["dpi"]
    figw, figh = VISUALIZATION["figsize"]
    px_w, px_h = int(figw * dpi), int(figh * dpi)

    # ── Frames ────────────────────────────────────────────────────────
    fps = VISUALIZATION["video_fps"]
    duration = VISUALIZATION["video_duration_s"]
    total_frames = fps * duration
    frame_to_step = np.linspace(0, n_steps - 1, total_frames).astype(int)

    # ── Pré-rendre le fond de carte (une seule fois) ──────────────────
    basemap_rgba = _render_basemap_image()

    # ── Figure réelle (axes matplotlib purs = très rapide) ────────────
    fig = plt.figure(figsize=(figw, figh), facecolor="#08080f", dpi=dpi)

    cb_w = 0.018
    cb_gap = 0.006
    map_w = 1.0 - cb_w - cb_gap

    # Axes carte — bord à bord (plain matplotlib, pas Cartopy)
    ax = fig.add_axes([0, 0, map_w, 1.0])
    ax.set_facecolor("#08080f")
    ax.set_xlim(data_ext[0], data_ext[1])
    ax.set_ylim(data_ext[2], data_ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Fond de carte (image statique pré-rendue)
    ax.imshow(basemap_rgba, extent=data_ext, origin="upper",
              aspect="auto", zorder=0, interpolation="bilinear")

    # Colorbar axis
    cax = fig.add_axes([map_w + cb_gap, 0.05, cb_w, 0.9])
    cax.set_facecolor("#08080f")

    # ── Heatmap probabilité ───────────────────────────────────────────
    prob_cmap = plt.cm.YlOrRd.copy()
    prob_cmap.set_under(alpha=0)
    prob_norm = mcolors.PowerNorm(gamma=0.45, vmin=0.01, vmax=1.0)

    im = ax.imshow(
        np.zeros((GRID["nlat"], GRID["nlon"])),
        extent=data_ext, origin="lower",
        cmap=prob_cmap, norm=prob_norm,
        alpha=0.75, aspect="auto", zorder=5,
        interpolation="bilinear",
    )

    # Colorbar
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("P(présence)", color="white", fontsize=20)
    cb.ax.tick_params(colors="white", labelsize=14)
    cb.outline.set_edgecolor("#555555")
    cb.outline.set_linewidth(1.0)

    # ── Info overlay ──────────────────────────────────────────────────
    info_text = ax.text(
        0.012, 0.97, "", transform=ax.transAxes,
        fontsize=18, color="#00ffaa", va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black",
                  alpha=0.88, edgecolor="#444"),
        zorder=20,
    )

    max_text = ax.text(
        0.988, 0.03, "", transform=ax.transAxes,
        fontsize=16, color="#ffcc00", va="bottom", ha="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                  alpha=0.85, edgecolor="#444"),
        zorder=20,
    )

    # ── Titre + date en overlay sur la carte ──────────────────────────
    accident_date = datetime.datetime(1986, 4, 26, 1, 23)

    ax.text(
        0.5, 0.99,
        "Probabilite de presence -- Monte Carlo dynamique"
        f"  ({SIMULATION['n_runs']} runs × {SIMULATION['n_particles']} particules)",
        transform=ax.transAxes,
        fontsize=22, fontweight="bold", color="white",
        ha="center", va="top", fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black",
                  alpha=0.7, edgecolor="#444"),
        zorder=20,
    )

    date_text = ax.text(
        0.5, 0.012, "", transform=ax.transAxes,
        fontsize=20, fontweight="bold", color="white",
        ha="center", va="bottom", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                  alpha=0.75, edgecolor="#444"),
        zorder=20,
    )

    # ── Update (très rapide : juste set_data + textes) ────────────────
    def update(frame_idx):
        t = frame_to_step[frame_idx]
        t_hours = t * dt

        current = accident_date + datetime.timedelta(hours=t_hours)
        date_str = current.strftime("%d %b %Y — %Hh00")

        date_text.set_text(
            f"t + {t_hours:>5.0f}h   │   {date_str}   │   "
            f"{_get_wind_phase(t_hours)}"
        )

        prob_t = time_prob_maps[t]
        im.set_data(prob_t)

        pmax = prob_t.max()
        n_cells = int(np.sum(prob_t > 0.01))
        info_text.set_text(
            f"Heure : t + {t_hours:.0f}h\n"
            f"Vent  : {_get_wind_phase(t_hours)}\n"
            f"P max : {pmax:.2f}"
        )
        max_text.set_text(
            f"Cellules contaminées : {n_cells:,}  │  "
            f"Surface : ~{n_cells * 40:.0f} km²"
        )

        return [im, date_text, info_text, max_text]

    # ── Encodage ──────────────────────────────────────────────────────
    print(f"  Encodage 4K ({total_frames} frames a {fps} fps)...")

    anim = FuncAnimation(fig, update, frames=total_frames,
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
        metadata={"title": "Chernobyl Probability Heatmap — 4K",
                  "artist": "MonteCarloSimu"},
    )

    path = os.path.join(VISUALIZATION["save_dir"], filename)
    anim.save(path, writer=writer)
    plt.close(fig)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] Video 4K sauvegardee : {path}  ({size_mb:.1f} Mo)")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Vidéo principale — 4 panneaux symétriques, 4K, 120 fps
# ═══════════════════════════════════════════════════════════════════════

def create_video(traj_lon, traj_lat, active, time_density_maps,
                 lon_edges, lat_edges,
                 prob_map, mean_conc, threshold_map, threshold,
                 filename="chernobyl_simulation.mp4"):

    _setup_output_dir()

    dt = SIMULATION["dt"]
    n_steps = traj_lon.shape[0]
    data_ext = _extent()

    # ── Pré-calcul ────────────────────────────────────────────────────
    cumulative_maps = np.cumsum(time_density_maps, axis=0)
    cum_max = max(cumulative_maps.max(), 1)
    inst_max = max(time_density_maps.max(), 1)

    # ── Frames : mapper n_steps → total_frames (120fps × durée) ──────
    fps = VISUALIZATION["video_fps"]
    duration = VISUALIZATION["video_duration_s"]
    total_video_frames = fps * duration  # 2400 frames à 120fps

    # Mapping linéaire : chaque frame vidéo → index de pas de temps
    frame_to_step = np.linspace(0, n_steps - 1, total_video_frames).astype(int)
    n_frames = total_video_frames

    # ── Projection ────────────────────────────────────────────────────
    proj = ccrs.LambertConformal(
        central_longitude=30.0, central_latitude=52.0,
        standard_parallels=(40, 60)
    )
    data_crs = ccrs.PlateCarree()

    # ═════════════════════════════════════════════════════════════════
    #  FIGURE — Layout symétrique avec colorbars dédiées
    # ═════════════════════════════════════════════════════════════════
    fig = plt.figure(
        figsize=VISUALIZATION["figsize"],   # 32×18 @ 120dpi = 3840×2160
        facecolor="#08080f",
        dpi=VISUALIZATION["dpi"],
    )

    # Marges parfaitement symétriques
    margin_lr = 0.035    # gauche / droite
    margin_tb = 0.04     # bas
    bar_h     = 0.025    # hauteur barre de temps
    gap_bar   = 0.025    # gap entre barre et panneaux (augmenté)
    gap_h     = 0.06     # gap vertical entre rangées
    gap_w     = 0.05     # gap horizontal entre colonnes
    title_h   = 0.055    # espace pour le suptitle (augmenté)
    cb_w      = 0.012    # largeur colorbar
    cb_gap    = 0.008    # gap entre panneau et colorbar

    top_start = 1.0 - title_h
    bar_bottom = top_start - bar_h
    panels_top = bar_bottom - gap_bar
    panels_bottom = margin_tb
    panel_h = (panels_top - panels_bottom - gap_h) / 2.0
    # Largeur panneau = (espace total - 2 marges - gap central - 2 colorbars - 2 gaps cb) / 2
    panel_w = (1.0 - 2 * margin_lr - gap_w - 2 * cb_w - 2 * cb_gap) / 2.0

    # Barre de progression — pleine largeur
    ax_bar = fig.add_axes([margin_lr, bar_bottom, 1.0 - 2 * margin_lr, bar_h])

    # 4 panneaux symétriques + 4 colorbars alignées
    row1_bottom = panels_top - panel_h
    row2_bottom = row1_bottom - gap_h - panel_h
    col1_left = margin_lr
    col2_left = margin_lr + panel_w + cb_w + cb_gap + gap_w

    ax1 = fig.add_axes([col1_left, row1_bottom, panel_w, panel_h], projection=proj)
    ax2 = fig.add_axes([col2_left, row1_bottom, panel_w, panel_h], projection=proj)
    ax3 = fig.add_axes([col1_left, row2_bottom, panel_w, panel_h], projection=proj)
    ax4 = fig.add_axes([col2_left, row2_bottom, panel_w, panel_h], projection=proj)

    # Axes dédiés pour les colorbars (parfaitement alignés avec les panneaux)
    cb_shrink = 0.05  # marge interne haut/bas des colorbars
    cax1 = fig.add_axes([col1_left + panel_w + cb_gap,
                         row1_bottom + panel_h * cb_shrink,
                         cb_w, panel_h * (1 - 2 * cb_shrink)])
    cax2 = fig.add_axes([col2_left + panel_w + cb_gap,
                         row1_bottom + panel_h * cb_shrink,
                         cb_w, panel_h * (1 - 2 * cb_shrink)])
    cax3 = fig.add_axes([col1_left + panel_w + cb_gap,
                         row2_bottom + panel_h * cb_shrink,
                         cb_w, panel_h * (1 - 2 * cb_shrink)])
    cax4 = fig.add_axes([col2_left + panel_w + cb_gap,
                         row2_bottom + panel_h * cb_shrink,
                         cb_w, panel_h * (1 - 2 * cb_shrink)])
    for cax in (cax1, cax2, cax3, cax4):
        cax.set_facecolor("#08080f")

    # ── Barre de progression ──────────────────────────────────────────
    ax_bar.set_xlim(0, n_steps * dt)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_facecolor("#12141a")
    ax_bar.set_yticks([])
    ax_bar.tick_params(colors="white", labelsize=_FONT_CB_TICK)
    for sp in ax_bar.spines.values():
        sp.set_color("#444")
        sp.set_linewidth(1.0)
    progress_fill = ax_bar.axvspan(0, 0, color="#ff4444", alpha=0.5)
    time_text = ax_bar.text(0.5, 0.5, "", transform=ax_bar.transAxes,
                            ha="center", va="center", fontsize=_FONT_BAR,
                            color="white", fontweight="bold",
                            fontfamily="monospace")

    # ═════════════════════════════════════════════════════════════════
    # PANNEAU 1 — Nuage instantané
    # ═════════════════════════════════════════════════════════════════
    _setup_map_ax(ax1, data_crs, add_cities=True,
                  title="Nuage instantane -- particules actives")

    scatter = ax1.scatter([], [], s=_SCATTER_SIZE, c="#ffcc00", alpha=0.55,
                          transform=data_crs, zorder=6, edgecolors="none")

    inst_cmap = plt.cm.hot.copy()
    inst_cmap.set_under(alpha=0)
    inst_norm = mcolors.PowerNorm(gamma=0.35, vmin=0.5, vmax=inst_max * 0.6)

    im_inst = ax1.imshow(
        np.zeros((GRID["nlat"], GRID["nlon"])),
        extent=data_ext, origin="lower",
        cmap=inst_cmap, norm=inst_norm,
        alpha=0.7, transform=data_crs, zorder=5,
        interpolation="bilinear",
    )
    _make_colorbar(fig, cax1, im_inst, "Densité instantanée")

    info_text = ax1.text(
        0.015, 0.97, "", transform=ax1.transAxes,
        fontsize=_FONT_INFO, color="#00ffaa", va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                  alpha=0.85, edgecolor="#444"),
        zorder=20
    )

    # ═════════════════════════════════════════════════════════════════
    # PANNEAU 2 — Densité cumulée
    # ═════════════════════════════════════════════════════════════════
    _setup_map_ax(ax2, data_crs, add_cities=True,
                  title="Densité cumulée (0 → t)")

    cum_cmap = plt.cm.inferno.copy()
    cum_cmap.set_under(alpha=0)
    cum_norm = mcolors.PowerNorm(gamma=0.4, vmin=0.5, vmax=cum_max * 0.25)

    im_cum = ax2.imshow(
        np.zeros((GRID["nlat"], GRID["nlon"])),
        extent=data_ext, origin="lower",
        cmap=cum_cmap, norm=cum_norm,
        alpha=0.8, transform=data_crs, zorder=5,
        interpolation="bilinear",
    )
    _make_colorbar(fig, cax2, im_cum, "Passages cumulés")

    # ═════════════════════════════════════════════════════════════════
    # PANNEAU 3 — Probabilité de présence (statique)
    # ═════════════════════════════════════════════════════════════════
    _setup_map_ax(ax3, data_crs, add_cities=True,
                  title="Probabilité de présence (Monte Carlo)")

    prob_cmap = plt.cm.YlOrRd.copy()
    prob_cmap.set_under(alpha=0)
    prob_norm = mcolors.PowerNorm(gamma=0.5, vmin=0.01, vmax=1.0)

    im_prob = ax3.imshow(
        prob_map, extent=data_ext, origin="lower",
        cmap=prob_cmap, norm=prob_norm,
        alpha=0.75, transform=data_crs, zorder=5,
        interpolation="bilinear",
    )
    _make_colorbar(fig, cax3, im_prob, "P(présence)")

    # ═════════════════════════════════════════════════════════════════
    # PANNEAU 4 — Dépassement de seuil (statique)
    # ═════════════════════════════════════════════════════════════════
    _setup_map_ax(ax4, data_crs, add_cities=True,
                  title=f"Dépassement de seuil — P(C > {threshold})")

    thr_cmap = plt.cm.RdYlGn_r.copy()
    thr_cmap.set_under(alpha=0)
    thr_norm = mcolors.PowerNorm(gamma=0.5, vmin=0.01, vmax=1.0)

    im_thr = ax4.imshow(
        threshold_map, extent=data_ext, origin="lower",
        cmap=thr_cmap, norm=thr_norm,
        alpha=0.75, transform=data_crs, zorder=5,
        interpolation="bilinear",
    )
    _make_colorbar(fig, cax4, im_thr, f"P(C > {threshold})")

    # ── Titre global ──────────────────────────────────────────────────
    fig.suptitle(
        "Simulation Monte Carlo -- Propagation du nuage de Tchernobyl"
        "   (26 avril – 15 mai 1986)",
        fontsize=_FONT_SUPTITLE, fontweight="bold", color="white",
        y=1.0 - title_h * 0.35,
        fontfamily="sans-serif",
    )

    # ── Date de l'accident ────────────────────────────────────────────
    accident_date = datetime.datetime(1986, 4, 26, 1, 23)

    # ── Fonction d'update ─────────────────────────────────────────────
    def update(frame_idx):
        nonlocal progress_fill
        t = frame_to_step[frame_idx]
        t_hours = t * dt

        current = accident_date + datetime.timedelta(hours=t_hours)
        date_str = current.strftime("%d %b %Y — %Hh00")

        # Barre de progression
        progress_fill.remove()
        progress_fill = ax_bar.axvspan(0, t_hours, color="#ff4444", alpha=0.5)
        time_text.set_text(
            f"t + {t_hours:>5.0f}h   │   {date_str}   │   "
            f"{_get_wind_phase(t_hours)}"
        )

        # Panneau 1 : particules + heatmap instantanée
        alive = active[t]
        n_alive = int(np.sum(alive))
        if n_alive > 0:
            lons = traj_lon[t, alive]
            lats = traj_lat[t, alive]
            scatter.set_offsets(
                proj.transform_points(data_crs, lons, lats)[:, :2]
            )
        else:
            scatter.set_offsets(np.empty((0, 2)))

        im_inst.set_data(time_density_maps[t])
        info_text.set_text(
            f"Particules : {n_alive:,}\n"
            f"Heure      : t + {t_hours:.0f}h\n"
            f"Vent       : {_get_wind_phase(t_hours)}"
        )

        # Panneau 2 : densité cumulée
        im_cum.set_data(cumulative_maps[t])

        return [scatter, im_inst, im_cum, progress_fill, time_text, info_text]

    # ── Encodage H.264 haute qualité ──────────────────────────────────
    print(f"  Encodage 4K UHD ({n_frames} frames a {fps} fps)...")
    print(f"     Résolution : {int(VISUALIZATION['figsize'][0] * VISUALIZATION['dpi'])}"
          f"×{int(VISUALIZATION['figsize'][1] * VISUALIZATION['dpi'])} px")
    print(f"     Bitrate    : {VISUALIZATION['video_bitrate'] // 1000} Mbps")

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // fps, blit=False)

    writer = FFMpegWriter(
        fps=fps,
        bitrate=VISUALIZATION["video_bitrate"],
        codec="libx264",
        extra_args=[
            "-preset", "slow",        # Meilleure compression
            "-crf", "18",             # Qualité quasi-lossless
            "-pix_fmt", "yuv420p",    # Compatibilité maximale
            "-movflags", "+faststart",  # Streaming-friendly
        ],
        metadata={"title": "Chernobyl Monte Carlo Simulation — 4K",
                  "artist": "MonteCarloSimu",
                  "comment": "3840x2160 120fps"},
    )

    path = os.path.join(VISUALIZATION["save_dir"], filename)
    anim.save(path, writer=writer)
    plt.close(fig)

    # Afficher la taille du fichier
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] Video 4K sauvegardee : {path}  ({size_mb:.1f} Mo)")
    return path
