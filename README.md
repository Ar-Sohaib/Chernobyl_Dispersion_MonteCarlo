# Chernobyl Dispersion -- Monte Carlo Simulation

Lagrangian particle model simulating the atmospheric dispersion of the Chernobyl radioactive plume (April 26 -- May 16, 1986) across Europe. The simulation uses Monte Carlo methods with real ERA5 reanalysis wind data to produce probability maps and animated visualizations.

---

## Overview

The model tracks thousands of virtual particles released from Reactor No. 4 (51.39N, 30.10E) and advected by wind fields over a 20-day period. Turbulent diffusion is modeled as a Gaussian random walk. Multiple Monte Carlo runs are aggregated to compute spatial probability distributions.

**Key features:**
- Lagrangian particle tracking with configurable emission parameters
- Real ERA5 wind fields at 850 hPa (0.25 deg, hourly) from ECMWF
- Fallback simplified wind model based on historical phase descriptions
- Probability, threshold, cumulative density and instantaneous cloud maps
- Animated heatmap video (H.264, up to 4K 120fps)
- CLI with resolution, particle count and render mode options

---

## Project Structure

```
config.py           Configuration (source, wind, grid, simulation, rendering)
engine.py           Lagrangian Monte Carlo particle engine
wind.py             Simplified uniform wind model (5 historical phases)
wind_era5.py        ERA5 spatiotemporal wind interpolation (bilinear)
aggregation.py      Spatial aggregation (density, probability, threshold)
visualization.py    All rendering (single maps, 4-panel video, proba video)
main.py             CLI entry point (argparse)
download_era5.py    ERA5 data downloader from Copernicus CDS
chart.md            Theoretical background and methodology
assets/             Background map image (Natural Earth Hypsometric)
```

---

## Requirements

- Python 3.9+
- NumPy, Matplotlib, Cartopy, SciPy
- xarray, netCDF4 (for ERA5 wind data)
- FFmpeg (for video encoding)
- cdsapi (only for downloading ERA5 data)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib cartopy scipy xarray netcdf4 cdsapi
```

FFmpeg must be installed separately:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## ERA5 Wind Data

The simulation uses ERA5 reanalysis data from ECMWF (u and v wind components at 850 hPa). To download:

1. Create an account at https://cds.climate.copernicus.eu
2. Get your Personal Access Token from your profile
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR_TOKEN>
   ```
4. Run the downloader:
   ```bash
   python download_era5.py
   ```

This downloads ~27 MB of hourly wind data covering the simulation domain (-12W to 75E, 33N to 72N) for April 26 -- May 16, 1986.

To use the simplified wind model instead, set `WIND["mode"] = "simplified"` in `config.py`.

---

## Usage

```bash
# Probability map (default)
python main.py

# Animated probability heatmap video
python main.py -m pv

# 4-panel video (cloud, density, probability, threshold)
python main.py -m all

# Specific render modes
python main.py -m instant       # Instantaneous cloud
python main.py -m cumul         # Cumulative density
python main.py -m threshold     # Threshold exceedance

# Options
python main.py -m pv -r 4k           # 4K resolution
python main.py -m pv -r 1080p        # 1080p resolution
python main.py -m pv -n 20000        # 20000 particles
python main.py --help                 # Full help
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --mode` | Render mode (pv, all, proba, instant, cumul, threshold) | probability |
| `-r, --resolution` | Output resolution (720p, 1080p, 2k, 4k) | 2k |
| `-n, --particles` | Number of particles to simulate | 8000 |

---

## Output

Generated files are saved to `output/`:
- `probability_map.png` -- Static probability map
- `chernobyl_proba_video.mp4` -- Animated probability heatmap
- `chernobyl_simulation.mp4` -- 4-panel animated video

---

## Configuration

All parameters are centralized in `config.py`:
- **SOURCE**: Emission location, rate, duration
- **WIND_PHASES**: Simplified historical wind phases
- **WIND**: Wind mode selection (ERA5 or simplified)
- **DIFFUSION**: Turbulent diffusion coefficients
- **GRID**: Spatial domain and resolution
- **SIMULATION**: Particle count, time step, duration, Monte Carlo runs
- **VISUALIZATION**: DPI, resolution, colormap, video settings

---

## License

MIT
