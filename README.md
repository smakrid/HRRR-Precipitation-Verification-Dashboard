# HRRR-Precipitation-Verification-Dashboard: New York City Testbed

An interactive tool for evaluating how well NOAA's HRRR model predicts rainfall over the New York City metro area. Pick a storm date, and the dashboard pulls forecast data from AWS, observed analyses from NOAA, and rain gauge measurements from airport and urban networks, then shows you exactly where the model got it right, where it missed, and by how much.

Built for the CESSRST-II research group at the City College of New York as part of ongoing work aimed at improving operational forecast performance.

---

## What it does

You select a date and click **Analyze**. Behind the scenes, the tool:

1. Downloads the HRRR 3km forecast from the hrrrzarr archive on AWS S3
2. Downloads the AORC 4km gridded analysis from NOAA's PSL FTP server
3. Attempts to download MRMS 1km radar-gauge QPE from the Iowa State archive
4. Queries ASOS/AWOS airport gauges from the Iowa Environmental Mesonet API
5. Fetches NY-uHMT urban gauge data from the NOAA CREST server
6. Regrids everything onto the same 3km grid, applies a geographic land mask, and computes verification metrics
7. Generates map overlays and downloadable figures

The results appear on an interactive Leaflet map where you can toggle between six precipitation layers (HRRR, AORC, MRMS, and the three pairwise differences), click on gauge stations to see per-site comparisons, and scrub through the storm hour by hour.

The sidebar shows standard verification metrics; bias, MAE, RMSE, correlation, categorical scores at eight thresholds, and Fractions Skill Score across spatial scales. A Summary/Full toggle lets you see just the headline numbers or dive into the complete statistical breakdown.

---

## Why this exists

Operational verification platforms are excellent at aggregate statistics across seasons and regions. But if you want to look at a specific storm and understand *spatially* where HRRR placed the rain versus where it actually fell, that capability doesn't currently exist in existing tools.

This dashboard fills that gap by adding event-level spatial detail, interactive map exploration, and multi-network gauge validation for individual precipitation events.

---

## Quick start

Run these commands in **Anaconda Prompt** (Windows) or **Terminal** (Mac/Linux):

```bash
# Clone the repo
git clone https://github.com/smakrid/HRRR-Precipitation-Verification-Dashboard
cd hrrr-precip-dashboard

# Create conda environment with all core dependencies
conda create -n hrrr-dash python=3.10 numpy pandas xarray scipy matplotlib netCDF4 requests s3fs fastapi uvicorn cartopy shapely -c conda-forge

# Activate the environment
conda activate hrrr-dash

# Optional but recommended for faster regridding and MRMS support
conda install -c conda-forge xesmf esmpy cfgrib eccodes

# Launch the dashboard
python app.py
```

Open [http://localhost:8050](http://localhost:8050) in your browser.

**First run for any date takes 2 to 4 minutes** (downloading data, building the land mask). Subsequent runs for the same date are instant since everything is cached locally.

---

## How the controls work

### Accumulation window
How many hours of precipitation to sum up. A 24-hour window starting at 00Z covers midnight to midnight UTC. Shorter windows (6h, 12h) are useful for isolating specific storm phases.

### Forecast strategy
Which HRRR model run to verify, and at what lead time:

| Strategy | What it uses | Mean lead time | Best for |
|----------|-------------|---------------|----------|
| **Day-Ahead 12Z** | Previous day's 12Z run, forecast hours 13 to 36 | 24.5 hours | Testing operational day-ahead skill (hardest test) |
| **Single Init 00Z** | Same day's 00Z run, forecast hours 1 to 24 | 12.5 hours | Single-run performance with fresher data assimilation |
| **Rolling 1h** | Each hour's most recent run, forecast hour 1 only | 1 hour | Best-case nowcast skill |

Day-Ahead 12Z is the default because it mirrors what forecasters actually do: review the afternoon model run to prepare for tomorrow. The code computes `fhr_start = (event_start - init_time) / 3600 + 1` and `fhr_end = (event_end - init_time) / 3600`, which produces forecast hours 13 through 36 for a 24-hour event window. The +1 accounts for HRRR's hour-ending accumulation convention, where fhr 13 covers the period from 12h to 13h after initialization.

### Statistics area
The region where verification metrics are computed. Ranges from 50km (tight around Manhattan) to 300km (tri-state plus). You can also select "Custom bounds" and either type coordinates or draw a rectangle directly on the map.

The map overlays always extend to the full 600km data extent (roughly DC to Boston) regardless of this setting, so you can see the broader storm context while computing metrics over your area of interest.

---

## Project structure

```
app.py                  FastAPI server and pipeline orchestrator
data_access.py          Data download functions for all sources
config.py               Constants, paths, station coordinates
metrics.py              Verification metric computations
visualization.py        Figure generation and map overlays
land_mask.py            Geographic coastline mask
regridding.py           Bilinear interpolation onto HRRR grid
static/
  index.html            Frontend: Leaflet map, sidebar, controls
requirements.txt
README.md
```

---

## Data sources

| Source | What | Resolution | How it's accessed |
|--------|------|-----------|-------------------|
| **HRRR** | Forecast precipitation (APCP) | 3km Lambert conformal | AWS S3 via s3fs (parallel, 8 workers) |
| **AORC** | Gridded analysis (prate) | 4km regular lat-lon | NOAA PSL FTP, cached locally |
| **MRMS** | Radar+gauge QPE (GaugeCorr) | 1km regular lat-lon | Iowa State archive, cached locally |
| **ASOS/AWOS** | Airport hourly rain gauges | Point observations | Iowa Environmental Mesonet API |
| **NY-uHMT** | Urban rain gauge network | Point observations | NOAA CREST server (17 stations) |

All data downloads automatically on first request and caches locally. No manual data transfers needed.

### A note on gauge independence

HRRR assimilates ASOS/AWOS observations for temperature, wind, moisture, and pressure through its data assimilation cycle, though it does not directly assimilate precipitation gauge values. ASOS precipitation also feeds into both AORC (through Stage IV QPE) and MRMS (direct gauge correction for bias). This means ASOS/AWOS gauge comparisons have some level of circularity across all three products.

The NY-uHMT urban gauge network is the only observation source that is fully independent of HRRR, AORC, and MRMS. None of these products incorporate NY-uHMT data in any way, making it the cleanest ground truth available. Note that some NY-uHMT stations have data quality limitations at certain sites, and the network's data coverage ends in August 2024.

---

## The land mask

HRRR produces precipitation values over ocean grid cells, and bilinear regridding smears AORC values into coastal pixels. Simply checking for zeros doesn't work because both products have legitimate nonzero values over water.

The solution is a geographic mask built from Natural Earth 10m coastline polygons via cartopy and shapely. Each grid cell center is tested against the actual coastline geometry. The land polygons are buffered outward by 0.03 degrees (roughly 3.3km, about one HRRR grid cell width) to capture narrow coastal features like Long Island's south shore that would otherwise get clipped.

The mask is computed once and cached as a `.npy` file. It is applied to HRRR, AORC, and MRMS before any metrics are calculated or overlays are generated.

---

## Verification metrics

The dashboard computes standard verification metrics:

- **Continuous**: multiplicative bias, MAE, RMSE, Pearson correlation
- **Categorical**: POD, FAR, CSI, ETS, frequency bias at eight thresholds (0.254mm through 101.6mm)
- **Spatial**: Fractions Skill Score from 3km to the full domain width, identifying the neighborhood size where the forecast becomes skillful
- **Distribution**: PDF/CDF comparisons with Kolmogorov-Smirnov test
- **Gauge**: per-station bias, MAE, and correlation for both forecast and analysis fields

Thresholds where fewer than 1% of cells exceed the value in both fields are reported as N/A to avoid artificially high FSS scores.

---

## Configuration

Key settings in `config.py`:

- `DOMAIN_CENTER_LAT/LON`: center point for the analysis domain (default: 40.7128, -74.006 for NYC)
- `CONTEXT_EXTENT_KM`: how far the overlay data extends (default: 600km)
- `IEM_NETWORKS`: which state ASOS/AWOS networks to query
- `NY_UHMT_STATIONS`: coordinates and filenames for the 17 urban gauge stations

---

## Deployment

The tool is designed to run on a server with minimal setup:

1. Copy the Python files and `static/index.html` to the server
2. Install the conda environment
3. Set up nginx as a reverse proxy with HTTPS (Let's Encrypt)
4. Create a systemd service for auto-start on reboot
5. Pre-cache a few demo dates for instant loading

No large data transfer is needed. AORC files download automatically from NOAA's FTP server on first request and are cached locally. HRRR streams directly from AWS S3 every time. MRMS caches the same way as AORC.

---

## Known limitations

- NY-uHMT data ends August 2024, and some stations have intermittent quality issues
- MRMS availability varies by date; if less than 75% of hours are available, MRMS is excluded to avoid misleading totals
- Install xESMF for a 10 to 20x speedup on the regridding step (otherwise falls back to scipy which is slower)
- The first land mask build takes a few minutes; subsequent runs use the cached version

---

## Acknowledgments

- CESSRST-II at the City College of New York for research support and NY-uHMT gauge data
- Iowa Environmental Mesonet for the ASOS/AWOS data API and MRMS archive
- NOAA Physical Sciences Laboratory for the AORC 4km precipitation analysis
- The hrrrzarr project (MesoWest/University of Utah) for efficient cloud-based HRRR access

---

## Author

Sebastian Makrides
