import os, sys, types
os.environ['MPLBACKEND'] = 'Agg'
sys.modules['tkinter'] = types.ModuleType('tkinter')
sys.modules['_tkinter'] = types.ModuleType('_tkinter')
"""
=============================================================================
HRRR vs AORC Verification Dashboard — Web Application Backend
=============================================================================
FastAPI server that wraps the existing verification pipeline and serves
results to an interactive browser-based dashboard.

HOW TO RUN:
    cd web_app
    python app.py

    Then open http://localhost:8050 in your browser.

ARCHITECTURE:
    - This file (app.py) is the web server
    - It imports the same modules you run in Spyder (data_access, regridding,
      metrics, visualization)
    - The frontend (static/index.html) sends requests, this server runs the
      pipeline and returns JSON results + generated figure images
    - Results are cached on disk so repeat queries for the same event are instant

DEPENDENCIES (beyond what the Spyder pipeline already needs):
    pip install fastapi uvicorn

Author: Sebastian Makrides
=============================================================================
"""

import os
import sys
import json
import hashlib
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Force matplotlib to use non-GUI backend (must be before any matplotlib import)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ---------------------------------------------------------------------------
# Path setup — add the parent directory so we can import the pipeline modules
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(APP_DIR))

# Pipeline module imports (same code that runs in Spyder)
from config import (
    DOMAIN_BBOX, DOMAIN_CENTER_LAT, DOMAIN_CENTER_LON,
    DOMAIN_SIZE_KM, ACCUM_WINDOWS, PRECIP_THRESHOLDS,
    NYC_TIGHT_BBOX, SOM_BASE_DIR, AORC_LOCAL_DIR,
    HRRR_FORECAST_STRATEGY, HRRR_INIT_HOUR, HRRR_FXX,
    AORC_FTP_HOST,
)
from data_access import (
    find_aorc_local, download_aorc_ftp, load_aorc,
    download_hrrr_hrrrzarr,
    download_asos_gauges, accumulate_gauge_precip,
    download_nyuhmt_gauges,
    download_mrms_qpe,
)
from regridding import (
    regrid_aorc_to_hrrr, accumulate_precip, save_regridded,
)
from metrics import (
    compute_continuous_metrics, compute_categorical_metrics,
    compute_all_metrics, print_metrics_report,
)
from visualization import generate_all_plots

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("dashboard")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
CACHE_DIR = APP_DIR / "cache"
FIGURES_DIR = APP_DIR / "figures"
STATIC_DIR = APP_DIR / "static"

CACHE_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="HRRR vs AORC Verification Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated figures at /figures/<filename>
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
# Serve frontend at /static/<filename>
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    date: str               # "YYYY-MM-DD"
    accum_hours: int = 24   # 6, 12, 18, or 24
    strategy: str = "day_ahead_12z"
    domain_km: int = 100    # Analysis domain size (50, 100, 150, 200, 300)
    reference: str = "aorc" # Reference dataset: "aorc" or "mrms"
    context_km: int = 600   # Overlay extent in km (broader storm context)
    custom_bbox: Optional[dict] = None  # Optional: {south, north, west, east} overrides domain_km

def compute_dynamic_bbox(center_lat, center_lon, domain_km):
    """Compute bounding box from center point and domain size in km."""
    from math import radians, cos
    lat_offset = (domain_km / 2) / 111.0
    lon_offset = (domain_km / 2) / (111.0 * cos(radians(center_lat)))
    return {
        'south': round(center_lat - lat_offset, 4),
        'north': round(center_lat + lat_offset, 4),
        'west': round(center_lon - lon_offset, 4),
        'east': round(center_lon + lon_offset, 4),
    }

# ---------------------------------------------------------------------------
# Helper: convert numpy types to JSON-serializable Python types
# ---------------------------------------------------------------------------
def jsonify(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj

# ---------------------------------------------------------------------------
# Helper: generate a cache key for an analysis run
# ---------------------------------------------------------------------------
def cache_key(date_str, accum_hours, strategy):
    raw = f"{date_str}_{accum_hours}h_{strategy}"
    return hashlib.md5(raw.encode()).hexdigest()

def _utc_to_et(dt_utc):
    """Convert naive UTC datetime to Eastern Time string (handles EDT/EST)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        try:
            from backports.zoneinfo import ZoneInfo
        except ImportError:
            # Fallback: assume EDT (UTC-4) for warm-season events
            return (dt_utc - timedelta(hours=4)).strftime('%I:%M %p ET')
    eastern = ZoneInfo('America/New_York')
    from datetime import timezone as tz
    aware = dt_utc.replace(tzinfo=tz.utc).astimezone(eastern)
    label = 'EDT' if aware.dst() else 'EST'
    return aware.strftime(f'%I:%M %p {label}')


def extract_hourly_series(hrrr_ds, aorc_on_hrrr, hrrr_precip_var,
                          gauge_hourly_df, gauge_accum_df,
                          lat_grid, lon_grid, event_start, accum_hours):
    """
    Extract hour-by-hour time series for the frontend slider and time series plot.
    
    For each hour of the event, this computes:
    - Domain-mean HRRR and AORC precipitation (instantaneous and cumulative)
    - Per-station HRRR, AORC, and observed gauge values (instantaneous and cumulative)
    
    Returns a dict suitable for JSON serialization.
    """
    hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
    aorc_times = pd.DatetimeIndex(aorc_on_hrrr.time.values)
    
    # Precompute nearest grid cell indices for each gauge station
    # Use cos(lat) correction so longitude distances aren't overweighted at 40°N
    station_indices = {}
    if len(gauge_accum_df) > 0:
        lat_arr = np.asarray(lat_grid, dtype=float)
        lon_arr = np.asarray(lon_grid, dtype=float)
        for _, stn in gauge_accum_df.iterrows():
            sid = stn['station']
            s_lat, s_lon = stn['lat'], stn['lon']
            cos_lat = np.cos(np.radians(s_lat))
            if lat_arr.ndim == 2:
                dist = np.sqrt((lat_arr - s_lat)**2 + (cos_lat * (lon_arr - s_lon))**2)
                idx = np.unravel_index(np.argmin(dist), dist.shape)
            else:
                lat_idx = np.argmin(np.abs(lat_arr - s_lat))
                lon_idx = np.argmin(np.abs(lon_arr - s_lon))
                idx = (lat_idx, lon_idx)
            station_indices[sid] = {
                'idx': idx,
                'lat': float(s_lat),
                'lon': float(s_lon),
            }
    
    # Build hourly series
    hours_meta = []
    domain_hrrr_hourly = []
    domain_aorc_hourly = []
    station_hourly = {sid: {'obs': [], 'hrrr': [], 'aorc': []} for sid in station_indices}
    
    for h in range(accum_hours):
        # HRRR uses hour-ending stamps: fhr N covers (N-1)h to Nh after init
        # Match by looking for timestamps AT the hour-end
        hour_utc = event_start + timedelta(hours=h)
        hour_end = event_start + timedelta(hours=h + 1)
        
        hours_meta.append({
            'hour': h,
            'utc': hour_utc.isoformat(),
            'et': _utc_to_et(hour_utc),
        })
        
        # HRRR: hour-ending convention — the timestamp IS the end of the accumulation period
        hrrr_hour_mask = (hrrr_times > pd.Timestamp(hour_utc)) & (hrrr_times <= pd.Timestamp(hour_end))
        if hrrr_hour_mask.any():
            hrrr_slice = hrrr_ds[hrrr_precip_var].isel(time=hrrr_hour_mask).sum(dim='time').values
        else:
            hrrr_slice = np.zeros_like(lat_grid)
        
        # AORC: same convention
        aorc_hour_mask = (aorc_times > pd.Timestamp(hour_utc)) & (aorc_times <= pd.Timestamp(hour_end))
        if aorc_hour_mask.any():
            aorc_slice = aorc_on_hrrr['aorc_precip'].isel(time=aorc_hour_mask).sum(dim='time').values
        else:
            aorc_slice = np.zeros_like(lat_grid)
        
        # Domain means (mean is correct here — median of a sparse precip field is
        # usually zero because most grid cells are dry at any given hour)
        domain_hrrr_hourly.append(round(float(np.nanmean(hrrr_slice)), 2))
        domain_aorc_hourly.append(round(float(np.nanmean(aorc_slice)), 2))
        
        # Per-station extraction
        for sid, info in station_indices.items():
            idx = info['idx']
            station_hourly[sid]['hrrr'].append(round(float(hrrr_slice[idx]), 2))
            station_hourly[sid]['aorc'].append(round(float(aorc_slice[idx]), 2))
            
            # Gauge observation for this hour
            if len(gauge_hourly_df) > 0:
                obs_mask = (
                    (gauge_hourly_df['station'] == sid) &
                    (gauge_hourly_df['valid_time'] >= pd.Timestamp(hour_utc, tz='UTC')) &
                    (gauge_hourly_df['valid_time'] < pd.Timestamp(hour_end, tz='UTC'))
                )
                obs_val = gauge_hourly_df.loc[obs_mask, 'precip_mm'].sum()
                station_hourly[sid]['obs'].append(round(float(obs_val), 2))
            else:
                station_hourly[sid]['obs'].append(0.0)
    
    # Build cumulative arrays
    domain_hrrr_cumul = list(np.cumsum(domain_hrrr_hourly).round(2))
    domain_aorc_cumul = list(np.cumsum(domain_aorc_hourly).round(2))
    
    stations_out = {}
    for sid, info in station_indices.items():
        obs_c = list(np.cumsum(station_hourly[sid]['obs']).round(2))
        hrrr_c = list(np.cumsum(station_hourly[sid]['hrrr']).round(2))
        aorc_c = list(np.cumsum(station_hourly[sid]['aorc']).round(2))
        stations_out[sid] = {
            'lat': info['lat'],
            'lon': info['lon'],
            'obs_hourly': station_hourly[sid]['obs'],
            'hrrr_hourly': station_hourly[sid]['hrrr'],
            'aorc_hourly': station_hourly[sid]['aorc'],
            'obs_cumul': obs_c,
            'hrrr_cumul': hrrr_c,
            'aorc_cumul': aorc_c,
        }
    
    return {
        'hours': hours_meta,
        'domain_mean': {
            'hrrr_hourly': domain_hrrr_hourly,
            'aorc_hourly': domain_aorc_hourly,
            'hrrr_cumul': domain_hrrr_cumul,
            'aorc_cumul': domain_aorc_cumul,
        },
        'stations': stations_out,
    }

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the main dashboard page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "Frontend not found. Place index.html in static/"}, 404)
    return FileResponse(index_path)

@app.get("/api/config")
async def get_config():
    """Return the current domain configuration so the frontend can display it."""
    return {
        "domain_center": {"lat": DOMAIN_CENTER_LAT, "lon": DOMAIN_CENTER_LON},
        "domain_size_km": DOMAIN_SIZE_KM,
        "domain_bbox": DOMAIN_BBOX,
        "nyc_tight_bbox": NYC_TIGHT_BBOX,
        "accum_windows": ACCUM_WINDOWS,
        "precip_thresholds": PRECIP_THRESHOLDS,
        "hrrr_strategy": HRRR_FORECAST_STRATEGY,
        "aorc_local_dir": str(AORC_LOCAL_DIR),
    }

@app.get("/api/available-dates")
async def get_available_dates():
    """
    Scan the AORC local directory and return a list of dates that have
    data files available. This lets the frontend show a date picker with
    only valid dates highlighted.
    """
    aorc_dir = Path(AORC_LOCAL_DIR)
    dates = []

    if aorc_dir.exists():
        for f in sorted(aorc_dir.glob("prate.aorc.*.nc")):
            # Extract date from filename: prate.aorc.YYYYMMDD.nc
            parts = f.stem.split(".")
            if len(parts) >= 3:
                date_str = parts[2]
                try:
                    dt = datetime.strptime(date_str, "%Y%m%d")
                    dates.append(dt.strftime("%Y-%m-%d"))
                except ValueError:
                    continue

    return {"count": len(dates), "dates": dates}

@app.post("/api/analyze")
async def run_analysis(req: AnalyzeRequest):
    """
    Run the full verification pipeline for a given event date.

    This is the main workhorse endpoint. It:
      1. Loads AORC from local files
      2. Downloads HRRR from hrrrzarr (S3)
      3. Downloads gauge data from IEM
      4. Regrids AORC onto the HRRR grid
      5. Computes all verification metrics
      6. Generates all figures
      7. Returns everything as JSON

    The HRRR download is the slowest step (~1-3 minutes on first run).
    Results are cached so subsequent requests for the same event are instant.
    """
    # --- Parse and validate date ---
    try:
        event_start = datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD (e.g., 2024-09-29).")

    if req.accum_hours not in [6, 12, 18, 24]:
        raise HTTPException(400, f"Accumulation window must be 6, 12, 18, or 24 hours (got {req.accum_hours}).")

    if req.domain_km not in [50, 100, 150, 200, 300] and req.custom_bbox is None:
        raise HTTPException(400, f"Domain size must be 50, 100, 150, 200, or 300 km (got {req.domain_km}).")

    if req.reference not in ["aorc", "mrms"]:
        raise HTTPException(400, f"Reference must be 'aorc' or 'mrms' (got {req.reference}).")

    # Compute bounding boxes
    domain_km = req.domain_km
    context_km = max(req.context_km, domain_km)  # context must be >= analysis domain

    if req.custom_bbox is not None:
        cb = req.custom_bbox
        # Validate bounds
        if not all(k in cb for k in ('south', 'north', 'west', 'east')):
            raise HTTPException(400, "custom_bbox must have south, north, west, east keys")
        if cb['north'] <= cb['south'] or cb['east'] <= cb['west']:
            raise HTTPException(400, "custom_bbox: north must be > south, east must be > west")
        if cb['north'] - cb['south'] > 8 or cb['east'] - cb['west'] > 10:
            raise HTTPException(400, "custom_bbox too large (max ~8° lat × 10° lon)")
        analysis_bbox = {
            'south': float(cb['south']),
            'north': float(cb['north']),
            'west': float(cb['west']),
            'east': float(cb['east']),
        }
        # Estimate domain_km from bbox for labeling
        from math import radians, cos
        lat_span = analysis_bbox['north'] - analysis_bbox['south']
        domain_km = int(lat_span * 111)
        domain_label = f"custom ({analysis_bbox['south']:.2f}–{analysis_bbox['north']:.2f}N, {analysis_bbox['west']:.2f}–{analysis_bbox['east']:.2f}W)"
        logger.info(f"Using custom bbox: {analysis_bbox} (~{domain_km}km)")
    else:
        analysis_bbox = compute_dynamic_bbox(DOMAIN_CENTER_LAT, DOMAIN_CENTER_LON, domain_km)
        domain_label = f"{domain_km}km domain"

    context_bbox = compute_dynamic_bbox(DOMAIN_CENTER_LAT, DOMAIN_CENTER_LON, context_km)
    # Add buffer for regridding edge effects
    download_bbox = {
        'south': context_bbox['south'] - 0.1,
        'north': context_bbox['north'] + 0.1,
        'west': context_bbox['west'] - 0.1,
        'east': context_bbox['east'] + 0.1,
    }
    ref_label = "AORC+MRMS"

    event_end = event_start + timedelta(hours=req.accum_hours)
    event_label = f"{event_start.strftime('%Y-%m-%d')} NYC {req.accum_hours}h ({domain_label})"
    bbox_str = f"{analysis_bbox['south']:.2f}_{analysis_bbox['north']:.2f}_{analysis_bbox['west']:.2f}_{analysis_bbox['east']:.2f}"
    ckey = cache_key(f"{req.date}_{bbox_str}_{context_km}ctx_dual", req.accum_hours, req.strategy)

    logger.info(f"Analyze request: {event_label} | strategy={req.strategy} | context={context_km}km")

    # --- Check cache ---
    cache_file = CACHE_DIR / f"{ckey}.json"
    if cache_file.exists():
        logger.info(f"Cache hit: {ckey}")
        with open(cache_file, "r") as f:
            return json.load(f)

    # --- Run the pipeline ---
    try:
        # Phase 1a: AORC (always — primary reference)
        logger.info("Phase 1a: Loading AORC...")
        aorc_files = find_aorc_local(event_start, event_end)
        if not aorc_files:
            logger.info("No local AORC files, trying FTP download with local caching...")
            aorc_files = download_aorc_ftp(event_start, event_end)
        if not aorc_files:
            raise HTTPException(404,
                f"No AORC data available for {req.date}. "
                f"Check that prate.aorc.{event_start.strftime('%Y%m%d')}.nc exists in {AORC_LOCAL_DIR}, "
                f"or that FTP access to {AORC_FTP_HOST} is available.")
        aorc_ds_raw = load_aorc(aorc_files, bbox=download_bbox,
                                 start_dt=event_start, end_dt=event_end)

        # Phase 1a2: MRMS (optional — graceful failure)
        mrms_ds_raw = None
        mrms_available = False
        try:
            logger.info("Phase 1a2: Downloading MRMS QPE (optional)...")
            # Use analysis bbox (not full context) for MRMS — at 1km resolution,
            # the full 600km context would be ~360,000 cells and crash scipy griddata
            mrms_bbox = {
                'south': analysis_bbox['south'] - 0.2,
                'north': analysis_bbox['north'] + 0.2,
                'west': analysis_bbox['west'] - 0.2,
                'east': analysis_bbox['east'] + 0.2,
            }
            mrms_ds = download_mrms_qpe(event_start, event_end, bbox=mrms_bbox)
            if mrms_ds is not None and hasattr(mrms_ds, 'time') and len(mrms_ds.time) > 0:
                mrms_ds_raw = mrms_ds.rename({'mrms_precip': 'prate'})
                if 'lat' in mrms_ds_raw.dims:
                    mrms_ds_raw = mrms_ds_raw.rename({'lat': 'latitude', 'lon': 'longitude'})
                mrms_available = True
                logger.info(f"  MRMS loaded: {len(mrms_ds_raw.time)} hours")
            else:
                logger.info("  MRMS: no data returned (archive may not cover this date)")
        except ImportError:
            logger.warning("  MRMS skipped: cfgrib not installed (conda install -c conda-forge cfgrib eccodes)")
        except Exception as e:
            logger.warning(f"  MRMS skipped (non-fatal): {e}")

        # Phase 1b: HRRR
        logger.info("Phase 1b: Downloading HRRR from hrrrzarr...")
        hrrr_ds = download_hrrr_hrrrzarr(
            event_start, event_end,
            strategy=req.strategy,
            init_hour=HRRR_INIT_HOUR,
            fxx=HRRR_FXX,
            bbox=download_bbox,
        )

        # Phase 1c: Gauges (ASOS + NY-uHMT)
        logger.info("Phase 1c: Downloading gauge data...")
        try:
            gauge_hourly_df = download_asos_gauges(event_start, event_end)
        except Exception as e:
            logger.warning(f"ASOS download failed: {e}")
            gauge_hourly_df = pd.DataFrame()
        
        # NY-uHMT dense urban network
        try:
            uhmt_hourly_df = download_nyuhmt_gauges(event_start, event_end)
            if len(uhmt_hourly_df) > 0:
                # Ensure ASOS has network column
                if len(gauge_hourly_df) > 0 and 'network' not in gauge_hourly_df.columns:
                    gauge_hourly_df['network'] = 'ASOS'
                # Merge both networks
                gauge_hourly_df = pd.concat([gauge_hourly_df, uhmt_hourly_df], ignore_index=True)
                logger.info(f"Merged gauges: {gauge_hourly_df['station'].nunique()} total stations "
                           f"(ASOS + NY-uHMT)")
        except Exception as e:
            logger.warning(f"NY-uHMT download failed (non-fatal): {e}")
        
        if len(gauge_hourly_df) > 0:
            gauge_accum_df = accumulate_gauge_precip(
                gauge_hourly_df, event_start, req.accum_hours
            )
        else:
            gauge_accum_df = pd.DataFrame()

        # Phase 2: Regridding
        logger.info("Phase 2: Regridding AORC → HRRR grid...")
        aorc_on_hrrr = regrid_aorc_to_hrrr(aorc_ds_raw, hrrr_ds, method='bilinear')

        mrms_on_hrrr = None
        if mrms_available and mrms_ds_raw is not None:
            logger.info("Phase 2b: Regridding MRMS → HRRR grid...")
            try:
                mrms_on_hrrr = regrid_aorc_to_hrrr(mrms_ds_raw, hrrr_ds, method='bilinear')
            except Exception as e:
                logger.warning(f"MRMS regridding failed (non-fatal): {e}")
                mrms_available = False

        # Find precip variable name in HRRR dataset
        hrrr_precip_var = None
        for v in ['tp', 'APCP_surface', 'APCP']:
            if v in hrrr_ds.data_vars:
                hrrr_precip_var = v
                break

        hrrr_accum = accumulate_precip(
            hrrr_ds, hrrr_precip_var, event_start, req.accum_hours
        ).values
        aorc_accum = accumulate_precip(
            aorc_on_hrrr, 'aorc_precip', event_start, req.accum_hours
        ).values

        mrms_accum = None
        if mrms_available and mrms_on_hrrr is not None:
            try:
                mrms_accum = accumulate_precip(
                    mrms_on_hrrr, 'aorc_precip', event_start, req.accum_hours
                ).values
                logger.info(f"  MRMS accumulated: shape {mrms_accum.shape}")
            except Exception as e:
                logger.warning(f"MRMS accumulation failed (non-fatal): {e}")
                mrms_available = False

        lat_grid = hrrr_ds['latitude'].values if 'latitude' in hrrr_ds.coords else hrrr_ds['lat'].values
        lon_grid = hrrr_ds['longitude'].values if 'longitude' in hrrr_ds.coords else hrrr_ds['lon'].values

        # Apply land mask to ALL fields
        try:
            from land_mask import get_land_mask, apply_land_mask
            logger.info("Applying land mask (Natural Earth 10m coastlines)...")
            land_mask = get_land_mask(lat_grid, lon_grid)
            hrrr_accum = apply_land_mask(hrrr_accum, land_mask)
            aorc_accum = apply_land_mask(aorc_accum, land_mask)
            if mrms_accum is not None:
                mrms_accum = apply_land_mask(mrms_accum, land_mask)
            logger.info(f"  Land mask applied: {land_mask.sum()} land / "
                       f"{(~land_mask).sum()} ocean cells masked")
        except ImportError:
            logger.warning("land_mask module not found — skipping ocean masking.")
        except Exception as e:
            logger.warning(f"Land mask failed (non-fatal): {e}")

        # Subset to analysis domain for metrics (keep full context for overlays)
        # The downloaded data covers context_km; metrics only use domain_km
        if context_km > domain_km:
            logger.info(f"Subsetting from {context_km}km context to {domain_km}km analysis domain for metrics...")
            if lat_grid.ndim == 2:
                lat_mask = (lat_grid >= analysis_bbox['south']) & (lat_grid <= analysis_bbox['north'])
                lon_mask = (lon_grid >= analysis_bbox['west']) & (lon_grid <= analysis_bbox['east'])
                domain_mask = lat_mask & lon_mask
                y_any = np.any(domain_mask, axis=1)
                x_any = np.any(domain_mask, axis=0)
                y_sl = slice(np.argmax(y_any), len(y_any) - np.argmax(y_any[::-1]))
                x_sl = slice(np.argmax(x_any), len(x_any) - np.argmax(x_any[::-1]))
            else:
                y_sl = slice(
                    np.searchsorted(lat_grid, analysis_bbox['south']),
                    np.searchsorted(lat_grid, analysis_bbox['north'])
                )
                x_sl = slice(
                    np.searchsorted(lon_grid, analysis_bbox['west']),
                    np.searchsorted(lon_grid, analysis_bbox['east'])
                )
            hrrr_accum_analysis = hrrr_accum[y_sl, x_sl]
            aorc_accum_analysis = aorc_accum[y_sl, x_sl]
            lat_analysis = lat_grid[y_sl, x_sl] if lat_grid.ndim == 2 else lat_grid[y_sl]
            lon_analysis = lon_grid[y_sl, x_sl] if lon_grid.ndim == 2 else lon_grid[x_sl]
            logger.info(f"  Context grid: {hrrr_accum.shape}, Analysis grid: {hrrr_accum_analysis.shape}")
        else:
            hrrr_accum_analysis = hrrr_accum
            aorc_accum_analysis = aorc_accum
            lat_analysis = lat_grid
            lon_analysis = lon_grid

        # Phase 2b: Extract hourly time series for slider + time series plot
        # Subset the datasets to analysis domain so station indices are consistent
        logger.info("Phase 2b: Extracting hourly time series...")
        if context_km > domain_km and hasattr(hrrr_ds, 'isel'):
            # Subset xarray datasets to analysis domain for consistent indexing
            try:
                hrrr_ds_analysis = hrrr_ds.isel(y=y_sl, x=x_sl) if 'y' in hrrr_ds.dims else hrrr_ds
                aorc_analysis = aorc_on_hrrr.isel(y=y_sl, x=x_sl) if 'y' in aorc_on_hrrr.dims else aorc_on_hrrr
            except Exception:
                hrrr_ds_analysis = hrrr_ds
                aorc_analysis = aorc_on_hrrr
        else:
            hrrr_ds_analysis = hrrr_ds
            aorc_analysis = aorc_on_hrrr
        hourly_data = extract_hourly_series(
            hrrr_ds_analysis, aorc_analysis, hrrr_precip_var,
            gauge_hourly_df, gauge_accum_df,
            lat_analysis, lon_analysis, event_start, req.accum_hours
        )

        # Phase 3: Metrics (computed on analysis domain only)
        logger.info("Phase 3: Computing metrics...")
        all_metrics = compute_all_metrics(
            hrrr_accum=hrrr_accum_analysis,
            aorc_accum=aorc_accum_analysis,
            gauge_accum_df=gauge_accum_df if len(gauge_accum_df) > 0 else None,
            hrrr_lat=lat_analysis,
            hrrr_lon=lon_analysis,
            thresholds=PRECIP_THRESHOLDS,
        )
        print_metrics_report(all_metrics, event_label=event_label)

        # Phase 4: Generate figures (use analysis-domain arrays for static plots)
        logger.info("Phase 4: Generating figures...")
        event_fig_dir = FIGURES_DIR / ckey
        event_fig_dir.mkdir(exist_ok=True)

        overlays_meta = generate_all_plots(
            hrrr_accum=hrrr_accum_analysis,
            aorc_accum=aorc_accum_analysis,
            lat=lat_analysis,
            lon=lon_analysis,
            metrics_dict=all_metrics,
            gauge_accum_df=gauge_accum_df if len(gauge_accum_df) > 0 else None,
            event_label=event_label,
            accum_hours=req.accum_hours,
            save_dir=str(event_fig_dir),
            show=False,
        )

        # Generate EXTENDED overlays covering the full context extent
        # These are the transparent PNGs for the Leaflet map (broader view)
        logger.info(f"Generating extended {context_km}km overlays for Leaflet map...")
        from visualization import generate_map_overlay, get_precip_cmap, get_diff_cmap
        prefix = event_label.replace(" ", "_").replace("/", "-")
        cmap_p, norm_p = get_precip_cmap()
        cmap_d, norm_d = get_diff_cmap()

        overlays_meta = {}

        # HRRR overlay
        overlays_meta['hrrr'] = generate_map_overlay(
            hrrr_accum, lat_grid, lon_grid, cmap_p, norm_p,
            save_path=str(event_fig_dir / f"{prefix}_overlay_hrrr.png"), alpha=0.85)

        # AORC overlay
        overlays_meta['aorc'] = generate_map_overlay(
            aorc_accum, lat_grid, lon_grid, cmap_p, norm_p,
            save_path=str(event_fig_dir / f"{prefix}_overlay_aorc.png"), alpha=0.85)

        # HRRR minus AORC diff
        diff_ha = hrrr_accum - aorc_accum
        overlays_meta['diff_ha'] = generate_map_overlay(
            diff_ha, lat_grid, lon_grid, cmap_d, norm_d,
            save_path=str(event_fig_dir / f"{prefix}_overlay_diff_ha.png"), alpha=0.85)

        # MRMS overlay + diffs (if available)
        if mrms_accum is not None:
            overlays_meta['mrms'] = generate_map_overlay(
                mrms_accum, lat_grid, lon_grid, cmap_p, norm_p,
                save_path=str(event_fig_dir / f"{prefix}_overlay_mrms.png"), alpha=0.85)

            diff_hm = hrrr_accum - mrms_accum
            overlays_meta['diff_hm'] = generate_map_overlay(
                diff_hm, lat_grid, lon_grid, cmap_d, norm_d,
                save_path=str(event_fig_dir / f"{prefix}_overlay_diff_hm.png"), alpha=0.85)

            diff_am = aorc_accum - mrms_accum
            overlays_meta['diff_am'] = generate_map_overlay(
                diff_am, lat_grid, lon_grid, cmap_d, norm_d,
                save_path=str(event_fig_dir / f"{prefix}_overlay_diff_am.png"), alpha=0.85)

            logger.info("  Generated 6 overlays: HRRR, AORC, MRMS, H-A, H-M, A-M")
        else:
            logger.info("  Generated 3 overlays: HRRR, AORC, H-A (MRMS unavailable)")

        # Close all matplotlib figures to prevent memory leaks
        import matplotlib.pyplot as plt
        plt.close('all')

        # Generate hourly time series plot (needs hourly_data from phase 2b)
        from visualization import plot_hourly_timeseries
        prefix = event_label.replace(" ", "_").replace("/", "-")
        logger.info("Generating hourly time series plot...")
        plot_hourly_timeseries(
            hourly_data, event_label=event_label,
            save_path=str(event_fig_dir / f"{prefix}_timeseries.png")
        )
        plt.close('all')

        # Build the figure URL list
        figure_files = sorted(event_fig_dir.glob("*.png"))
        figure_urls = {}
        for fig_path in figure_files:
            name = fig_path.stem
            url = f"/figures/{ckey}/{fig_path.name}"
            if name.endswith("_overlay_hrrr") or name.endswith("_overlay_aorc") or name.endswith("_overlay_diff") or "_overlay_" in name:
                continue  # Skip overlays — handled separately below
            elif name.endswith("_hrrr"):
                figure_urls["hrrr_map"] = url
            elif name.endswith("_aorc"):
                figure_urls["aorc_map"] = url
            elif name.endswith("_diff"):
                figure_urls["diff_map"] = url
            elif name.endswith("_pdf"):
                figure_urls["pdf"] = url
            elif name.endswith("_cdf"):
                figure_urls["cdf"] = url
            elif name.endswith("_gauge_bias"):
                figure_urls["gauge_bias"] = url
            elif name.endswith("_stats_panel"):
                figure_urls["stats_panel"] = url
            elif name.endswith("_fss"):
                figure_urls["fss"] = url
            elif name.endswith("_timeseries"):
                figure_urls["timeseries"] = url

        # Build overlay URLs for Leaflet interactive map layers
        overlay_urls = {}
        if overlays_meta:
            for layer_name, meta in overlays_meta.items():
                overlay_urls[layer_name] = {
                    "url": f"/figures/{ckey}/{Path(meta['path']).name}",
                    "bounds": meta['bounds'],
                }

        # Build gauge data for the interactive map
        # The per-station HRRR/AORC values are in all_metrics['gauge_detail']
        # which is a DataFrame returned by compute_gauge_metrics()
        
        # Build station→network lookup from the hourly data
        station_network = {}
        if len(gauge_hourly_df) > 0 and 'network' in gauge_hourly_df.columns:
            for stn, net in gauge_hourly_df.groupby('station')['network'].first().items():
                station_network[stn] = net
        
        gauge_data = []
        gauge_detail = all_metrics.get('gauge_detail', pd.DataFrame())
        if len(gauge_detail) > 0:
            for _, row in gauge_detail.iterrows():
                stn = row.get("station", "")
                gauge_data.append({
                    "station": stn,
                    "lat": float(row["gauge_lat"]),
                    "lon": float(row["gauge_lon"]),
                    "observed_mm": float(row["observed_mm"]),
                    "hrrr_mm": float(row["hrrr_mm"]),
                    "aorc_mm": float(row["aorc_mm"]),
                    "hrrr_bias_mm": float(row["hrrr_error_mm"]),
                    "aorc_bias_mm": float(row["aorc_error_mm"]),
                    "qc_flag": row.get("qc_flag", ""),
                    "network": station_network.get(stn, "ASOS"),
                })
        elif len(gauge_accum_df) > 0:
            for _, row in gauge_accum_df.iterrows():
                stn = row.get("station", "")
                gauge_data.append({
                    "station": stn,
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "observed_mm": round(float(row["total_precip_mm"]), 1),
                    "hrrr_mm": None,
                    "aorc_mm": None,
                    "hrrr_bias_mm": None,
                    "aorc_bias_mm": None,
                    "qc_flag": row.get("qc_flag", ""),
                    "network": station_network.get(stn, "ASOS"),
                })

        # Build a JSON-safe version of metrics (exclude DataFrames and large arrays)
        metrics_for_json = {}
        for k, v in all_metrics.items():
            if k == 'gauge_detail':
                continue  # Sent separately as gauge_data
            elif k == 'spatial':
                continue  # Large 2D arrays, not needed in JSON
            elif k == 'categorical' and isinstance(v, pd.DataFrame):
                # Convert categorical DataFrame to dict keyed by threshold
                cat_dict = {}
                for _, row in v.iterrows():
                    thresh = row.get('threshold', row.name)
                    cat_dict[f"{thresh}mm"] = jsonify(row.to_dict())
                metrics_for_json['categorical_multi'] = cat_dict
            else:
                metrics_for_json[k] = jsonify(v)

        # Assemble the response
        result = {
            "event": {
                "date": req.date,
                "label": event_label,
                "start_utc": event_start.isoformat(),
                "end_utc": event_end.isoformat(),
                "start_et": _utc_to_et(event_start),
                "end_et": _utc_to_et(event_end),
                "accum_hours": req.accum_hours,
                "strategy": req.strategy,
                "reference": "aorc",
                "mrms_available": mrms_accum is not None,
            },
            "domain": {
                "center_lat": DOMAIN_CENTER_LAT,
                "center_lon": DOMAIN_CENTER_LON,
                "size_km": domain_km,
                "context_km": context_km,
                "bbox": analysis_bbox,
                "context_bbox": context_bbox,
                "nyc_tight_bbox": NYC_TIGHT_BBOX,
                "grid_shape": list(hrrr_accum_analysis.shape),
            },
            "metrics": metrics_for_json,
            "gauges": gauge_data,
            "figures": figure_urls,
            "overlays": overlay_urls,
            "grid_summary": {
                "hrrr_mean_mm": round(float(np.nanmean(hrrr_accum_analysis)), 2),
                "hrrr_max_mm": round(float(np.nanmax(hrrr_accum_analysis)), 2),
                "aorc_mean_mm": round(float(np.nanmean(aorc_accum_analysis)), 2),
                "aorc_max_mm": round(float(np.nanmax(aorc_accum_analysis)), 2),
                "n_cells": int(hrrr_accum_analysis.size),
            },
            "hourly": hourly_data,
        }

        # Sanitize the ENTIRE result through jsonify to kill any inf/nan
        result = jsonify(result)

        # Cache the result
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Analysis complete: {event_label}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ConnectionError as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(503,
            "Network error: Could not reach HRRR (S3) or gauge (IEM) servers. "
            "Check your internet connection and try again.")
    except TimeoutError as e:
        logger.error(f"Timeout: {e}")
        raise HTTPException(504,
            "HRRR download timed out. This usually means slow S3 access. "
            "Try again in a few minutes.")
    except Exception as e:
        logger.error(f"Pipeline failed: {traceback.format_exc()}")
        err_msg = str(e)
        # Provide helpful context for common failures
        if "s3fs" in err_msg.lower() or "s3://" in err_msg.lower():
            detail = (f"HRRR download error: {err_msg}. "
                     "Check internet connection and that s3fs is installed.")
        elif "aorc" in err_msg.lower() or "prate" in err_msg.lower() or "netcdf" in err_msg.lower():
            detail = (f"AORC data error: {err_msg}. "
                     f"Verify AORC files exist in {AORC_LOCAL_DIR}.")
        elif "gauge" in err_msg.lower() or "iem" in err_msg.lower() or "mesonet" in err_msg.lower():
            detail = (f"Gauge data error: {err_msg}. "
                     "IEM API may be temporarily unavailable.")
        else:
            detail = f"Analysis failed: {err_msg}"
        raise HTTPException(500, detail)

@app.delete("/api/cache")
async def clear_cache():
    """Clear all cached results (useful during development)."""
    import shutil
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    for d in FIGURES_DIR.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
            count += 1
    return {"cleared": count}

@app.get("/api/live-hrrr")
async def get_live_hrrr():
    """
    Fetch the latest available HRRR run and return a quick-look overlay.
    This runs on page load to show current forecast conditions immediately,
    before the user triggers a full analysis.
    
    Returns the most recent HRRR 24h accumulated precipitation as an
    overlay image URL + basic metadata (init time, valid period, domain).
    """
    from datetime import datetime, timedelta
    
    # Try today's 12Z, fall back to yesterday's 12Z
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    candidates = [
        datetime(now.year, now.month, now.day, 12),  # today 12Z
        datetime(now.year, now.month, now.day, 0),    # today 00Z
        datetime(now.year, now.month, now.day, 12) - timedelta(days=1),  # yesterday 12Z
    ]
    # Only try inits that are at least 13h in the past (fhr 13 needs to exist)
    candidates = [c for c in candidates if (now - c).total_seconds() > 13 * 3600]
    
    if not candidates:
        return JSONResponse({"available": False, "message": "No recent HRRR run available yet."})
    
    try:
        init_dt = candidates[0]
        event_start = init_dt + timedelta(hours=13)
        event_end = init_dt + timedelta(hours=36)
        
        # Use default domain
        bbox = DOMAIN_BBOX
        
        logger.info(f"Live HRRR: fetching {init_dt.strftime('%Y-%m-%d %HZ')}...")
        hrrr_ds = download_hrrr_hrrrzarr(
            event_start, min(event_end, now),
            strategy='single_init',
            init_hour=init_dt.hour,
        )
        
        # Find precip variable and accumulate
        hrrr_precip_var = None
        for v in ['tp', 'APCP_surface', 'APCP']:
            if v in hrrr_ds.data_vars:
                hrrr_precip_var = v
                break
        
        if hrrr_precip_var is None:
            return JSONResponse({"available": False, "message": "HRRR data downloaded but no precipitation variable found."})
        
        hrrr_accum = hrrr_ds[hrrr_precip_var].sum(dim='time').values
        lat_grid = hrrr_ds['latitude'].values if 'latitude' in hrrr_ds.coords else hrrr_ds['lat'].values
        lon_grid = hrrr_ds['longitude'].values if 'longitude' in hrrr_ds.coords else hrrr_ds['lon'].values
        
        # Generate overlay
        from visualization import generate_map_overlay, get_precip_cmap
        cmap, norm = get_precip_cmap()
        live_dir = FIGURES_DIR / "live"
        live_dir.mkdir(exist_ok=True)
        overlay_path = str(live_dir / "live_hrrr.png")
        
        meta = generate_map_overlay(hrrr_accum, lat_grid, lon_grid, cmap, norm,
                                     overlay_path, alpha=0.85)
        plt.close('all')
        
        return {
            "available": True,
            "init_time": init_dt.strftime("%Y-%m-%d %HZ"),
            "valid_start": event_start.isoformat(),
            "valid_end": min(event_end, now).isoformat(),
            "n_hours": len(hrrr_ds.time),
            "overlay_url": f"/figures/live/live_hrrr.png",
            "bounds": meta['bounds'],
            "mean_mm": round(float(np.nanmean(hrrr_accum)), 2),
            "max_mm": round(float(np.nanmax(hrrr_accum)), 2),
        }
    except Exception as e:
        logger.warning(f"Live HRRR failed: {e}")
        return JSONResponse({"available": False, "message": f"Could not fetch live HRRR: {str(e)}"})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("HRRR vs AORC Verification Dashboard")
    print("=" * 60)
    print(f"  Frontend: http://localhost:8050")
    print(f"  API docs: http://localhost:8050/docs")
    print(f"  AORC dir: {AORC_LOCAL_DIR}")
    print(f"  Strategy: {HRRR_FORECAST_STRATEGY}")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")
