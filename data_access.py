"""
=============================================================================
HRRR vs AORC Precipitation Verification Dashboard — Data Access
=============================================================================
Handles downloading and loading of:
  1. AORC precipitation data via FTP (hydrology.nws.noaa.gov)
  2. HRRR forecast data via hrrrzarr (S3 Zarr archive via s3fs)
  3. ASOS/AWOS rain gauge observations via Iowa Environmental Mesonet API

Each function downloads raw data, subsets to the configured domain,
and returns xarray Datasets or pandas DataFrames.

Future: NY-uHMT gauge integration will be added as a separate loader.
=============================================================================
"""

import os
import ftplib
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from io import StringIO

# Import project configuration
from config import (
    DOMAIN_BBOX, DOMAIN_BBOX_BUFFERED,
    AORC_FTP_HOST, AORC_FTP_BASE_DIR,
    IEM_ASOS_URL, IEM_NETWORKS,
    HRRR_RAW_DIR, AORC_RAW_DIR, GAUGE_RAW_DIR,
    DOMAIN_CENTER_LAT, DOMAIN_CENTER_LON
)


# =============================================================================
# 1. AORC DATA — PSL 4km Daily Files (prate.aorc.YYYYMMDD.nc)
# =============================================================================
#
# AORC precipitation comes from the NOAA Physical Sciences Laboratory 4km
# product, the SAME data source used in the SOM project. Each daily NetCDF
# file contains 24 hourly timesteps of precipitation RATE.
#
# Key details:
#   Server:    ftp2.psl.noaa.gov
#   Path:      /Projects/AORC_CONUS_4km/{YYYY}/prate.aorc.{YYYYMMDD}.nc
#   Variable:  prate (precipitation rate)
#   Units:     mm/hr (verified from file attrs — NOT kg/m²/s as some docs say)
#   Grid:      Regular lat/lon, ~0.04° (~4km) resolution, 885×1770 cells CONUS
#   Temporal:  24 hourly timesteps per file (hours 0-23 of that calendar day)
#
# Since the units are mm/hr and each timestep represents one hour,
# the prate value IS the hourly accumulation in mm. No unit conversion needed.
#
# For a 24h event starting at 00Z, we typically need just ONE daily file.
# For events spanning midnight UTC or sub-daily windows, we may need two.
# =============================================================================

def find_aorc_local(start_dt, end_dt, search_dirs=None):
    """
    Find AORC daily .nc files on your local machine.
    
    Searches configured directories for files matching the PSL 4km naming
    convention: prate.aorc.YYYYMMDD.nc
    
    For a 24h event starting at 00Z Sep 29 and ending at 00Z Sep 30:
      - We need prate.aorc.20240929.nc (contains hours 0-23 of Sep 29)
      - If the end hour is >00Z on the last day, we also need the next day's file
    
    Parameters
    ----------
    start_dt : datetime
        Start of the accumulation period (e.g., 2024-09-29 00:00 UTC)
    end_dt : datetime
        End of the accumulation period (e.g., 2024-09-30 00:00 UTC)
    search_dirs : list of str, optional
        Directories to search. Defaults to [AORC_LOCAL_DIR, AORC_RAW_DIR]
    
    Returns
    -------
    list of str
        Paths to found AORC daily files, sorted chronologically.
    """
    from config import AORC_LOCAL_DIR, AORC_RAW_DIR
    
    if search_dirs is None:
        search_dirs = [AORC_LOCAL_DIR, AORC_RAW_DIR]
    
    # Figure out which calendar days we need files for.
    # Each daily file covers hours 0-23 of that date.
    # For a 24h window 00Z Sep 29 → 00Z Sep 30, we need just Sep 29.
    # For a 24h window 06Z Sep 29 → 06Z Sep 30, we need Sep 29 AND Sep 30.
    needed_dates = set()
    current = start_dt
    while current < end_dt:
        needed_dates.add(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    # If end_dt is past midnight and not exactly at 00Z, include that day too
    if end_dt.hour > 0:
        needed_dates.add(end_dt.strftime('%Y%m%d'))
    
    needed_dates = sorted(needed_dates)
    print(f"Looking for {len(needed_dates)} AORC daily file(s): "
          f"{', '.join(needed_dates)}")
    
    # Scan search directories for .nc files matching the pattern
    available_files = {}  # date_str → full_path
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            print(f"  [SKIP] Directory not found: {search_dir}")
            continue
        
        for root, dirs, files in os.walk(search_dir):
            for fname in files:
                if not (fname.endswith('.nc') or fname.endswith('.nc4')):
                    continue
                # Match pattern: prate.aorc.YYYYMMDD.nc
                if 'prate' in fname.lower() and 'aorc' in fname.lower():
                    # Extract the 8-digit date
                    digits = ''.join(c for c in fname if c.isdigit())
                    if len(digits) >= 8:
                        date_str = digits[:8]
                        available_files[date_str] = os.path.join(root, fname)
    
    if not available_files:
        print(f"  WARNING: No AORC prate.aorc.*.nc files found in: {search_dirs}")
        print(f"  Update AORC_LOCAL_DIR in config.py (currently set in config)")
        return []
    
    print(f"  Found {len(available_files)} total AORC daily files in search paths")
    
    # Match needed dates to available files
    found_files = []
    missing_dates = []
    
    for date_str in needed_dates:
        if date_str in available_files:
            found_files.append(available_files[date_str])
        else:
            missing_dates.append(date_str)
    
    if missing_dates:
        print(f"  WARNING: Missing files for: {', '.join(missing_dates)}")
    
    print(f"  Matched {len(found_files)}/{len(needed_dates)} daily files")
    return sorted(found_files)


def download_aorc_ftp(start_dt, end_dt, save_dir=None):
    """
    Download AORC 4km daily files via FTP from ftp2.psl.noaa.gov.
    
    Matches the approach in 6.5_AORC_4km_Downloader_From_FTP.py but
    downloads only the specific dates needed for the analysis window.
    
    Parameters
    ----------
    start_dt : datetime
        Start of the period
    end_dt : datetime
        End of the period
    save_dir : str, optional
        Directory to save files. Defaults to AORC_RAW_DIR.
    
    Returns
    -------
    list of str
        Paths to downloaded files
    """
    if save_dir is None:
        save_dir = AORC_RAW_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # Build list of dates we need
    needed_dates = []
    current = start_dt
    while current < end_dt:
        needed_dates.append(current)
        current += timedelta(days=1)
    if end_dt.hour > 0:
        needed_dates.append(end_dt.replace(hour=0, minute=0, second=0))
    
    # Deduplicate
    seen = set()
    unique_dates = []
    for d in needed_dates:
        key = d.strftime('%Y%m%d')
        if key not in seen:
            seen.add(key)
            unique_dates.append(d)
    needed_dates = unique_dates
    
    print(f"Connecting to AORC FTP: {AORC_FTP_HOST}")
    print(f"Downloading {len(needed_dates)} daily file(s)")
    
    downloaded_files = []
    
    try:
        ftp = ftplib.FTP(AORC_FTP_HOST, timeout=30)
        ftp.login()  # Anonymous login
        print("FTP connection established.")
        
        for dt in needed_dates:
            year = dt.strftime('%Y')
            date_str = dt.strftime('%Y%m%d')
            filename = f"prate.aorc.{date_str}.nc"
            
            ftp_path = f"{AORC_FTP_BASE_DIR}/{year}/{filename}"
            local_path = os.path.join(save_dir, filename)
            
            # Skip if already downloaded and reasonable size (>1MB)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1_000_000:
                print(f"  [SKIP] {filename} (already exists)")
                downloaded_files.append(local_path)
                continue
            
            try:
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f'RETR {ftp_path}', f.write)
                
                file_size = os.path.getsize(local_path)
                if file_size < 100_000:  # < 100KB probably corrupt
                    os.remove(local_path)
                    print(f"  [FAIL] {filename}: too small ({file_size} bytes)")
                    continue
                
                size_mb = file_size / 1e6
                print(f"  [OK]   {filename} ({size_mb:.1f}MB)")
                downloaded_files.append(local_path)
                
            except ftplib.error_perm as e:
                print(f"  [FAIL] {filename}: {e}")
            except Exception as e:
                print(f"  [ERR]  {filename}: {e}")
        
        ftp.quit()
    
    except ftplib.all_errors as e:
        print(f"FTP connection error: {e}")
    
    print(f"Downloaded {len(downloaded_files)}/{len(needed_dates)} AORC files.")
    return downloaded_files


def download_aorc_https(start_dt, end_dt, save_dir=None):
    """
    AORC HTTPS download is not available for the PSL 4km product.
    Falls back to FTP download.
    """
    print("  [NOTE] PSL 4km AORC does not have an HTTPS endpoint.")
    print("         Using FTP download instead (ftp2.psl.noaa.gov).")
    return download_aorc_ftp(start_dt, end_dt, save_dir)


def load_aorc(file_paths, bbox=None, start_dt=None, end_dt=None):
    """
    Load AORC PSL 4km daily NetCDF files and extract precipitation.
    
    Each daily file contains 24 hourly timesteps of precipitation RATE
    (variable: 'prate', units: kg/m²/s). This function:
      1. Opens the daily file(s)
      2. Subsets to the analysis domain
      3. Converts precipitation rate → hourly accumulation (mm)
      4. Optionally slices to a specific time window within the day(s)
    
    Parameters
    ----------
    file_paths : list of str
        Paths to AORC daily .nc files
    bbox : dict, optional
        Bounding box. Defaults to DOMAIN_BBOX_BUFFERED.
    start_dt : datetime, optional
        If provided, only keep timesteps at or after this time.
        Needed when the event doesn't start at 00Z.
    end_dt : datetime, optional
        If provided, only keep timesteps before this time.
    
    Returns
    -------
    xr.Dataset with variable 'aorc_precip' in mm (hourly accumulation),
    dimensions (time, latitude, longitude)
    """
    if bbox is None:
        bbox = DOMAIN_BBOX_BUFFERED
    
    if not file_paths:
        raise ValueError("No AORC file paths provided!")
    
    print(f"Loading {len(file_paths)} AORC daily file(s)...")
    
    # Open all daily files — each has 24 hourly timesteps
    ds = xr.open_mfdataset(
        file_paths,
        combine='nested',
        concat_dim='time',
        engine='netcdf4',
        chunks={'time': 24}
    )
    
    # Identify coordinate names — AORC PSL uses 'latitude'/'longitude' or 'lat'/'lon'
    lat_name, lon_name = None, None
    for candidate_lat in ['latitude', 'lat', 'Latitude']:
        if candidate_lat in ds.dims or candidate_lat in ds.coords:
            lat_name = candidate_lat
            break
    for candidate_lon in ['longitude', 'lon', 'Longitude']:
        if candidate_lon in ds.dims or candidate_lon in ds.coords:
            lon_name = candidate_lon
            break
    
    if lat_name is None or lon_name is None:
        print(f"  Available dims: {list(ds.dims)}")
        print(f"  Available coords: {list(ds.coords)}")
        raise ValueError("Cannot identify lat/lon coordinates in AORC file!")
    
    # Handle longitude convention (some AORC files use 0-360)
    lons = ds[lon_name].values
    if np.any(lons > 180):
        print("  Converting AORC longitudes from 0–360 to -180–180...")
        ds[lon_name] = xr.where(ds[lon_name] > 180, ds[lon_name] - 360, ds[lon_name])
        ds = ds.sortby(lon_name)
    
    # Spatial subset to analysis domain
    ds_subset = ds.sel(**{
        lat_name: slice(bbox['south'], bbox['north']),
        lon_name: slice(bbox['west'], bbox['east'])
    })
    
    # If lat is in descending order, flip the slice
    if ds_subset[lat_name].size == 0:
        ds_subset = ds.sel(**{
            lat_name: slice(bbox['north'], bbox['south']),
            lon_name: slice(bbox['west'], bbox['east'])
        })
    
    # Identify the precipitation variable
    precip_var = None
    # PSL 4km product uses 'prate' (precipitation rate)
    prate_candidates = ['prate', 'PRATE', 'Precipitation_rate',
                        'precipitation_rate', 'precip_rate']
    # Also check for accumulated precip in case it's the OWP product
    accum_candidates = ['APCP_surface', 'APCP', 'Total_precipitation_surface',
                        'Total_precipitation_surface_1_Hour_Accumulation']
    
    is_rate = False
    for v in prate_candidates:
        if v in ds_subset.data_vars:
            precip_var = v
            is_rate = True
            break
    
    if precip_var is None:
        for v in accum_candidates:
            if v in ds_subset.data_vars:
                precip_var = v
                is_rate = False
                break
    
    if precip_var is None:
        print(f"  Available variables: {list(ds_subset.data_vars)}")
        raise ValueError(
            "Cannot identify precipitation variable in AORC file! "
            "Expected 'prate' (PSL 4km) or 'APCP_surface' (OWP 1km).")
    
    print(f"  Variable: '{precip_var}' ({'rate' if is_rate else 'accumulation'})")
    
    # Convert rate → hourly accumulation if needed
    if is_rate:
        # Check the units attribute to determine conversion factor
        units_str = str(ds_subset[precip_var].attrs.get('units', '')).lower().strip()
        
        if 'kg' in units_str and 's' in units_str:
            # kg/m²/s (or kg m-2 s-1): multiply by 3600 to get mm/hr
            # 1 kg/m² of water = 1 mm depth, so kg/m²/s × 3600s = mm/hr
            precip_mm = ds_subset[precip_var] * 3600.0
            print(f"  Units: '{units_str}' → ×3600 → mm/hr")
            
        elif 'mm/hr' in units_str or 'mm/h' in units_str or 'mm hr' in units_str:
            # Already in mm/hr — each hourly timestep value IS mm for that hour.
            # NO conversion needed.
            precip_mm = ds_subset[precip_var]
            print(f"  Units: '{units_str}' → already mm/hr, no conversion needed")
            
        elif 'mm' in units_str:
            # Plain 'mm' — likely already hourly accumulation
            precip_mm = ds_subset[precip_var]
            print(f"  Units: '{units_str}' → treating as mm (no conversion)")
            
        else:
            # Unknown units — warn and assume no conversion
            print(f"  ⚠ Unknown prate units: '{units_str}'")
            print(f"    Assuming values are already in mm/hr. Check your data!")
            print(f"    Sample values: min={float(ds_subset[precip_var].min()):.4f}, "
                  f"max={float(ds_subset[precip_var].max()):.4f}")
            precip_mm = ds_subset[precip_var]
    else:
        # Already accumulated precipitation in mm
        precip_mm = ds_subset[precip_var]
        print(f"  Units: accumulation (mm)")
    
    # Time slicing — if the event doesn't start at 00Z, trim to the window
    if start_dt is not None or end_dt is not None:
        time_vals = pd.DatetimeIndex(ds_subset.time.values)
        if start_dt is not None and end_dt is not None:
            time_mask = (time_vals >= np.datetime64(start_dt)) & \
                        (time_vals < np.datetime64(end_dt))
            precip_mm = precip_mm.isel(time=time_mask)
            print(f"  Time subset: {start_dt} to {end_dt} "
                  f"({int(time_mask.sum())}/{len(time_vals)} hours)")
        elif start_dt is not None:
            time_mask = time_vals >= np.datetime64(start_dt)
            precip_mm = precip_mm.isel(time=time_mask)
    
    # Build output dataset with clean variable name
    aorc_ds = xr.Dataset(
        data_vars={
            'aorc_precip': precip_mm.rename('aorc_precip') if precip_mm.name != 'aorc_precip' 
                          else precip_mm,
        },
        attrs={
            'source': f'AORC PSL 4km ({precip_var})',
            'units': 'mm (hourly accumulation)',
            'original_variable': precip_var,
            'converted_from_rate': is_rate,
        }
    )
    
    print(f"  Grid shape: {dict(aorc_ds.dims)}")
    print(f"  Spatial extent: {lat_name}=[{ds_subset[lat_name].min().values:.3f}, "
          f"{ds_subset[lat_name].max().values:.3f}], "
          f"{lon_name}=[{ds_subset[lon_name].min().values:.3f}, "
          f"{ds_subset[lon_name].max().values:.3f}]")
    print(f"  Time steps: {len(aorc_ds.time)}")
    
    return aorc_ds


# =============================================================================
# 2. HRRR DATA — VIA HRRRZARR (S3 Zarr Archive)
# =============================================================================
#
# HRRR data is accessed from the hrrrzarr archive on S3:
#   s3://hrrrzarr/sfc/{YYYYMMDD}/{YYYYMMDD}_{HH}z_fcst.zarr/
#     surface/APCP_1hr_acc_fcst/surface
#
# This is the same source used in the SOM project's parallel HRRR loader.
# Advantages over GRIB2:
#   - Cloud-native (no full file downloads)
#   - Spatial subsetting at read time (only fetches domain cells)
#   - No cfgrib/eccodes dependency
#   - Very fast for repeated access
#
# PRECIPITATION VARIABLE:
#   APCP_1hr_acc_fcst is a CUMULATIVE accumulation field. The value at
#   forecast hour N is the TOTAL precip accumulated from init to init+Nh.
#   To get the hourly amount for hour N, you must compute:
#       hourly_precip[N] = APCP[N] - APCP[N-1]
#
# FORECAST STRATEGY IMPLEMENTATION:
#   - rolling_1h:    Each valid hour uses its own init at fxx=1
#                    → Opens a DIFFERENT zarr store for each hour
#   - single_init:   One init, consecutive fxx values
#                    → Opens ONE zarr store, reads fxx=1 through fxx=N
#   - day_ahead_12z: Previous day 12Z, fxx=13 through fxx=36
#                    → Opens ONE zarr store (prev day 12Z), reads fxx 13-36
#                    This is what the SOM project uses.
# =============================================================================

def _get_hrrrzarr_domain_slices(fs=None, bbox=None):
    """
    Compute the y/x index slices that cover our analysis domain on the
    HRRR Lambert Conformal grid. Uses the chunk_index.zarr reference grid.
    
    Parameters
    ----------
    fs : s3fs.S3FileSystem, optional
    bbox : dict, optional
        Custom bounding box {south, north, west, east}. 
        Defaults to DOMAIN_BBOX_BUFFERED.
    
    Returns
    -------
    tuple (y_slice, x_slice, lat_2d, lon_2d)
        y_slice, x_slice: slice objects for isel() (integer-based, dim-agnostic)
        lat_2d, lon_2d: 2D coordinate arrays for the subset domain
    """
    import s3fs
    
    if fs is None:
        fs = s3fs.S3FileSystem(anon=True)
    
    if bbox is None:
        bbox = DOMAIN_BBOX_BUFFERED
    
    # Read the HRRR grid reference file
    chunk_index = xr.open_zarr(
        fs.get_mapper("s3://hrrrzarr/grid/HRRR_chunk_index.zarr"),
        consolidated=True
    )
    
    # Detect the y/x dimension names (they vary between files)
    dims = list(chunk_index.latitude.dims)
    grid_y_dim = dims[0]  # first dim = y-like
    grid_x_dim = dims[1]  # second dim = x-like
    print(f"    chunk_index dims: ({grid_y_dim}, {grid_x_dim})")
    
    # Create a boolean mask for grid cells inside our domain
    lat_mask = (chunk_index.latitude >= bbox['south']) & (chunk_index.latitude <= bbox['north'])
    lon_mask = (chunk_index.longitude >= bbox['west']) & (chunk_index.longitude <= bbox['east'])
    domain_mask = (lat_mask & lon_mask).compute()
    
    y_indices, x_indices = np.where(domain_mask.values)
    
    if len(y_indices) == 0:
        raise ValueError(
            f"No HRRR grid points found in domain {bbox}. "
            f"Check DOMAIN_BBOX in config.py.")
    
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    
    y_slice = slice(y_min, y_max + 1)
    x_slice = slice(x_min, x_max + 1)
    
    # Extract the lat/lon for this subset using detected dim names
    lat_2d = chunk_index.latitude.isel(
        **{grid_y_dim: y_slice, grid_x_dim: x_slice}
    ).values
    lon_2d = chunk_index.longitude.isel(
        **{grid_y_dim: y_slice, grid_x_dim: x_slice}
    ).values
    
    print(f"    HRRR domain: {lat_2d.shape[0]}x{lat_2d.shape[1]} cells "
          f"(y={y_min}:{y_max+1}, x={x_min}:{x_max+1})")
    print(f"    Lat range: {lat_2d.min():.2f} to {lat_2d.max():.2f}")
    print(f"    Lon range: {lon_2d.min():.2f} to {lon_2d.max():.2f}")
    
    return y_slice, x_slice, lat_2d, lon_2d


def _open_hrrrzarr_run(init_dt, fs=None):
    """
    Open one HRRR initialization run from hrrrzarr and return the
    APCP_1hr_acc_fcst DataArray.
    
    Parameters
    ----------
    init_dt : datetime
        Initialization time (e.g., 2024-09-28 12:00 for prev day 12Z)
    fs : s3fs.S3FileSystem, optional
        Reuse an existing connection
    
    Returns
    -------
    xr.DataArray
        The APCP cumulative accumulation field with dims (time, y, x)
    """
    import s3fs
    
    if fs is None:
        fs = s3fs.S3FileSystem(anon=True)
    
    date_str = init_dt.strftime('%Y%m%d')
    hour_str = init_dt.strftime('%H')
    
    zarr_path = (f"s3://hrrrzarr/sfc/{date_str}/{date_str}_{hour_str}z_fcst.zarr/"
                 f"surface/APCP_1hr_acc_fcst/surface")
    
    store = s3fs.S3Map(root=zarr_path, s3=fs, check=False)
    ds = xr.open_zarr(store, consolidated=True)
    
    return ds['APCP_1hr_acc_fcst']


def download_hrrr_hrrrzarr(start_dt, end_dt, strategy=None, init_hour=None,
                            fxx=None, bbox=None):
    """
    Load HRRR precipitation from the hrrrzarr S3 archive.
    
    This is the PRIMARY HRRR access method, matching the approach used in
    the SOM project's 8_Load_HRRR_Forecast_Parallell.py. It reads the
    Zarr-formatted HRRR archive directly from S3 via s3fs, which is:
      - Cloud-native (no file downloads, spatial subsetting at read time)
      - Fast (only fetches the grid cells in our domain)
      - Dependency-light (s3fs + xarray, no cfgrib/eccodes needed)
    
    HOURLY PRECIPITATION is computed by differencing consecutive cumulative
    accumulation steps: hourly[fhr] = APCP[fhr] - APCP[fhr-1].
    
    Parameters
    ----------
    start_dt : datetime
        Start of the accumulation period
    end_dt : datetime
        End of the accumulation period
    strategy : str, optional
        Override HRRR_FORECAST_STRATEGY from config.py.
        'rolling_1h', 'single_init', or 'day_ahead_12z'
    init_hour : int, optional
        Override HRRR_INIT_HOUR (for 'single_init' strategy)
    fxx : int, optional
        Override HRRR_FXX (for 'rolling_1h' strategy)
    
    Returns
    -------
    xr.Dataset with variables:
        'tp' : (time, y, x) — hourly precipitation in mm (kg/m²)
        'latitude' : (y, x) — 2D lat array
        'longitude' : (y, x) — 2D lon array
    Also stores 'lead_times' as an attribute for verification transparency.
    """
    import s3fs
    from config import (HRRR_FORECAST_STRATEGY, HRRR_INIT_HOUR, HRRR_FXX)
    
    if strategy is None:
        strategy = HRRR_FORECAST_STRATEGY
    if init_hour is None:
        init_hour = HRRR_INIT_HOUR
    if fxx is None:
        fxx = HRRR_FXX
    
    n_hours = int((end_dt - start_dt).total_seconds() / 3600)
    
    print(f"\n  HRRR Download (hrrrzarr):")
    print(f"    Strategy:  {strategy}")
    print(f"    Period:    {start_dt} → {end_dt} ({n_hours}h)")
    
    # Connect to S3
    fs = s3fs.S3FileSystem(anon=True)
    
    # Get domain indices and coordinate arrays
    print("    Setting up domain grid...")
    y_slice, x_slice, lat_2d, lon_2d = _get_hrrrzarr_domain_slices(fs, bbox=bbox)
    

    # ── Shared HRRR hourly reader ──────────────────────────────────────
    def _fetch_hours_parallel(apcp, init_dt, fhr_list, y_sl, x_sl):
        """Read HRRR forecast hours in parallel from a single zarr store."""
        from concurrent.futures import ThreadPoolExecutor
        fcst_dims = [d for d in apcp.dims if d != 'time']
        yd, xd = fcst_dims[0], fcst_dims[1]
        n_avail = apcp.sizes['time']
        print(f"    Zarr: {n_avail} hours, dims: ({yd}, {xd})")

        def _read_one(fhr):
            if fhr >= n_avail:
                return None
            try:
                curr = apcp.isel(time=fhr, **{yd: y_sl, xd: x_sl}).values
                prev = apcp.isel(time=fhr - 1, **{yd: y_sl, xd: x_sl}).values
                h = curr - prev
                h[(h <= -9999) | (h < 0) | (h > 500) | np.isnan(h)] = 0.0
                return (fhr, h, init_dt + timedelta(hours=fhr))
            except Exception as e:
                print(f"    [ERR] fhr={fhr}: {e}")
                return None

        print(f"    Downloading {len(fhr_list)} forecast hours (8 workers)...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_read_one, fhr_list))

        fields, times, leads, ok = [], [], [], 0
        for r in results:
            if r:
                fhr, arr, vt = r
                fields.append(arr); times.append(vt); leads.append(fhr); ok += 1
        print(f"    Loaded {ok}/{len(fhr_list)} hourly fields")
        return fields, times, leads, ok

    # ── Strategy: day_ahead_12z ──────────────────────────────────────────
    if strategy == 'day_ahead_12z':
        init_dt = (start_dt - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        fhr_start = int((start_dt - init_dt).total_seconds() / 3600) + 1
        fhr_end = int((end_dt - init_dt).total_seconds() / 3600)
        print(f"    Init: {init_dt} | fhr {fhr_start}-{fhr_end}")
        apcp = _open_hrrrzarr_run(init_dt, fs)
        hourly_fields, valid_times, lead_times, n_ok = _fetch_hours_parallel(
            apcp, init_dt, list(range(fhr_start, fhr_end + 1)), y_slice, x_slice)

    # ── Strategy: single_init ────────────────────────────────────────────
    elif strategy == 'single_init':
        init_dt = start_dt.replace(hour=init_hour, minute=0, second=0, microsecond=0)
        print(f"    Init: {init_dt} | fhr 1-{n_hours}")
        apcp = _open_hrrrzarr_run(init_dt, fs)
        hourly_fields, valid_times, lead_times, n_ok = _fetch_hours_parallel(
            apcp, init_dt, list(range(1, n_hours + 1)), y_slice, x_slice)

    # ── Strategy: rolling_1h ─────────────────────────────────────────────
    elif strategy == 'rolling_1h':
        print(f"    Rolling fxx={fxx}, opening {n_hours} zarr stores...")
        hourly_fields, valid_times, lead_times = [], [], []
        n_ok, yd, xd = 0, None, None

        for h in range(n_hours):
            vt = start_dt + timedelta(hours=h + 1)
            idt = vt - timedelta(hours=fxx)
            try:
                apcp = _open_hrrrzarr_run(idt, fs)
                if yd is None:
                    dims = [d for d in apcp.dims if d != 'time']
                    yd, xd = dims[0], dims[1]
                curr = apcp.isel(time=fxx, **{yd: y_slice, xd: x_slice}).values
                prev = apcp.isel(time=fxx-1, **{yd: y_slice, xd: x_slice}).values
                hr = curr - prev
                hr[(hr <= -9999) | (hr < 0) | (hr > 500) | np.isnan(hr)] = 0.0
                hourly_fields.append(hr); valid_times.append(vt); lead_times.append(fxx); n_ok += 1
            except Exception as e:
                print(f"    [FAIL] {idt.strftime('%m/%d %HZ')}: {e}")
        print(f"    Loaded {n_ok}/{n_hours}")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ── Build xarray Dataset ─────────────────────────────────────────────
    if not hourly_fields:
        raise ValueError(
            f"No HRRR data loaded! Check:\n"
            f"  1. Internet connection (s3fs needs access to s3://hrrrzarr)\n"
            f"  2. Date range: hrrrzarr covers ~2014 onwards\n"
            f"  3. For 'day_ahead_12z', init must be after 2020-12-02 (HRRRv4)")
    
    # Stack hourly fields into a 3D array (time, y, x)
    precip_3d = np.stack(hourly_fields, axis=0)
    
    hrrr_ds = xr.Dataset(
        data_vars={
            'tp': (['time', 'y', 'x'], precip_3d,
                   {'units': 'mm', 'long_name': '1-hour accumulated precipitation'}),
        },
        coords={
            'time': pd.DatetimeIndex(valid_times),
            'latitude': (['y', 'x'], lat_2d),
            'longitude': (['y', 'x'], lon_2d),
        },
        attrs={
            'source': 'hrrrzarr (s3://hrrrzarr/sfc/)',
            'strategy': strategy,
            'lead_times': lead_times,
            'mean_lead_h': float(np.mean(lead_times)),
            'n_valid_hours': n_ok,
            'n_expected_hours': n_hours,
        }
    )
    
    print(f"\n  HRRR Result:")
    print(f"    Shape: {precip_3d.shape} (time, y, x)")
    print(f"    Lead time: mean={np.mean(lead_times):.1f}h, "
          f"range={min(lead_times)}–{max(lead_times)}h")
    print(f"    Domain mean precip: {precip_3d.sum(axis=0).mean():.1f}mm "
          f"({n_ok}h accumulation)")
    
    return hrrr_ds




def subset_hrrr_to_domain(hrrr_ds, bbox=None):
    """
    Subset HRRR data to the domain bounding box.
    
    HRRR uses a Lambert Conformal Conic projection, so its lat/lon are
    2D arrays (curvilinear grid). We can't just do .sel() — instead we
    create a mask based on the 2D lat/lon coordinates.
    
    Parameters
    ----------
    hrrr_ds : xr.Dataset
        Full CONUS HRRR dataset
    bbox : dict, optional
        Bounding box. Defaults to DOMAIN_BBOX_BUFFERED.
    
    Returns
    -------
    xr.Dataset
        HRRR subset containing only grid cells within the domain
    """
    if bbox is None:
        bbox = DOMAIN_BBOX_BUFFERED
    
    # HRRR lat/lon are typically 2D arrays named 'latitude'/'longitude'
    lat = hrrr_ds['latitude'] if 'latitude' in hrrr_ds.coords else hrrr_ds['lat']
    lon = hrrr_ds['longitude'] if 'longitude' in hrrr_ds.coords else hrrr_ds['lon']
    
    # Create a boolean mask for grid cells inside the bounding box
    mask = (
        (lat >= bbox['south']) & (lat <= bbox['north']) &
        (lon >= bbox['west']) & (lon <= bbox['east'])
    )
    
    # Find the rectangular index range that contains all True values
    # This gives us a regular subset even from a curvilinear grid
    if 'y' in mask.dims and 'x' in mask.dims:
        y_dim, x_dim = 'y', 'x'
    else:
        y_dim = mask.dims[-2]
        x_dim = mask.dims[-1]
    
    y_indices = np.where(mask.any(dim=x_dim))[0]
    x_indices = np.where(mask.any(dim=y_dim))[0]
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("No HRRR grid cells found within the bounding box!")
    
    y_slice = slice(y_indices[0], y_indices[-1] + 1)
    x_slice = slice(x_indices[0], x_indices[-1] + 1)
    
    hrrr_subset = hrrr_ds.isel(**{y_dim: y_slice, x_dim: x_slice})
    
    print(f"  HRRR subset shape: {dict(hrrr_subset.dims)}")
    print(f"  Lat range: [{float(lat.isel(**{y_dim: y_slice, x_dim: x_slice}).min()):.3f}, "
          f"{float(lat.isel(**{y_dim: y_slice, x_dim: x_slice}).max()):.3f}]")
    
    return hrrr_subset


# =============================================================================
# 3. ASOS/AWOS RAIN GAUGE DATA VIA IOWA ENVIRONMENTAL MESONET
# =============================================================================

def download_asos_gauges(start_dt, end_dt, networks=None, bbox=None):
    """
    Download ASOS/AWOS hourly precipitation observations from the 
    Iowa Environmental Mesonet (IEM) API.
    
    IEM provides the most reliable, unified access to ASOS/AWOS/COOP data
    across the US. The API returns CSV formatted observation data.
    
    Parameters
    ----------
    start_dt : datetime
        Start of observation period
    end_dt : datetime
        End of observation period
    networks : list of str, optional
        IEM network codes (e.g., ['NY_ASOS', 'NJ_ASOS']).
        Defaults to IEM_NETWORKS from config.
    bbox : dict, optional
        Geographic filter. Defaults to DOMAIN_BBOX.
    
    Returns
    -------
    pd.DataFrame
        Hourly gauge observations with columns:
        station, name, lat, lon, valid_time, precip_mm, precip_in
    """
    if networks is None:
        networks = IEM_NETWORKS
    if bbox is None:
        bbox = DOMAIN_BBOX
    
    all_obs = []
    
    print(f"Downloading ASOS/AWOS gauge data: {start_dt} to {end_dt}")
    print(f"Networks: {networks}")
    
    for network in networks:
        # First, get the station metadata for this network
        stations = get_iem_stations(network, bbox)
        
        if not stations:
            print(f"  No stations found in {network} within bounding box")
            continue
        
        station_ids = [s['id'] for s in stations]
        print(f"  {network}: {len(station_ids)} stations in domain")
        
        # Build the IEM ASOS request URL
        params = {
            'station': station_ids,
            'data': 'p01i',  # 1-hour precipitation (inches)
            'tz': 'Etc/UTC',
            'format': 'comma',
            'latlon': 'yes',
            'year1': start_dt.year,
            'month1': start_dt.month,
            'day1': start_dt.day,
            'hour1': start_dt.hour,
            'year2': end_dt.year,
            'month2': end_dt.month,
            'day2': end_dt.day,
            'hour2': end_dt.hour,
        }
        
        try:
            response = requests.get(IEM_ASOS_URL, params=params, timeout=60)
            
            if response.status_code == 200 and len(response.text) > 100:
                # Parse the CSV response
                # IEM CSV has a header line starting with 'station'
                df = pd.read_csv(StringIO(response.text), comment='#')
                
                if len(df) > 0:
                    # Clean up the data
                    df = df.rename(columns={
                        'valid': 'valid_time',
                        'p01i': 'precip_in',
                        'lon': 'lon',
                        'lat': 'lat'
                    })
                    
                    # Convert precipitation from inches to mm
                    df['precip_in'] = pd.to_numeric(df['precip_in'], errors='coerce')
                    df['precip_mm'] = df['precip_in'] * 25.4
                    
                    # Parse timestamps
                    df['valid_time'] = pd.to_datetime(df['valid_time'], utc=True)
                    
                    # Filter to bounding box (IEM sometimes returns nearby stations)
                    df = df[
                        (df['lat'] >= bbox['south']) & (df['lat'] <= bbox['north']) &
                        (df['lon'] >= bbox['west']) & (df['lon'] <= bbox['east'])
                    ]
                    
                    # Basic QC: remove negative precip, flag extreme values
                    df.loc[df['precip_mm'] < 0, 'precip_mm'] = np.nan
                    df['qc_flag'] = 'OK'
                    df.loc[df['precip_mm'] > 100, 'qc_flag'] = 'EXTREME'
                    df.loc[df['precip_mm'].isna(), 'qc_flag'] = 'MISSING'
                    
                    all_obs.append(df)
                    print(f"    Retrieved {len(df)} observations from {df['station'].nunique()} stations")
            else:
                print(f"  [WARN] No data returned for {network}")
                
        except requests.RequestException as e:
            print(f"  [ERR]  {network}: {e}")
    
    if not all_obs:
        print("WARNING: No gauge observations retrieved!")
        return pd.DataFrame()
    
    # Combine all networks
    gauge_df = pd.concat(all_obs, ignore_index=True)
    
    # Tag as ASOS network
    gauge_df['network'] = 'ASOS'
    
    # Remove duplicate observations (same station+time from overlapping networks)
    gauge_df = gauge_df.drop_duplicates(subset=['station', 'valid_time'], keep='first')
    
    # Sort by station and time
    gauge_df = gauge_df.sort_values(['station', 'valid_time']).reset_index(drop=True)
    
    print(f"\nTotal: {len(gauge_df)} observations from "
          f"{gauge_df['station'].nunique()} unique stations")
    
    return gauge_df


def get_iem_stations(network, bbox=None):
    """
    Get IEM station metadata for a given network, filtered by bounding box.
    
    Parameters
    ----------
    network : str
        IEM network code (e.g., 'NY_ASOS')
    bbox : dict, optional
        Geographic filter
    
    Returns
    -------
    list of dict
        Station metadata: id, name, lat, lon, elevation
    """
    url = f"https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return []
        
        data = response.json()
        stations = []
        
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            coords = feature.get('geometry', {}).get('coordinates', [None, None])
            
            lon, lat = coords[0], coords[1]
            
            # Filter by bounding box
            if bbox and (lat < bbox['south'] or lat > bbox['north'] or
                        lon < bbox['west'] or lon > bbox['east']):
                continue
            
            stations.append({
                'id': props.get('sid', ''),
                'name': props.get('sname', ''),
                'lat': lat,
                'lon': lon,
                'elevation': props.get('elevation', None),
                'network': network,
            })
        
        return stations
    
    except Exception as e:
        print(f"  Error fetching stations for {network}: {e}")
        return []


def get_gauge_station_metadata(networks=None, bbox=None):
    """
    Get metadata (location, name) for all gauge stations in the domain.
    Useful for plotting station locations on maps.
    
    Returns
    -------
    pd.DataFrame
        Station metadata with columns: station_id, name, lat, lon, network
    """
    if networks is None:
        networks = IEM_NETWORKS
    if bbox is None:
        bbox = DOMAIN_BBOX
    
    all_stations = []
    for network in networks:
        stations = get_iem_stations(network, bbox)
        all_stations.extend(stations)
    
    if not all_stations:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_stations)
    df = df.rename(columns={'id': 'station_id'})
    df = df.drop_duplicates(subset='station_id')
    
    print(f"Found {len(df)} gauge stations in domain:")
    for _, row in df.iterrows():
        print(f"  {row['station_id']:6s}  {row['name']:30s}  "
              f"{row['lat']:.3f}°N  {abs(row['lon']):.3f}°W")
    
    return df


def accumulate_gauge_precip(gauge_df, start_dt, accum_hours):
    """
    Accumulate hourly gauge observations over a specified window.
    
    Parameters
    ----------
    gauge_df : pd.DataFrame
        Hourly gauge observations (from download_asos_gauges)
    start_dt : datetime
        Start of accumulation window
    accum_hours : int
        Number of hours to accumulate (6, 12, 18, or 24)
    
    Returns
    -------
    pd.DataFrame
        One row per station with total accumulated precipitation,
        plus columns for number of valid hours and data completeness.
    """
    end_dt = start_dt + timedelta(hours=accum_hours)
    
    # Filter to the time window
    mask = (
        (gauge_df['valid_time'] >= pd.Timestamp(start_dt, tz='UTC')) &
        (gauge_df['valid_time'] < pd.Timestamp(end_dt, tz='UTC'))
    )
    window_data = gauge_df[mask].copy()
    
    if len(window_data) == 0:
        print(f"WARNING: No gauge data in window {start_dt} to {end_dt}")
        return pd.DataFrame()
    
    # Aggregate by station
    agg = window_data.groupby('station').agg(
        total_precip_mm=('precip_mm', 'sum'),
        n_valid_hours=('precip_mm', 'count'),
        n_missing=('precip_mm', lambda x: x.isna().sum()),
        lat=('lat', 'first'),
        lon=('lon', 'first'),
    ).reset_index()
    
    # Data completeness: what fraction of hours have valid data?
    agg['completeness'] = agg['n_valid_hours'] / accum_hours
    
    # Flag stations with too much missing data (< 75% complete)
    agg['qc_flag'] = 'GOOD'
    agg.loc[agg['completeness'] < 0.75, 'qc_flag'] = 'INCOMPLETE'
    agg.loc[agg['completeness'] < 0.50, 'qc_flag'] = 'POOR'
    
    print(f"Accumulated {accum_hours}h gauge data: {len(agg)} stations")
    print(f"  GOOD: {(agg['qc_flag']=='GOOD').sum()}, "
          f"INCOMPLETE: {(agg['qc_flag']=='INCOMPLETE').sum()}, "
          f"POOR: {(agg['qc_flag']=='POOR').sum()}")
    
    return agg


# =============================================================================
# 4. NY-uHMT (New York Urban Hydrometeorological Testbed)
# =============================================================================
#
# Dense urban rain gauge network across NYC, operated by NOAA CREST.
# 15-minute tipping bucket CSV files from:
#   https://datadb.noaacrest.org/public/uhmt/Processed_Data
#
# CSV columns: TIMESTAMP, RECORD, AirTF, RH, Rainfall_Tot
# - TIMESTAMP: Eastern local time (EDT/EST), no timezone indicator
# - Rainfall_Tot: INCREMENTAL precipitation per 15-min interval, in INCHES
# - To get hourly: SUM four 15-min readings (not max, unlike ASOS cumulative)
#
# Key data quality issues (from audit):
# - Site18 has bogus timestamps (1990s, 2031) — filter to 2017-2025
# - Site14 only covers 2018-2019 (short record)
# - Some sites have AirTF sensor errors (>150°F) — doesn't affect rainfall
# - Site13 Astoria gauge failed during Sep 2023 flood (all zeros)
# =============================================================================

def download_nyuhmt_gauges(start_dt, end_dt, bbox=None):
    """
    Download and process NY-uHMT 15-minute gauge data for the event window.
    
    Reads CSVs from the NOAA CREST server (or local cache), converts
    timestamps from Eastern time to UTC, aggregates 15-min readings to
    hourly, and converts inches to mm.
    
    Parameters
    ----------
    start_dt, end_dt : datetime
        Event window in UTC
    bbox : dict, optional
        Geographic filter {south, north, west, east}. Defaults to DOMAIN_BBOX.
    
    Returns
    -------
    pd.DataFrame
        Hourly gauge observations with columns matching ASOS format:
        station, name, lat, lon, valid_time (UTC), precip_mm, network
    """
    from config import (NY_UHMT_ENABLED, NY_UHMT_BASE_URL, 
                        NY_UHMT_LOCAL_DIR, NY_UHMT_STATIONS)
    
    if not NY_UHMT_ENABLED:
        return pd.DataFrame()
    
    if bbox is None:
        bbox = DOMAIN_BBOX
    
    print(f"\nDownloading NY-uHMT gauge data: {start_dt} to {end_dt}")
    print(f"  {len(NY_UHMT_STATIONS)} stations configured")
    
    # We need a buffer around the event window because the timestamps are
    # in Eastern time. A 6-hour buffer ensures we capture data that spans
    # the UTC/ET boundary.
    buffer_hours = 6
    fetch_start = start_dt - timedelta(hours=buffer_hours)
    fetch_end = end_dt + timedelta(hours=buffer_hours)
    
    all_hourly = []
    stations_loaded = 0
    stations_skipped = 0
    
    for site_num, (name, code, lat, lon, csv_file) in NY_UHMT_STATIONS.items():
        # Filter by bounding box
        if (lat < bbox['south'] or lat > bbox['north'] or
            lon < bbox['west'] or lon > bbox['east']):
            continue
        
        # Try to load the CSV: local first, then download
        df = None
        
        # Option 1: Local directory
        if NY_UHMT_LOCAL_DIR:
            local_path = os.path.join(NY_UHMT_LOCAL_DIR, csv_file)
            if os.path.exists(local_path):
                try:
                    df = pd.read_csv(local_path, low_memory=False)
                except Exception as e:
                    print(f"  [WARN] Failed to read local {csv_file}: {e}")
        
        # Option 2: Download from server
        if df is None:
            url = f"{NY_UHMT_BASE_URL}/{csv_file}"
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/csv,text/plain,*/*',
                }
                print(f"  Fetching {code} ...")
                response = requests.get(url, timeout=120, headers=headers, verify=False,
                                         allow_redirects=True)
                if response.status_code == 200 and len(response.text) > 100:
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text), low_memory=False)
                    print(f"    ✓ {code}: {len(df)} rows")
                    
                    # Cache locally for next time
                    if NY_UHMT_LOCAL_DIR:
                        try:
                            os.makedirs(NY_UHMT_LOCAL_DIR, exist_ok=True)
                            cache_path = os.path.join(NY_UHMT_LOCAL_DIR, csv_file)
                            with open(cache_path, 'w', newline='') as fout:
                                fout.write(response.text)
                        except Exception:
                            pass
                else:
                    print(f"  [WARN] {code}: HTTP {response.status_code} (len={len(response.text)})")
                    stations_skipped += 1
                    continue
            except requests.exceptions.SSLError as e:
                print(f"  [WARN] {code}: SSL error — {e}")
                stations_skipped += 1
                continue
            except requests.exceptions.ConnectionError as e:
                print(f"  [WARN] {code}: Connection error — {e}")
                stations_skipped += 1
                continue
            except requests.exceptions.Timeout:
                print(f"  [WARN] {code}: Timed out after 120s")
                stations_skipped += 1
                continue
            except Exception as e:
                print(f"  [WARN] {code}: {type(e).__name__}: {e}")
                stations_skipped += 1
                continue
        
        if df is None or len(df) == 0:
            stations_skipped += 1
            continue
        
        # --- Parse and process ---
        # Parse timestamps (Eastern local time)
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        
        # Filter out bogus timestamps (Site18 has 1990s and 2031 dates)
        df = df[df['TIMESTAMP'].notna()]
        df = df[(df['TIMESTAMP'].dt.year >= 2017) & (df['TIMESTAMP'].dt.year <= 2025)]
        
        if len(df) == 0:
            stations_skipped += 1
            continue
        
        # Localize as Eastern time, convert to UTC
        # ambiguous='NaT' handles fall-back DST hour; nonexistent='NaT' handles spring-forward
        try:
            df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_localize(
                'America/New_York', ambiguous='NaT', nonexistent='NaT'
            )
            df = df[df['TIMESTAMP'].notna()]  # Drop the ~4 rows/year lost to DST transitions
            df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_convert('UTC')
        except Exception as e:
            print(f"  [WARN] Timezone conversion failed for {code}: {e}")
            stations_skipped += 1
            continue
        
        # Filter to the buffered event window
        mask = (df['TIMESTAMP'] >= pd.Timestamp(fetch_start, tz='UTC')) & \
               (df['TIMESTAMP'] < pd.Timestamp(fetch_end, tz='UTC'))
        df = df[mask]
        
        if len(df) == 0:
            stations_skipped += 1
            continue
        
        # Parse rainfall (force numeric, handle any non-numeric values)
        df['Rainfall_Tot'] = pd.to_numeric(df['Rainfall_Tot'], errors='coerce').fillna(0)
        
        # QC: negative values → 0, extreme values (>2 in per 15 min) flagged
        df.loc[df['Rainfall_Tot'] < 0, 'Rainfall_Tot'] = 0
        
        # Aggregate 15-minute readings to hourly SUMS
        # (NY-uHMT Rainfall_Tot is incremental per 15-min, NOT cumulative like ASOS)
        df['hour'] = df['TIMESTAMP'].dt.floor('h')
        hourly = df.groupby('hour').agg(
            precip_in=('Rainfall_Tot', 'sum'),
            n_readings=('Rainfall_Tot', 'count'),
        ).reset_index()
        
        # Convert inches to mm
        hourly['precip_mm'] = hourly['precip_in'] * 25.4
        
        # QC: flag hours with fewer than 3 of 4 expected readings
        hourly['qc_flag'] = 'OK'
        hourly.loc[hourly['n_readings'] < 3, 'qc_flag'] = 'INCOMPLETE'
        hourly.loc[hourly['precip_mm'] > 100, 'qc_flag'] = 'EXTREME'
        
        # Build output rows matching ASOS format
        hourly['station'] = code
        hourly['name'] = name
        hourly['lat'] = lat
        hourly['lon'] = lon
        hourly['valid_time'] = hourly['hour']
        hourly['network'] = 'NY-uHMT'
        
        all_hourly.append(hourly[['station', 'name', 'lat', 'lon', 
                                   'valid_time', 'precip_mm', 'precip_in',
                                   'qc_flag', 'network']])
        stations_loaded += 1
    
    if not all_hourly:
        print(f"  No NY-uHMT data loaded (skipped {stations_skipped} stations)")
        return pd.DataFrame()
    
    result = pd.concat(all_hourly, ignore_index=True)
    
    # Final filter to exact event window (remove the buffer)
    result = result[
        (result['valid_time'] >= pd.Timestamp(start_dt, tz='UTC')) &
        (result['valid_time'] < pd.Timestamp(end_dt, tz='UTC'))
    ]
    
    result = result.sort_values(['station', 'valid_time']).reset_index(drop=True)
    
    print(f"  NY-uHMT: {len(result)} hourly obs from {stations_loaded} stations "
          f"({stations_skipped} skipped)")
    
    return result


# =============================================================================
# 5. MRMS (Multi-Radar Multi-Sensor) QPE
# =============================================================================
#
# NOAA MRMS provides radar+gauge merged precipitation at ~1km resolution.
# We use the hourly GaugeCorr (Pass2) product from the Iowa State archive:
#   https://mtarchive.geol.iastate.edu/YYYY/MM/DD/mrms/ncep/
#
# Files are GRIB2 format, one per hour. We download, subset to the domain,
# and return an xarray Dataset matching the AORC format for comparison.
# =============================================================================

def download_mrms_qpe(start_dt, end_dt, bbox=None):
    """
    Download MRMS hourly QPE for the event window from the Iowa State archive.
    
    Uses the MultiSensor_QPE_01H_Pass2 product (gauge-corrected radar QPE).
    Returns an xarray Dataset with hourly precipitation in mm on a regular
    lat-lon grid, subset to the analysis domain.
    
    Parameters
    ----------
    start_dt, end_dt : datetime
        Event window in UTC
    bbox : dict, optional
        Geographic filter. Defaults to DOMAIN_BBOX.
    
    Returns
    -------
    xr.Dataset with variable 'mrms_precip' (time, lat, lon) in mm
    None if download fails or data unavailable
    """
    from config import MRMS_ENABLED, MRMS_ARCHIVE_URL, MRMS_PRODUCT, MRMS_RAW_DIR
    import gzip
    
    if not MRMS_ENABLED:
        print("MRMS is disabled in config.py (MRMS_ENABLED = False)")
        return None
    
    # Check for cfgrib before starting downloads
    try:
        import cfgrib
        print(f"  cfgrib available (v{cfgrib.__version__})")
    except ImportError:
        print("ERROR: cfgrib is required for MRMS GRIB2 reading.")
        print("  Install with: conda install -c conda-forge cfgrib eccodes")
        print("  Or: pip install cfgrib (requires eccodes C library)")
        raise ImportError(
            "cfgrib package not found. MRMS requires cfgrib to read GRIB2 files. "
            "Install with: conda install -c conda-forge cfgrib eccodes"
        )
    
    if bbox is None:
        bbox = DOMAIN_BBOX
    
    os.makedirs(MRMS_RAW_DIR, exist_ok=True)
    
    n_hours = int((end_dt - start_dt).total_seconds() / 3600)
    print(f"\nDownloading MRMS QPE: {start_dt} to {end_dt} ({n_hours} hours)")
    print(f"  Product: {MRMS_PRODUCT}")
    print(f"  Cache dir: {MRMS_RAW_DIR}")
    
    hourly_fields = []
    valid_times = []
    lat_out = None
    lon_out = None
    
    for h in range(n_hours):
        # MRMS QPE files are stamped by accumulation END time
        # For the period 00Z-01Z, the file is stamped 01Z
        # So to cover 00Z-24Z, we fetch files stamped 01Z through 24Z(=00Z next day)
        valid_time = start_dt + timedelta(hours=h + 1)  # hour-ending stamp
        
        # Check local cache first
        cache_filename = f"mrms_{valid_time.strftime('%Y%m%d_%H')}.grib2"
        cache_path = os.path.join(MRMS_RAW_DIR, cache_filename)
        
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 10000:
            # Use cached file
            tmp_path = cache_path
        else:
            # Download from Iowa State archive
            url = (f"{MRMS_ARCHIVE_URL}/{valid_time.strftime('%Y/%m/%d')}/mrms/ncep/"
                   f"{MRMS_PRODUCT}/{MRMS_PRODUCT}_00.00_"
                   f"{valid_time.strftime('%Y%m%d-%H')}0000.grib2.gz")
            
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    print(f"  [WARN] HTTP {resp.status_code} for {valid_time.strftime('%Y-%m-%d %HZ')}")
                    continue
                
                # Decompress and save to cache
                decompressed = gzip.decompress(resp.content)
                with open(cache_path, 'wb') as f:
                    f.write(decompressed)
                tmp_path = cache_path
                
            except requests.RequestException as e:
                print(f"  [WARN] Download failed for {valid_time.strftime('%HZ')}: {e}")
                continue
        
        # Read GRIB2 with cfgrib
        try:
            ds = xr.open_dataset(tmp_path, engine='cfgrib')
        except Exception as e1:
            print(f"  [WARN] cfgrib failed for {valid_time.strftime('%HZ')}: {e1}")
            continue
        
        # Find the precipitation variable
        precip_var = None
        for v in ds.data_vars:
            arr = ds[v]
            if arr.dims and len(arr.dims) >= 2 and arr.dtype in [np.float32, np.float64]:
                precip_var = v
                break
        
        if precip_var is None:
            ds.close()
            continue
        
        precip = ds[precip_var]
        
        # Get lat/lon (MRMS is on a regular lat-lon grid)
        if 'latitude' in ds.coords:
            lat = ds['latitude'].values
            lon = ds['longitude'].values
        elif 'lat' in ds.coords:
            lat = ds['lat'].values
            lon = ds['lon'].values
        else:
            ds.close()
            continue
        
        # Subset to domain
        if lat.ndim == 1:
            # Convert lon from 0-360 to -180 to 180 if needed
            lon_vals = lon.copy()
            if lon_vals.max() > 180:
                lon_vals = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
            
            lat_mask = (lat >= bbox['south']) & (lat <= bbox['north'])
            lon_mask = (lon_vals >= bbox['west']) & (lon_vals <= bbox['east'])
            
            precip_sub = precip.values
            if precip_sub.ndim == 2:
                precip_sub = precip_sub[np.ix_(lat_mask, lon_mask)]
            elif precip_sub.ndim == 3:
                precip_sub = precip_sub[0][np.ix_(lat_mask, lon_mask)]
            
            if lat_out is None:
                lat_out = lat[lat_mask]
                lon_out = lon_vals[lon_mask]
        else:
            # 2D lat/lon — fall back to full array
            precip_sub = precip.values
            if lat_out is None:
                lat_out = lat
                lon_out = lon
            
        # Replace negative/missing with 0
        precip_sub = np.where(np.isnan(precip_sub) | (precip_sub < 0), 0, precip_sub)
        
        hourly_fields.append(precip_sub)
        valid_times.append(valid_time)
        ds.close()
    
    if not hourly_fields or lat_out is None:
        print(f"  MRMS: No data retrieved")
        return None
    
    coverage = len(hourly_fields) / n_hours
    print(f"  MRMS: {len(hourly_fields)}/{n_hours} hours retrieved ({coverage*100:.0f}% coverage)")
    
    if coverage < 0.75:
        print(f"  MRMS: Below 75% coverage threshold — returning None to avoid misleading totals")
        return None
    
    # Stack into 3D array
    precip_3d = np.stack(hourly_fields, axis=0)
    
    mrms_ds = xr.Dataset(
        data_vars={
            'mrms_precip': (['time', 'lat', 'lon'], precip_3d,
                           {'units': 'mm', 'long_name': 'MRMS hourly QPE (Pass2)'}),
        },
        coords={
            'time': pd.DatetimeIndex(valid_times),
            'lat': lat_out,
            'lon': lon_out,
        },
        attrs={
            'source': f'MRMS {MRMS_PRODUCT} via {MRMS_ARCHIVE_URL}',
            'n_valid_hours': len(hourly_fields),
            'n_expected_hours': n_hours,
        }
    )
    
    return mrms_ds


# =============================================================================
# CONVENIENCE: Download everything for an event
# =============================================================================

def download_event_data(event_start, accum_hours=24, use_https=True,
                        prefer_local=True):
    """
    Convenience function to download all data for a precipitation event.
    Downloads HRRR, AORC, and gauge data for the configured domain.
    
    Tries LOCAL AORC files first (from your SOM project directory) before
    attempting FTP/HTTPS downloads. This avoids re-downloading data you
    already have.
    
    Parameters
    ----------
    event_start : datetime or str
        Event start time. If str, parsed as 'YYYY-MM-DD HH:MM'
    accum_hours : int
        Accumulation window in hours
    use_https : bool
        If True, use HTTPS for AORC (more reliable than FTP in many environments)
    prefer_local : bool
        If True, search local directories for AORC before downloading
    
    Returns
    -------
    dict with keys:
        'hrrr': xr.Dataset — HRRR precip forecast
        'aorc': xr.Dataset — AORC precip analysis
        'gauges': pd.DataFrame — Gauge observations
        'gauge_accum': pd.DataFrame — Accumulated gauge totals
        'event_start': datetime
        'event_end': datetime
        'accum_hours': int
        'n_aorc_hours': int — actual number of AORC hours loaded
    """
    if isinstance(event_start, str):
        event_start = datetime.strptime(event_start, "%Y-%m-%d %H:%M")
    
    event_end = event_start + timedelta(hours=accum_hours)
    
    print("=" * 60)
    print(f"DOWNLOADING EVENT DATA")
    print(f"  Period: {event_start} to {event_end} ({accum_hours}h)")
    print(f"  Domain: {DOMAIN_BBOX}")
    print("=" * 60)
    
    # 1. Load AORC — try local files first, then download
    print("\n--- AORC DATA ---")
    aorc_files = []
    aorc_ds = None
    
    if prefer_local:
        aorc_files = find_aorc_local(event_start, event_end)
    
    if not aorc_files:
        print("  No local AORC files found. Downloading...")
        if use_https:
            aorc_files = download_aorc_https(event_start, event_end)
        else:
            aorc_files = download_aorc_ftp(event_start, event_end)
    
    n_aorc_hours = len(aorc_files)
    if aorc_files:
        aorc_ds = load_aorc(aorc_files)
    else:
        print("  ⚠ No AORC data available for this event!")
    
    # 2. Download HRRR
    print("\n--- HRRR DATA ---")
    hrrr_ds = download_hrrr_hrrrzarr(event_start, event_end)
    
    # 3. Download gauge data
    print("\n--- GAUGE DATA ---")
    gauge_df = download_asos_gauges(event_start, event_end)
    gauge_accum = None
    if len(gauge_df) > 0:
        gauge_accum = accumulate_gauge_precip(gauge_df, event_start, accum_hours)
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    if n_aorc_hours < accum_hours:
        print(f"  ⚠ AORC: {n_aorc_hours}/{accum_hours} hours loaded "
              f"({n_aorc_hours/accum_hours*100:.0f}% complete)")
    else:
        print(f"  ✓ AORC: {n_aorc_hours}/{accum_hours} hours — complete")
    print("=" * 60)
    
    return {
        'hrrr': hrrr_ds,
        'aorc': aorc_ds,
        'gauges': gauge_df,
        'gauge_accum': gauge_accum,
        'event_start': event_start,
        'event_end': event_end,
        'accum_hours': accum_hours,
        'n_aorc_hours': n_aorc_hours,
    }


# =============================================================================
if __name__ == "__main__":
    # Quick test: just get station metadata (no heavy downloads)
    print("\n--- Testing gauge station discovery ---")
    stations = get_gauge_station_metadata()
    print(f"\nFound {len(stations)} stations in the NYC domain")
