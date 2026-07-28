"""
=============================================================================
HRRR vs AORC Precipitation Verification Dashboard — Regridding
=============================================================================
Handles regridding (interpolation) of AORC data onto the HRRR grid so that
grid-to-grid comparison is possible.

Strategy: Regrid AORC (1km, regular lat/lon) → HRRR (3km, Lambert Conformal)
We upscale AORC to HRRR's coarser grid because we're evaluating HRRR 
forecasts, so metrics should be on HRRR's native grid.

Method: Bilinear interpolation (as requested).
Note: For strict precipitation mass conservation, conservative regridding 
(via xESMF) would be more appropriate, but bilinear is faster and simpler.
The trade-off is that bilinear smooths peak values slightly.

Fallback: If scipy is available but xESMF is not, we use scipy's 
griddata() for interpolation. If xESMF IS available, we use it for 
cleaner and faster bilinear interpolation with weight caching.
=============================================================================
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from config import DOMAIN_BBOX, PROCESSED_DIR


def regrid_aorc_to_hrrr(aorc_ds, hrrr_ds, method='bilinear', cache_weights=True):
    """
    Regrid AORC data onto the HRRR grid using bilinear interpolation.
    
    This function tries xESMF first (faster, supports weight caching), 
    and falls back to scipy.interpolate.griddata if xESMF is not available.
    
    Parameters
    ----------
    aorc_ds : xr.Dataset
        AORC precipitation data on its native ~1km regular lat/lon grid
    hrrr_ds : xr.Dataset
        HRRR data — we use its lat/lon coordinates as the target grid
    method : str
        Interpolation method: 'bilinear' (default), 'nearest_s2d', or 'conservative'
    cache_weights : bool
        If True and using xESMF, cache the regridding weights for reuse
    
    Returns
    -------
    xr.Dataset
        AORC precipitation data interpolated onto the HRRR grid
    """
    print(f"Regridding AORC → HRRR grid (method: {method})")
    
    try:
        return _regrid_xesmf(aorc_ds, hrrr_ds, method, cache_weights)
    except ImportError:
        print("  xESMF not available, falling back to scipy griddata")
        return _regrid_scipy(aorc_ds, hrrr_ds)


def _regrid_xesmf(aorc_ds, hrrr_ds, method='bilinear', cache_weights=True):
    """
    Regrid using xESMF (preferred method).
    
    xESMF wraps ESMF (Earth System Modeling Framework) which handles
    curvilinear grids natively — perfect for HRRR's Lambert Conformal grid.
    
    Weight computation is expensive but only needs to happen once per 
    source-target grid pair. Weights are cached to disk for reuse.
    """
    import xesmf as xe
    
    # Determine the weight file path for caching
    weight_file = os.path.join(PROCESSED_DIR, f"regrid_weights_{method}.nc")
    reuse_weights = cache_weights and os.path.exists(weight_file)
    
    if reuse_weights:
        print(f"  Reusing cached weights: {weight_file}")
    
    # Build the regridder
    # xESMF needs datasets with 'lat'/'lon' as coordinate names
    # Normalize AORC coordinate names
    aorc_for_regrid = _normalize_coords(aorc_ds)
    hrrr_for_regrid = _normalize_coords(hrrr_ds)
    
    regridder = xe.Regridder(
        aorc_for_regrid,
        hrrr_for_regrid,
        method=method,
        periodic=False,
        unmapped_to_nan=True,
        filename=weight_file if cache_weights else None,
        reuse_weights=reuse_weights,
    )
    
    # Identify the precipitation variable in AORC
    precip_var = _find_precip_var(aorc_ds)
    
    if precip_var is None:
        raise ValueError(f"Cannot find precipitation variable in AORC. "
                        f"Available: {list(aorc_ds.data_vars)}")
    
    print(f"  Regridding variable: '{precip_var}'")
    
    # Apply regridding for each timestep
    # xESMF handles the spatial interpolation; we loop over time
    aorc_precip = aorc_for_regrid[precip_var]
    
    # Apply the regridder
    regridded = regridder(aorc_precip)
    
    # Package as a Dataset
    result = regridded.to_dataset(name='aorc_precip')
    
    # Copy over the HRRR lat/lon for downstream use
    if 'latitude' in hrrr_ds.coords:
        result = result.assign_coords(latitude=hrrr_ds['latitude'],
                                       longitude=hrrr_ds['longitude'])
    
    print(f"  Regridded shape: {dict(result.dims)}")
    print(f"  Regridding complete ✓")
    
    return result


def _regrid_scipy(aorc_ds, hrrr_ds):
    """
    Fallback regridding using scipy.interpolate.griddata.
    
    This is slower than xESMF and doesn't cache weights, but works
    without any special dependencies beyond scipy.
    
    The approach:
    1. Flatten AORC lat/lon/precip into 1D arrays of (lat, lon, value) points
    2. Create the target grid from HRRR's 2D lat/lon arrays
    3. Interpolate using scipy's griddata with 'linear' method (= bilinear)
    """
    print("  Using scipy.interpolate.griddata (bilinear)")
    
    # Get AORC coordinates and precip variable
    aorc_norm = _normalize_coords(aorc_ds)
    precip_var = _find_precip_var(aorc_ds)
    
    if precip_var is None:
        raise ValueError(f"Cannot find precip variable. Available: {list(aorc_ds.data_vars)}")
    
    # Get HRRR target coordinates (2D lat/lon for Lambert Conformal grid)
    hrrr_lat, hrrr_lon = _get_hrrr_latlon(hrrr_ds)
    
    # Get AORC source coordinates (1D lat/lon for regular grid)
    if 'lat' in aorc_norm.dims:
        aorc_lat_1d = aorc_norm['lat'].values
        aorc_lon_1d = aorc_norm['lon'].values
    else:
        aorc_lat_1d = aorc_norm['latitude'].values
        aorc_lon_1d = aorc_norm['longitude'].values
    
    # Create 2D meshgrid from AORC's 1D coordinates
    aorc_lon_2d, aorc_lat_2d = np.meshgrid(aorc_lon_1d, aorc_lat_1d)
    
    # Flatten the source coordinates into (N, 2) array of points
    source_points = np.column_stack([aorc_lat_2d.ravel(), aorc_lon_2d.ravel()])
    
    # Flatten the target coordinates
    target_points = np.column_stack([hrrr_lat.ravel(), hrrr_lon.ravel()])
    
    # Handle time dimension: regrid each timestep separately
    aorc_precip = aorc_norm[precip_var]
    
    if 'time' in aorc_precip.dims:
        timesteps = aorc_precip.time.values
        regridded_list = []
        
        for t_idx, t in enumerate(timesteps):
            precip_slice = aorc_precip.sel(time=t).values
            source_values = precip_slice.ravel()
            
            # Remove NaN points from source (griddata can't handle them)
            valid = ~np.isnan(source_values)
            
            if valid.sum() < 10:
                print(f"  [WARN] Timestep {t}: only {valid.sum()} valid points, skipping")
                regridded_values = np.full(hrrr_lat.shape, np.nan)
            else:
                regridded_values = griddata(
                    source_points[valid],
                    source_values[valid],
                    target_points,
                    method='linear',  # bilinear interpolation
                    fill_value=0.0     # Fill edges with 0 (no precip)
                ).reshape(hrrr_lat.shape)
            
            regridded_list.append(regridded_values)
            
            if (t_idx + 1) % 6 == 0:
                print(f"  Regridded {t_idx + 1}/{len(timesteps)} timesteps")
        
        # Stack into a 3D array (time, y, x)
        regridded_array = np.stack(regridded_list, axis=0)
        
        # Build xarray Dataset
        y_dim = 'y' if 'y' in hrrr_ds.dims else list(hrrr_ds.dims)[-2]
        x_dim = 'x' if 'x' in hrrr_ds.dims else list(hrrr_ds.dims)[-1]
        
        result = xr.Dataset(
            {'aorc_precip': (['time', y_dim, x_dim], regridded_array)},
            coords={
                'time': timesteps,
                'latitude': ([y_dim, x_dim], hrrr_lat) if hrrr_lat.ndim == 2 
                           else ([y_dim], hrrr_lat),
                'longitude': ([y_dim, x_dim], hrrr_lon) if hrrr_lon.ndim == 2 
                            else ([x_dim], hrrr_lon),
            }
        )
    else:
        # Single timestep
        precip_values = aorc_precip.values.ravel()
        valid = ~np.isnan(precip_values)
        
        regridded_values = griddata(
            source_points[valid],
            precip_values[valid],
            target_points,
            method='linear',
            fill_value=0.0
        ).reshape(hrrr_lat.shape)
        
        y_dim = 'y' if 'y' in hrrr_ds.dims else list(hrrr_ds.dims)[-2]
        x_dim = 'x' if 'x' in hrrr_ds.dims else list(hrrr_ds.dims)[-1]
        
        result = xr.Dataset(
            {'aorc_precip': ([y_dim, x_dim], regridded_values)},
            coords={
                'latitude': ([y_dim, x_dim], hrrr_lat) if hrrr_lat.ndim == 2
                           else ([y_dim], hrrr_lat),
                'longitude': ([y_dim, x_dim], hrrr_lon) if hrrr_lon.ndim == 2
                            else ([x_dim], hrrr_lon),
            }
        )
    
    # Ensure no negative precip from interpolation artifacts
    result['aorc_precip'] = result['aorc_precip'].clip(min=0)
    
    print(f"  Regridded shape: {dict(result.dims)}")
    print(f"  Regridding complete ✓")
    
    return result


def accumulate_precip(ds, precip_var, start_time, accum_hours):
    """
    Sum hourly precipitation over an accumulation window.
    
    Parameters
    ----------
    ds : xr.Dataset
        Hourly precipitation dataset with time dimension
    precip_var : str
        Name of the precipitation variable in the dataset
    start_time : datetime-like
        Start of accumulation window
    accum_hours : int
        Number of hours to sum (6, 12, 18, or 24)
    
    Returns
    -------
    xr.DataArray
        2D accumulated precipitation field (y, x)
    """
    from datetime import timedelta
    import pandas as pd
    
    end_time = pd.Timestamp(start_time) + timedelta(hours=accum_hours)
    
    # Select the time window
    ds_window = ds.sel(time=slice(str(start_time), str(end_time)))
    
    n_times = len(ds_window.time)
    print(f"  Accumulating {precip_var}: {n_times} timesteps over {accum_hours}h")
    
    if n_times == 0:
        raise ValueError(f"No data found in time window {start_time} to {end_time}")
    
    # Sum over time dimension
    accumulated = ds_window[precip_var].sum(dim='time', skipna=True)
    
    # Warn if we're missing timesteps
    expected = accum_hours  # 1 per hour
    if n_times < expected:
        pct = n_times / expected * 100
        print(f"  WARNING: Only {n_times}/{expected} hours available ({pct:.0f}% complete)")
    
    return accumulated


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize_coords(ds):
    """Rename lat/lon coordinates to 'lat'/'lon' for consistency."""
    rename_map = {}
    for name in ds.coords:
        if name.lower() in ['latitude', 'lat_0']:
            rename_map[name] = 'lat'
        elif name.lower() in ['longitude', 'lon_0']:
            rename_map[name] = 'lon'
    
    if rename_map:
        return ds.rename(rename_map)
    return ds


def _find_precip_var(ds):
    """Find the precipitation variable name in a dataset."""
    candidates = ['APCP_surface', 'APCP', 'tp', 'Total_precipitation_surface',
                  'Total_precipitation_surface_1_Hour_Accumulation',
                  'precip', 'precipitation', 'aorc_precip',
                  'RAINRATE', 'PRATE']
    
    for var in candidates:
        if var in ds.data_vars:
            return var
    
    # Last resort: return the first data variable
    data_vars = list(ds.data_vars)
    if data_vars:
        print(f"  [WARN] Using first data variable as precip: '{data_vars[0]}'")
        return data_vars[0]
    
    return None


def _get_hrrr_latlon(hrrr_ds):
    """Extract 2D lat/lon arrays from HRRR dataset."""
    if 'latitude' in hrrr_ds.coords:
        lat = hrrr_ds['latitude'].values
        lon = hrrr_ds['longitude'].values
    elif 'lat' in hrrr_ds.coords:
        lat = hrrr_ds['lat'].values
        lon = hrrr_ds['lon'].values
    else:
        raise ValueError(f"Cannot find lat/lon in HRRR. Coords: {list(hrrr_ds.coords)}")
    
    return lat, lon


def save_regridded(result, filename=None):
    """Save regridded dataset to NetCDF in the processed directory."""
    if filename is None:
        filename = "aorc_on_hrrr_grid.nc"
    
    filepath = os.path.join(PROCESSED_DIR, filename)
    result.to_netcdf(filepath)
    print(f"  Saved regridded data: {filepath}")
    return filepath


def load_regridded(filename=None):
    """Load previously saved regridded data."""
    if filename is None:
        filename = "aorc_on_hrrr_grid.nc"
    
    filepath = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(filepath):
        ds = xr.open_dataset(filepath)
        print(f"  Loaded regridded data: {filepath}")
        return ds
    else:
        raise FileNotFoundError(f"No regridded data found at {filepath}")


# =============================================================================
if __name__ == "__main__":
    print("Regridding module loaded.")
    print("Usage:")
    print("  from regridding import regrid_aorc_to_hrrr")
    print("  aorc_on_hrrr = regrid_aorc_to_hrrr(aorc_ds, hrrr_ds)")
