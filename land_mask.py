"""
=============================================================================
Land Mask Generator — Natural Earth 10m Coastline
=============================================================================
Builds a boolean land mask for any HRRR grid subset by testing each cell 
center against Natural Earth 10m coastline polygons via cartopy/shapely.

Why this is needed:
    HRRR produces precipitation values over ocean grid cells.
    AORC bilinear regridding spreads small nonzero precipitation into 
    coastal ocean pixels. Simply checking for zero values fails because
    both products have legitimate nonzero values over water.
    
    The solution is a geographic mask built from actual coastline geometry.

Usage:
    from land_mask import get_land_mask
    
    mask = get_land_mask(lat_2d, lon_2d)  # True = land, False = ocean
    hrrr_accum[~mask] = np.nan            # Set ocean pixels to NaN
    aorc_accum[~mask] = np.nan

The mask is cached as a .npy file so it only needs to be computed once
per grid shape. Regenerate if the domain changes.

Author: Sebastian Makrides
=============================================================================
"""

import os
import numpy as np
from pathlib import Path

# Cache directory for saved masks
MASK_CACHE_DIR = Path(__file__).parent / "cache" / "masks"


def build_land_mask(lat_2d, lon_2d):
    """
    Build a boolean land mask by testing each grid cell center against
    Natural Earth 10m coastline polygons.
    
    Parameters
    ----------
    lat_2d : np.ndarray
        2D latitude array (ny, nx)
    lon_2d : np.ndarray
        2D longitude array (ny, nx)
    
    Returns
    -------
    np.ndarray of bool, shape (ny, nx)
        True where the grid cell center is over land, False over ocean.
    """
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.prepared import prep
    from shapely.ops import unary_union
    
    print("Building land mask from Natural Earth 10m coastlines...")
    print(f"  Grid shape: {lat_2d.shape}")
    
    # Load Natural Earth 10m land polygons
    land_shp = shpreader.natural_earth(
        resolution='10m', category='physical', name='land'
    )
    reader = shpreader.Reader(land_shp)
    land_geom = unary_union(list(reader.geometries()))
    
    # Buffer the land polygons outward by ~0.03° (~3km, one HRRR grid cell)
    # This ensures coastal grid cells whose centers fall slightly offshore
    # (e.g., Long Island south shore, Connecticut coast) are classified as land.
    # Without this, narrow land features get masked out because the cell center
    # is technically over water even though the cell covers land.
    BUFFER_DEG = 0.03
    land_geom = land_geom.buffer(BUFFER_DEG)
    print(f"  Land polygons buffered by {BUFFER_DEG}° (~{BUFFER_DEG*111:.1f}km)")
    
    # Use prepared geometry for fast point-in-polygon testing
    prepared_land = prep(land_geom)
    
    ny, nx = lat_2d.shape
    mask = np.zeros((ny, nx), dtype=bool)
    
    total = ny * nx
    checked = 0
    
    for j in range(ny):
        for i in range(nx):
            pt = Point(float(lon_2d[j, i]), float(lat_2d[j, i]))
            mask[j, i] = prepared_land.contains(pt)
            checked += 1
        
        if (j + 1) % 50 == 0 or j == ny - 1:
            pct = 100.0 * checked / total
            print(f"  Mask progress: {pct:.1f}% ({checked}/{total} cells)")
    
    n_land = mask.sum()
    n_ocean = total - n_land
    print(f"  Result: {n_land} land cells, {n_ocean} ocean cells "
          f"({100*n_ocean/total:.1f}% ocean)")
    
    return mask


def get_land_mask(lat_2d, lon_2d, cache_dir=None):
    """
    Get a land mask for the given grid, using cached version if available.
    
    The cache key is based on grid shape and corner coordinates so the mask
    is regenerated if the domain changes.
    
    Parameters
    ----------
    lat_2d, lon_2d : np.ndarray
        2D coordinate arrays
    cache_dir : str or Path, optional
        Directory for mask cache. Defaults to cache/masks/ in the app dir.
    
    Returns
    -------
    np.ndarray of bool
    """
    if cache_dir is None:
        cache_dir = MASK_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Build a cache key from grid shape and corner coordinates
    ny, nx = lat_2d.shape
    corners = (
        f"{lat_2d[0,0]:.4f}_{lon_2d[0,0]:.4f}_"
        f"{lat_2d[-1,-1]:.4f}_{lon_2d[-1,-1]:.4f}"
    )
    cache_file = cache_dir / f"land_mask_{ny}x{nx}_{corners}_buf03.npy"
    
    if cache_file.exists():
        print(f"Loading cached land mask: {cache_file.name}")
        mask = np.load(cache_file)
        if mask.shape == (ny, nx):
            return mask
        else:
            print(f"  Cache shape mismatch, rebuilding...")
    
    # Build the mask
    mask = build_land_mask(lat_2d, lon_2d)
    
    # Save to cache
    np.save(cache_file, mask)
    print(f"  Saved mask to {cache_file}")
    
    return mask


def apply_land_mask(data, mask):
    """
    Apply land mask to a 2D precipitation array.
    Sets ocean pixels to NaN.
    
    Parameters
    ----------
    data : np.ndarray (ny, nx)
        Precipitation field
    mask : np.ndarray of bool (ny, nx)
        True = land, False = ocean
    
    Returns
    -------
    np.ndarray with ocean cells set to NaN
    """
    result = np.array(data, dtype=float)
    result[~mask] = np.nan
    return result
