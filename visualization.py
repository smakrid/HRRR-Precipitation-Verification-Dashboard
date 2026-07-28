"""
=============================================================================
HRRR vs AORC Precipitation Verification Dashboard — Visualization
=============================================================================
Generates all plots and map figures for the verification analysis.

Plot categories:
  1. SPATIAL MAPS: Precipitation fields, difference maps, metric overlays
     with gauge stations as bubble markers (using cartopy for projections)
  2. PDF PLOTS: Probability Density Functions comparing distributions
  3. CDF PLOTS: Cumulative Distribution Functions
  4. GAUGE COMPARISON: Scatter plots, bar charts of station-level metrics
  5. DASHBOARD: Multi-panel summary figure combining key views
  
All figures are designed for interactive viewing in Spyder's plot pane
AND for saving as high-resolution PNGs for reports.

If cartopy is not installed, falls back to plain matplotlib (no coastlines).
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server use (TkAgg crashes with FastAPI)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import (DOMAIN_BBOX, FIGURES_DIR, PRECIP_CMAP_LEVELS, 
                    DIFF_CMAP_LEVELS, DOMAIN_CENTER_LAT, DOMAIN_CENTER_LON,
                    NYC_TIGHT_BBOX)

# Try to import cartopy for map projections (not strictly required)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("Cartopy available — maps will include coastlines and state borders")
except ImportError:
    HAS_CARTOPY = False
    print("Cartopy not installed — maps will use plain lat/lon axes")
    print("Install with: conda install -c conda-forge cartopy")

# =============================================================================
# COLOR MAPS — NWS-inspired precipitation colormaps
# =============================================================================

def get_precip_cmap(levels=None):
    """
    Create an NWS-style precipitation colormap.
    
    Colors go from white/gray (trace) through greens (light rain),
    yellows/oranges (moderate), reds (heavy), to purples (extreme).
    This is the standard color scheme meteorologists expect to see.
    """
    if levels is None:
        levels = PRECIP_CMAP_LEVELS
    
    colors = [
        '#FFFFFF',  # 0:     White (no precip)
        '#C8E6C8',  # 0.5:   Very light green
        '#7BC87B',  # 1:     Light green
        '#4CAF50',  # 2:     Green
        '#2E7D32',  # 5:     Dark green
        '#FFEB3B',  # 10:    Yellow
        '#FFC107',  # 15:    Amber
        '#FF9800',  # 20:    Orange
        '#F44336',  # 30:    Red
        '#B71C1C',  # 50:    Dark red
        '#9C27B0',  # 75:    Purple
        '#4A148C',  # 100+:  Deep purple
    ]
    
    # Trim colors to match number of levels
    n = len(levels) - 1
    colors = colors[:n]
    
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    
    return cmap, norm

def get_diff_cmap(levels=None):
    """
    Create a diverging colormap for difference fields (HRRR - AORC).
    Blue = HRRR under-predicts, White = no difference, Red = HRRR over-predicts.
    
    The number of colors must exactly equal len(levels) - 1 (one color per bin).
    For the default DIFF_CMAP_LEVELS with 13 boundaries, that means 12 colors,
    symmetric around the zero-crossing: 6 blues → white → 6 reds.
    """
    if levels is None:
        levels = DIFF_CMAP_LEVELS
    
    # Define a rich palette — must have exactly len(levels)-1 colors.
    # Default DIFF_CMAP_LEVELS = [-30,-20,-15,-10,-5,-2, 0, 2, 5, 10, 15, 20, 30]
    # That's 13 boundaries → 12 bins → 12 colors needed.
    full_palette = [
        '#08306B',  # -30 to -20: very strong under-prediction (darkest blue)
        '#1D5996',  # -20 to -15
        '#2171B5',  # -15 to -10
        '#6BAED6',  # -10 to -5
        '#BDD7E7',  # -5  to -2
        '#E8EEF2',  # -2  to  0: slight under-prediction (near white)
        '#FDEAE3',  #  0  to  2: slight over-prediction (near white)
        '#FCBBA1',  #  2  to  5
        '#FB6A4A',  #  5  to 10
        '#CB181D',  # 10  to 15
        '#99000D',  # 15  to 20
        '#67000D',  # 20  to 30: very strong over-prediction (darkest red)
    ]
    
    n_bins = len(levels) - 1
    
    # If the user passes custom levels, dynamically build colors to match
    if n_bins == len(full_palette):
        colors = full_palette
    else:
        # Interpolate from a matplotlib diverging colormap to guarantee 
        # we always get exactly the right number of colors
        base_cmap = plt.cm.RdBu_r  # Red-Blue reversed (red=positive, blue=negative)
        colors = [base_cmap(i / (n_bins - 1)) for i in range(n_bins)]
    
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N, clip=True)
    
    return cmap, norm

# =============================================================================
# NYC TIGHT DOMAIN BOX — drawn on every map as a reference outline
# =============================================================================

def generate_map_overlay(data, lat, lon, cmap, norm, save_path, alpha=0.7):
    """
    Generate a transparent PNG overlay suitable for Leaflet's L.imageOverlay.
    
    This renders the 2D precipitation field as a colored image with no axes,
    no borders, no labels — just colored pixels with transparency. The result
    maps directly onto the geographic domain via Leaflet's LatLngBounds.
    
    Parameters
    ----------
    data : np.ndarray
        2D field to render (y, x)
    lat, lon : np.ndarray
        2D coordinate arrays (same shape as data)
    cmap : matplotlib.colors.Colormap
        Colormap to use
    norm : matplotlib.colors.Normalize
        Normalizer for data-to-color mapping
    save_path : str
        Where to write the PNG
    alpha : float
        Transparency for non-zero pixels (0=invisible, 1=opaque)
    
    Returns
    -------
    dict with 'path', 'bounds' (south, west, north, east)
    """
    data = np.asarray(data, dtype=float)
    
    # Map data to RGBA using the colormap
    colored = cmap(norm(data))  # shape (ny, nx, 4)
    
    # Set alpha: transparent where data is NaN or negligible
    colored[..., 3] = alpha            # set all to desired alpha
    
    if np.any(data < 0):
        # Difference field — only make near-zero values transparent
        mask_transparent = np.isnan(data) | (np.abs(data) < 0.5)
    else:
        # Precip field — make zero/trace values transparent
        mask_transparent = np.isnan(data) | (data < 0.1)
    
    colored[mask_transparent, 3] = 0.0  # make masked cells fully transparent
    
    # Flip if needed — Leaflet ImageOverlay expects north at the top of the image.
    # If row 0 has a smaller latitude than the last row (south at top of array),
    # we need to flip so north ends up at the image top.
    if lat.ndim == 2:
        lat_first_row = np.nanmean(lat[0, :])
        lat_last_row = np.nanmean(lat[-1, :])
    else:
        lat_first_row = lat[0]
        lat_last_row = lat[-1]
    
    if lat_first_row < lat_last_row:
        # Row 0 is south → flip so north is at image top
        colored = np.flipud(colored)
    # else: row 0 is already north → no flip needed
    
    # Save as PNG with no borders, no padding, exact pixel dimensions
    ny, nx = data.shape
    dpi = 150
    fig_w = nx / dpi
    fig_h = ny / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(colored, aspect='auto', interpolation='bilinear')
    ax.axis('off')
    fig.savefig(save_path, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Compute geographic bounds from the lat/lon arrays
    bounds = {
        'south': float(np.nanmin(lat)),
        'north': float(np.nanmax(lat)),
        'west': float(np.nanmin(lon)),
        'east': float(np.nanmax(lon)),
    }
    
    return {'path': save_path, 'bounds': bounds}

def generate_all_overlays(hrrr_accum, aorc_accum, lat, lon, metrics_dict,
                          save_dir, prefix="event"):
    """
    Generate transparent PNG overlays for HRRR, AORC, and difference fields.
    
    Returns a dict of overlay metadata (path + geographic bounds) for each layer.
    """
    cmap_p, norm_p = get_precip_cmap()
    cmap_d, norm_d = get_diff_cmap()
    
    overlays = {}
    
    print("  Generating HRRR overlay...")
    overlays['hrrr'] = generate_map_overlay(
        hrrr_accum, lat, lon, cmap_p, norm_p,
        save_path=os.path.join(save_dir, f"{prefix}_overlay_hrrr.png"),
        alpha=0.85
    )
    
    print("  Generating AORC overlay...")
    overlays['aorc'] = generate_map_overlay(
        aorc_accum, lat, lon, cmap_p, norm_p,
        save_path=os.path.join(save_dir, f"{prefix}_overlay_aorc.png"),
        alpha=0.85
    )
    
    print("  Generating difference overlay...")
    diff = metrics_dict['spatial']['difference']
    overlays['diff'] = generate_map_overlay(
        diff, lat, lon, cmap_d, norm_d,
        save_path=os.path.join(save_dir, f"{prefix}_overlay_diff.png"),
        alpha=0.85
    )
    
    return overlays

def _add_nyc_box(ax, plot_kwargs=None):
    """
    Draw the tight NYC metro domain as a black dashed rectangle on the map.
    This box shows the domain from the SOM verification project
    (40.48-40.92°N, 74.26-73.70°W) for spatial reference.
    """
    from matplotlib.patches import Rectangle
    
    box = NYC_TIGHT_BBOX
    width = box['east'] - box['west']
    height = box['north'] - box['south']
    
    rect_kwargs = dict(
        linewidth=2.0, edgecolor='black', facecolor='none',
        linestyle='--', zorder=8
    )
    
    if HAS_CARTOPY and plot_kwargs and 'transform' in plot_kwargs:
        rect_kwargs['transform'] = plot_kwargs['transform']
    
    rect = Rectangle((box['west'], box['south']), width, height, **rect_kwargs)
    ax.add_patch(rect)
    
    # Small label at top-left corner of the box
    label_kwargs = {}
    if HAS_CARTOPY and plot_kwargs and 'transform' in plot_kwargs:
        label_kwargs['transform'] = plot_kwargs['transform']
    
    ax.text(box['west'] + 0.01, box['north'] - 0.02, 'NYC Metro',
            fontsize=7, fontweight='bold', color='black', alpha=0.85,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7,
                     ec='black', lw=0.5),
            zorder=9, **label_kwargs)

# =============================================================================
# 1. SPATIAL MAP PLOTS
# =============================================================================

def _create_map_axes(fig, subplot_spec=None, position=None):
    """Create a map axes with or without cartopy projection."""
    kwargs = {}
    if subplot_spec is not None:
        pos_args = (subplot_spec,)
    elif position is not None:
        pos_args = (position,)
    else:
        pos_args = (111,)
    
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(*pos_args, projection=proj, **kwargs)
        ax.set_extent([DOMAIN_BBOX['west'], DOMAIN_BBOX['east'],
                       DOMAIN_BBOX['south'], DOMAIN_BBOX['north']], 
                      crs=ccrs.PlateCarree())
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#333333')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='#666666')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#666666')
        ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                     alpha=0.5, linestyle='--')
    else:
        ax = fig.add_subplot(*pos_args, **kwargs)
        ax.set_xlim(DOMAIN_BBOX['west'], DOMAIN_BBOX['east'])
        ax.set_ylim(DOMAIN_BBOX['south'], DOMAIN_BBOX['north'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    return ax

def plot_precip_field(data, lat, lon, title="Precipitation", 
                      gauge_df=None, save_path=None, levels=None,
                      ax=None, fig=None, show_colorbar=True):
    """
    Plot a precipitation field on a map with optional gauge station overlays.
    
    This is the core spatial plotting function. It renders the gridded 
    precipitation as a filled contour/pcolormesh plot with the NWS-style 
    colormap, and optionally overlays rain gauge observations as circles 
    sized by observed amount.
    
    Parameters
    ----------
    data : np.ndarray
        2D precipitation field (y, x) in mm
    lat, lon : np.ndarray
        2D coordinate arrays matching data shape
    title : str
        Plot title
    gauge_df : pd.DataFrame, optional
        Gauge data with columns: lat, lon, total_precip_mm (for bubble markers)
    save_path : str, optional
        If provided, save figure to this path
    levels : list, optional
        Contour levels for the colormap
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    fig : matplotlib.figure.Figure, optional
        Existing figure. Required if ax is provided and you want a colorbar.
    show_colorbar : bool
        Whether to add a colorbar
    
    Returns
    -------
    fig, ax
    """
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = _create_map_axes(fig)
        created_fig = True
    
    # Prepare data
    data_2d = np.asarray(data, dtype=float)
    lat_2d = np.asarray(lat, dtype=float)
    lon_2d = np.asarray(lon, dtype=float)
    
    # Get colormap
    cmap, norm = get_precip_cmap(levels)
    
    # Plot the precipitation field
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    plot_kwargs = {'transform': transform} if transform else {}
    
    mesh = ax.pcolormesh(lon_2d, lat_2d, data_2d, cmap=cmap, norm=norm,
                         shading='auto', **plot_kwargs)
    
    # Add colorbar — pad=0.08 prevents overlap with cartopy gridline labels
    if show_colorbar and fig is not None:
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.08, extend='max')
        cbar.set_label('Precipitation (mm)', fontsize=10)
    
    # Overlay gauge stations as bubble markers
    if gauge_df is not None and len(gauge_df) > 0:
        _add_gauge_bubbles(ax, gauge_df, cmap, norm, plot_kwargs)
    
    # Draw the tight NYC metro domain box
    _add_nyc_box(ax, plot_kwargs)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    if save_path and created_fig:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    return fig, ax

def plot_difference_field(diff_data, lat, lon, title="HRRR − AORC Difference",
                         gauge_df=None, save_path=None, levels=None):
    """
    Plot the difference field (HRRR - AORC) with a diverging colormap.
    
    Blue areas: HRRR predicts less rain than AORC (under-prediction)
    Red areas: HRRR predicts more rain than AORC (over-prediction)
    White/gray: Good agreement between HRRR and AORC
    """
    fig = plt.figure(figsize=(12, 8))
    ax = _create_map_axes(fig)
    
    cmap, norm = get_diff_cmap(levels)
    
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    plot_kwargs = {'transform': transform} if transform else {}
    
    mesh = ax.pcolormesh(np.asarray(lon), np.asarray(lat), np.asarray(diff_data),
                         cmap=cmap, norm=norm, shading='auto', **plot_kwargs)
    
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.08, extend='both')
    cbar.set_label('HRRR − AORC (mm)', fontsize=10)
    
    # Draw the tight NYC metro domain box
    _add_nyc_box(ax, plot_kwargs)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    return fig, ax

def _add_gauge_bubbles(ax, gauge_df, cmap, norm, plot_kwargs, neutral_style=False):
    """
    Add rain gauge observations as colored bubble markers on the map.
    
    Bubble SIZE encodes the observed amount (larger = more rain).
    Bubble COLOR matches the precipitation colormap for consistency,
    UNLESS neutral_style=True (used on difference maps) where bubbles
    are white-filled with dark borders to avoid confusing the diverging scale.
    
    Labels are offset AWAY from the nearest domain edge so they don't clip.
    """
    precip_col = 'total_precip_mm' if 'total_precip_mm' in gauge_df.columns else 'observed_mm'
    
    if precip_col not in gauge_df.columns:
        return
    
    # Domain edges for label positioning logic
    lon_mid = (DOMAIN_BBOX['west'] + DOMAIN_BBOX['east']) / 2
    lat_mid = (DOMAIN_BBOX['south'] + DOMAIN_BBOX['north']) / 2
    
    for _, row in gauge_df.iterrows():
        obs = row.get(precip_col, 0)
        if np.isnan(obs):
            continue
        
        # Size: scale observation amount to marker size (min 30, max 200)
        size = max(30, min(200, obs * 5))
        
        # Color: either from colormap or neutral white
        if neutral_style:
            color = 'white'
            edge_color = '#333333'
        else:
            color = cmap(norm(obs))
            edge_color = 'black'
        
        scatter_kwargs = {}
        if 'transform' in plot_kwargs:
            scatter_kwargs['transform'] = plot_kwargs['transform']
        
        ax.scatter(row['lon'], row['lat'], s=size, c=[color], 
                   edgecolors=edge_color, linewidths=1.2, zorder=5,
                   **scatter_kwargs)
        
        # Smart label offset: push labels AWAY from the nearest domain edge
        # so they don't get clipped. Default is upper-right offset.
        x_offset = -55 if row['lon'] > lon_mid else 8
        y_offset = -12 if row['lat'] > lat_mid else 8
        
        # Additional check: if very close to right/top edge, force left/down
        lon_frac = (row['lon'] - DOMAIN_BBOX['west']) / (DOMAIN_BBOX['east'] - DOMAIN_BBOX['west'])
        lat_frac = (row['lat'] - DOMAIN_BBOX['south']) / (DOMAIN_BBOX['north'] - DOMAIN_BBOX['south'])
        if lon_frac > 0.85:
            x_offset = -60
        if lat_frac > 0.90:
            y_offset = -16
        
        station_label = row.get('station', row.get('station_id', ''))
        if station_label:
            ax.annotate(f"{station_label}\n{obs:.1f}mm",
                       (row['lon'], row['lat']),
                       textcoords="offset points", xytext=(x_offset, y_offset),
                       fontsize=7, color='#333333',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', 
                                alpha=0.8, ec='gray', lw=0.5),
                       zorder=6,
                       **({k: v for k, v in scatter_kwargs.items() if k != 'transform'}))

# =============================================================================
# 2. PDF PLOTS
# =============================================================================

def plot_pdf_comparison(distributions, title="Precipitation PDF Comparison",
                        save_path=None, log_y=True):
    """
    Plot the Probability Density Functions of HRRR, AORC, and gauges.
    
    The PDF reveals differences in the precipitation intensity distribution:
    - If HRRR's PDF peaks further right, it produces more intense rain
    - If HRRR's PDF has a thinner tail, it under-predicts extreme events
    - The ideal result is HRRR's PDF matching AORC's shape closely
    
    Log-y scale is used by default because precipitation distributions are 
    highly skewed, and we care about the tails (heavy rain events).
    
    Parameters
    ----------
    distributions : dict
        Output from compute_pdf_cdf_comparison() in metrics.py
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    log_y : bool
        Use logarithmic y-axis (recommended for precipitation)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot HRRR PDF
    hrrr_pdf = distributions['hrrr_pdf']
    if len(hrrr_pdf['bin_centers']) > 0:
        ax.step(hrrr_pdf['bin_centers'], hrrr_pdf['pdf'], where='mid',
                color='#2196F3', linewidth=2.0, label=f"HRRR (n={hrrr_pdf['n_samples']})",
                alpha=0.9)
        ax.fill_between(hrrr_pdf['bin_centers'], hrrr_pdf['pdf'], 
                        step='mid', alpha=0.15, color='#2196F3')
    
    # Plot AORC PDF
    aorc_pdf = distributions['aorc_pdf']
    if len(aorc_pdf['bin_centers']) > 0:
        ax.step(aorc_pdf['bin_centers'], aorc_pdf['pdf'], where='mid',
                color='#4CAF50', linewidth=2.0, label=f"AORC (n={aorc_pdf['n_samples']})",
                alpha=0.9)
        ax.fill_between(aorc_pdf['bin_centers'], aorc_pdf['pdf'], 
                        step='mid', alpha=0.15, color='#4CAF50')
    
    # Plot Gauge PDF (if available)
    # Use KDE (kernel density estimate) instead of histogram when gauge count
    # is small, because a histogram with <30 points produces misleading spikes.
    if 'gauge_pdf' in distributions and distributions['gauge_pdf'] is not None:
        gauge_pdf = distributions['gauge_pdf']
        n_gauge = gauge_pdf['n_samples']
        if n_gauge >= 5 and len(gauge_pdf['bin_centers']) > 0:
            try:
                from scipy.stats import gaussian_kde
                # Reconstruct raw gauge values from the PDF for KDE
                # (we stored n_samples, so we use the bin_centers as proxy)
                # Better approach: pass raw gauge data. For now, use the 
                # bin_centers weighted by the pdf as an approximation.
                # Actually, just overlay vertical lines for each gauge.
                gauge_vals = distributions.get('_gauge_raw_values', None)
                if gauge_vals is not None and len(gauge_vals) >= 5:
                    kde = gaussian_kde(gauge_vals, bw_method=0.4)
                    x_kde = np.logspace(np.log10(gauge_pdf['bin_centers'][0]),
                                       np.log10(gauge_pdf['bin_centers'][-1]), 100)
                    ax.plot(x_kde, kde(x_kde), color='#FF9800', linewidth=2.0,
                            linestyle='--', label=f"Gauges KDE (n={n_gauge})", alpha=0.9)
                else:
                    # Fallback: plot histogram but with fewer bins to reduce spikiness
                    ax.step(gauge_pdf['bin_centers'], gauge_pdf['pdf'], where='mid',
                            color='#FF9800', linewidth=1.5, linestyle='--',
                            label=f"Gauges (n={n_gauge})", alpha=0.7)
            except ImportError:
                ax.step(gauge_pdf['bin_centers'], gauge_pdf['pdf'], where='mid',
                        color='#FF9800', linewidth=1.5, linestyle='--',
                        label=f"Gauges (n={n_gauge})", alpha=0.7)
    
    ax.set_xlabel('Precipitation (mm)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add KS test annotation
    ks_text = f"KS stat: {distributions['ks_statistic']}\np = {distributions['ks_pvalue']:.2e}"
    ax.text(0.02, 0.98, ks_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', fc='lightyellow', 
                                                alpha=0.8, ec='gray'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    return fig, ax

# =============================================================================
# 3. CDF PLOTS
# =============================================================================

def plot_cdf_comparison(distributions, title="Precipitation CDF Comparison",
                        save_path=None):
    """
    Plot the Cumulative Distribution Functions of HRRR, AORC, and gauges.
    
    The CDF is especially useful for identifying systematic shifts in the 
    precipitation distribution. Key things to look for:
    - If HRRR's CDF is to the right of AORC's: HRRR is wetter overall
    - The vertical gap at any precipitation value = the difference in
      exceedance probability at that amount
    - The maximum vertical gap = the KS statistic
    
    We also annotate key percentiles (P50, P90, P95, P99) for quick reference.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left panel: Full CDF ---
    hrrr_cdf = distributions['hrrr_cdf']
    aorc_cdf = distributions['aorc_cdf']
    
    if len(hrrr_cdf['values']) > 0:
        ax1.plot(hrrr_cdf['values'], hrrr_cdf['cdf'], 
                 color='#2196F3', linewidth=2.0, label='HRRR')
    if len(aorc_cdf['values']) > 0:
        ax1.plot(aorc_cdf['values'], aorc_cdf['cdf'],
                 color='#4CAF50', linewidth=2.0, label='AORC')
    
    if 'gauge_cdf' in distributions and distributions['gauge_cdf'] is not None:
        gauge_cdf = distributions['gauge_cdf']
        if len(gauge_cdf['values']) > 0:
            ax1.plot(gauge_cdf['values'], gauge_cdf['cdf'],
                     color='#FF9800', linewidth=2.0, linestyle='--', label='Gauges')
    
    ax1.set_xlabel('Precipitation (mm)', fontsize=12)
    ax1.set_ylabel('Cumulative Probability', fontsize=12)
    ax1.set_title('Full CDF', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Add percentile markers
    for pct, label in [(0.5, 'P50'), (0.9, 'P90'), (0.95, 'P95')]:
        ax1.axhline(y=pct, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        ax1.text(ax1.get_xlim()[1] * 0.95, pct, label, fontsize=8, 
                 color='gray', ha='right', va='bottom')
    
    # --- Right panel: Upper tail CDF (exceedance probability) ---
    # This zooms in on the heavy rain events (most meteorologically interesting)
    if len(hrrr_cdf['values']) > 0:
        ax2.plot(hrrr_cdf['values'], 1 - hrrr_cdf['cdf'],
                 color='#2196F3', linewidth=2.0, label='HRRR')
    if len(aorc_cdf['values']) > 0:
        ax2.plot(aorc_cdf['values'], 1 - aorc_cdf['cdf'],
                 color='#4CAF50', linewidth=2.0, label='AORC')
    
    if 'gauge_cdf' in distributions and distributions['gauge_cdf'] is not None:
        gauge_cdf = distributions['gauge_cdf']
        if len(gauge_cdf['values']) > 0:
            ax2.plot(gauge_cdf['values'], 1 - gauge_cdf['cdf'],
                     color='#FF9800', linewidth=2.0, linestyle='--', label='Gauges')
    
    ax2.set_xlabel('Precipitation (mm)', fontsize=12)
    ax2.set_ylabel('Exceedance Probability (1 − CDF)', fontsize=12)
    ax2.set_title('Upper Tail (Heavy Rain)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add percentile annotation table
    pct_text = "Percentiles (mm):\n"
    pct_text += f"{'':>6s}  {'HRRR':>6s}  {'AORC':>6s}\n"
    pct_text += f"{'P50':>6s}  {hrrr_cdf['median']:6.1f}  {aorc_cdf['median']:6.1f}\n"
    pct_text += f"{'P90':>6s}  {hrrr_cdf['p90']:6.1f}  {aorc_cdf['p90']:6.1f}\n"
    pct_text += f"{'P95':>6s}  {hrrr_cdf['p95']:6.1f}  {aorc_cdf['p95']:6.1f}\n"
    pct_text += f"{'P99':>6s}  {hrrr_cdf['p99']:6.1f}  {aorc_cdf['p99']:6.1f}"
    ax2.text(0.98, 0.98, pct_text, transform=ax2.transAxes, fontsize=8,
             verticalalignment='top', horizontalalignment='right',
             family='monospace',
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9, ec='gray'))
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    return fig, (ax1, ax2)

# =============================================================================
# 4. GAUGE COMPARISON PLOTS
# =============================================================================

def plot_gauge_bias_bars(gauge_detail_df, save_path=None):
    """
    Bar chart showing per-station bias (HRRR − Observed and AORC − Observed).
    
    Positive bars = over-prediction, negative = under-prediction.
    Stations are sorted by observed amount for readability.
    """
    if gauge_detail_df is None or len(gauge_detail_df) == 0:
        return None, None
    
    df = gauge_detail_df.sort_values('observed_mm', ascending=True)
    stations = df['station'].values
    y_pos = np.arange(len(stations))
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(stations) * 0.5)))
    
    bar_height = 0.35
    ax.barh(y_pos - bar_height/2, df['hrrr_error_mm'].values, bar_height,
            label='HRRR − Observed', color='#2196F3', alpha=0.8, edgecolor='white')
    ax.barh(y_pos + bar_height/2, df['aorc_error_mm'].values, bar_height,
            label='AORC − Observed', color='#4CAF50', alpha=0.8, edgecolor='white')
    
    # Zero line
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s} ({o:.0f}mm obs)" for s, o in 
                        zip(stations, df['observed_mm'].values)], fontsize=9)
    ax.set_xlabel('Bias (mm) — Predicted minus Observed', fontsize=11)
    ax.set_title('Per-Station Precipitation Bias', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    return fig, ax

# =============================================================================
# 4b. HOURLY TIME SERIES
# =============================================================================

def plot_hourly_timeseries(hourly_data, event_label="", save_path=None):
    """
    Plot cumulative precipitation time series: HRRR domain-mean, AORC domain-mean,
    and each gauge station. Shows how precipitation accumulates hour by hour
    and where HRRR diverges from observations.
    
    Parameters
    ----------
    hourly_data : dict
        Output from extract_hourly_series() with hours, domain_mean, stations.
    event_label : str
        Event description for the title.
    save_path : str, optional
        Where to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1])
    
    hours = [h['hour'] for h in hourly_data['hours']]
    n_hours = len(hours)
    dm = hourly_data.get('domain_mean', hourly_data.get('domain_median', {}))
    stations = hourly_data['stations']
    
    # --- Top panel: Cumulative precipitation ---
    # HRRR and AORC domain means (thick lines)
    ax1.plot(hours, dm['hrrr_cumul'], color='#2196F3', linewidth=3,
             label='HRRR domain mean', zorder=10)
    ax1.plot(hours, dm['aorc_cumul'], color='#4CAF50', linewidth=3,
             label='AORC domain mean', zorder=10)
    
    # Individual gauge stations (thinner lines)
    station_colors = plt.cm.Set2(np.linspace(0, 1, max(len(stations), 1)))
    for i, (sid, sdata) in enumerate(sorted(stations.items())):
        color = station_colors[i % len(station_colors)]
        ax1.plot(hours, sdata['obs_cumul'], color=color, linewidth=1.5,
                 alpha=0.7, linestyle='-', marker='.', markersize=4,
                 label=f'{sid} (gauge)')
    
    ax1.set_ylabel('Cumulative Precipitation (mm)', fontsize=11)
    ax1.set_title(f'Hourly Accumulation — {event_label}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left', ncol=3, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_hours - 1)
    
    # Add hour labels on x-axis
    tick_labels = []
    for h in hourly_data['hours']:
        tick_labels.append(h.get('et', f'H{h["hour"]}'))
    if n_hours <= 24:
        ax1.set_xticks(hours[::3])
        ax1.set_xticklabels([tick_labels[i] for i in range(0, n_hours, 3)],
                            fontsize=8, rotation=30, ha='right')
    
    # --- Bottom panel: Hourly increments (bar chart) ---
    bar_width = 0.35
    x = np.array(hours)
    ax2.bar(x - bar_width/2, dm['hrrr_hourly'], bar_width,
            color='#2196F3', alpha=0.7, label='HRRR hourly')
    ax2.bar(x + bar_width/2, dm['aorc_hourly'], bar_width,
            color='#4CAF50', alpha=0.7, label='AORC hourly')
    
    ax2.set_xlabel('Hour of Event', fontsize=11)
    ax2.set_ylabel('Hourly Precip (mm)', fontsize=11)
    ax2.set_title('Hourly Precipitation Rate (Domain Mean)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, n_hours - 0.5)
    
    if n_hours <= 24:
        ax2.set_xticks(hours[::3])
        ax2.set_xticklabels([tick_labels[i] for i in range(0, n_hours, 3)],
                            fontsize=8, rotation=30, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved time series: {save_path}")
    
    return fig

# =============================================================================
# 5. FRACTIONS SKILL SCORE PLOT
# =============================================================================

def plot_fss(fss_data, event_label="", save_path=None):
    """
    Plot Fractions Skill Score vs neighborhood scale for all thresholds.
    
    This is the key spatial verification figure. It answers: "At what spatial
    scale does HRRR become useful for predicting precipitation above each
    threshold?" The conventional target is FSS >= 0.5 ("useful" skill).
    
    X-axis: neighborhood size in km (0 to domain width)
    Y-axis: FSS (0 to 1)
    One line per precipitation threshold, with the FSS=0.5 reference line.
    
    Parameters
    ----------
    fss_data : dict
        Output from compute_fss() containing thresholds, neighborhood sizes,
        and FSS values.
    event_label : str
        Event description for the title.
    save_path : str, optional
        Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    
    neighborhood_km = fss_data['neighborhood_km']
    thresholds = fss_data['thresholds_mm']
    
    # Color palette — distinct colors for each threshold (8 WPC standard)
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
              '#00BCD4', '#FF5722', '#795548']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
    
    for i, thresh in enumerate(thresholds):
        fss_vals = fss_data['fss'].get(thresh, fss_data['fss'].get(float(thresh), []))
        if not fss_vals:
            continue
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        useful_km = fss_data['useful_scale_km'].get(thresh, 
                    fss_data['useful_scale_km'].get(float(thresh)))
        
        # Map mm to inch label for WPC convention
        inch_map = {0.254: '0.01"', 2.54: '0.10"', 6.35: '0.25"', 12.7: '0.50"',
                    25.4: '1.00"', 50.8: '2.00"', 76.2: '3.00"', 101.6: '4.00"'}
        inch_label = inch_map.get(thresh, '')
        
        label = f'{thresh}mm ({inch_label})' if inch_label else f'{thresh}mm'
        if useful_km is not None:
            label += f'  → {useful_km}km'
        else:
            label += '  → > domain'
        
        ax.plot(neighborhood_km, fss_vals, color=color, linewidth=2.0,
                marker=marker, markersize=5, markeredgecolor='white',
                markeredgewidth=0.8, label=label, alpha=0.9)
    
    # FSS = 0.5 "useful skill" reference line
    ax.axhline(y=0.5, color='#666666', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Useful skill (FSS=0.5)')
    
    # FSS = 1.0 perfect line
    ax.axhline(y=1.0, color='#999999', linestyle=':', linewidth=0.8, alpha=0.4)
    
    ax.set_xlabel('Neighborhood Scale (km)', fontsize=11)
    ax.set_ylabel('Fractions Skill Score', fontsize=11)
    ax.set_title(f'FSS vs Spatial Scale — {event_label}', fontsize=13, fontweight='bold')
    
    ax.set_xlim(0, max(neighborhood_km) * 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9, ncol=1)
    ax.grid(True, alpha=0.3)
    
    # Add secondary annotation
    ax.text(0.02, 0.97, 
            f'Grid: ~3km HRRR\nDomain: {max(neighborhood_km):.0f}km\n'
            f'Thresholds: {len(thresholds)}',
            transform=ax.transAxes, fontsize=8, color='#888888',
            verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved FSS plot: {save_path}")
    
    return fig

# =============================================================================
# 6. MULTI-PANEL DASHBOARD
# =============================================================================

def plot_stats_panel(metrics_dict, gauge_accum_df=None, event_label="",
                     accum_hours=24, save_path=None):
    """
    Generate the 3-panel statistics row: PDF, CDF, Gauge scatter.
    Previously the middle row of the combined dashboard.
    """
    fig, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(20, 6))
    
    fig.suptitle(f"Distribution & Gauge Comparison — {event_label}\n"
                 f"{accum_hours}-hour Accumulation", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    dist = metrics_dict['distributions']
    
    # Panel 1: PDF
    if len(dist['hrrr_pdf']['bin_centers']) > 0:
        ax4.step(dist['hrrr_pdf']['bin_centers'], dist['hrrr_pdf']['pdf'],
                 where='mid', color='#2196F3', linewidth=2.0, label='HRRR')
        ax4.fill_between(dist['hrrr_pdf']['bin_centers'], dist['hrrr_pdf']['pdf'],
                         step='mid', alpha=0.15, color='#2196F3')
    if len(dist['aorc_pdf']['bin_centers']) > 0:
        ax4.step(dist['aorc_pdf']['bin_centers'], dist['aorc_pdf']['pdf'],
                 where='mid', color='#4CAF50', linewidth=2.0, label='AORC')
        ax4.fill_between(dist['aorc_pdf']['bin_centers'], dist['aorc_pdf']['pdf'],
                         step='mid', alpha=0.15, color='#4CAF50')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Precip (mm)', fontsize=10)
    ax4.set_ylabel('PDF', fontsize=10)
    ax4.set_title('Probability Density Function', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    
    # Panel 2: CDF
    if len(dist['hrrr_cdf']['values']) > 0:
        ax5.plot(dist['hrrr_cdf']['values'], dist['hrrr_cdf']['cdf'],
                 color='#2196F3', linewidth=2.0, label='HRRR')
    if len(dist['aorc_cdf']['values']) > 0:
        ax5.plot(dist['aorc_cdf']['values'], dist['aorc_cdf']['cdf'],
                 color='#4CAF50', linewidth=2.0, label='AORC')
    ax5.set_xlabel('Precip (mm)', fontsize=10)
    ax5.set_ylabel('CDF', fontsize=10)
    ax5.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Panel 3: Gauge scatter
    gd = metrics_dict.get('gauge_detail')
    if gd is not None and len(gd) > 0:
        obs_vals = gd['observed_mm'].values
        hrrr_vals = gd['hrrr_mm'].values
        max_v = max(obs_vals.max(), hrrr_vals.max()) * 1.1
        ax6.scatter(obs_vals, hrrr_vals, s=80, c='#2196F3', edgecolors='white',
                    linewidths=0.8, alpha=0.8)
        ax6.plot([0, max_v], [0, max_v], 'k--', alpha=0.5, linewidth=1)
        ax6.set_xlim(0, max_v)
        ax6.set_ylim(0, max_v)
        ax6.set_aspect('equal')
        for _, row in gd.iterrows():
            ax6.annotate(row.get('station', ''), (row['observed_mm'], row['hrrr_mm']),
                         fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')
    ax6.set_xlabel('Observed (mm)', fontsize=10)
    ax6.set_ylabel('HRRR (mm)', fontsize=10)
    ax6.set_title('HRRR vs Gauge Obs', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved stats panel: {save_path}")
    
    return fig

# Keep the old function for backwards compatibility but have it call the new ones
# =============================================================================
# CONVENIENCE: Generate all plots for an event
# =============================================================================

def generate_all_plots(hrrr_accum, aorc_accum, lat, lon, metrics_dict,
                       gauge_accum_df=None, event_label="", accum_hours=24,
                       save_dir=None, show=True):
    """
    Generate and optionally save all verification plots for a single event.
    
    This is the main entry point for the visualization module. It produces:
    1. HRRR precipitation map
    2. AORC precipitation map 
    3. Difference map (HRRR - AORC)
    4. PDF comparison
    5. CDF comparison
    6. Gauge scatter plot
    7. Gauge bias bar chart
    8. Fractions Skill Score (FSS) vs spatial scale
    9-11. Split dashboard panels (maps, stats, metrics text)
    
    Parameters
    ----------
    save_dir : str, optional
        Directory to save figures. Defaults to FIGURES_DIR from config.
    show : bool
        If True, display plots interactively (for Spyder)
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    prefix = event_label.replace(" ", "_").replace("/", "-") if event_label else "event"
    
    print("\n" + "=" * 60)
    print(f"GENERATING VERIFICATION PLOTS")
    print(f"  Event: {event_label}")
    print(f"  Output: {save_dir}")
    print("=" * 60)
    
    # 1. HRRR map
    print("\n[1/8] HRRR precipitation map...")
    plot_precip_field(hrrr_accum, lat, lon,
                      title=f"HRRR {accum_hours}h Forecast — {event_label}",
                      gauge_df=gauge_accum_df,
                      save_path=os.path.join(save_dir, f"{prefix}_hrrr.png"))
    
    # 2. AORC map
    print("[2/8] AORC precipitation map...")
    plot_precip_field(aorc_accum, lat, lon,
                      title=f"AORC {accum_hours}h — {event_label}",
                      gauge_df=gauge_accum_df,
                      save_path=os.path.join(save_dir, f"{prefix}_aorc.png"))
    
    # 3. Difference map
    print("[3/8] Difference map...")
    plot_difference_field(metrics_dict['spatial']['difference'], lat, lon,
                         title=f"HRRR − AORC ({accum_hours}h) — {event_label}",
                         gauge_df=gauge_accum_df,
                         save_path=os.path.join(save_dir, f"{prefix}_diff.png"))
    
    # 4. PDF comparison
    print("[4/8] PDF comparison...")
    plot_pdf_comparison(metrics_dict['distributions'],
                        title=f"PDF Comparison ({accum_hours}h) — {event_label}",
                        save_path=os.path.join(save_dir, f"{prefix}_pdf.png"))
    
    # 5. CDF comparison
    print("[5/8] CDF comparison...")
    plot_cdf_comparison(metrics_dict['distributions'],
                        title=f"CDF Comparison ({accum_hours}h) — {event_label}",
                        save_path=os.path.join(save_dir, f"{prefix}_cdf.png"))
    
    # 6-7. Gauge plots
    gd = metrics_dict.get('gauge_detail')
    if gd is not None and len(gd) > 0:
        print("[6/8] Gauge bias bars...")
        plot_gauge_bias_bars(gd, save_path=os.path.join(save_dir, f"{prefix}_gauge_bias.png"))
    else:
        print("[6/8] Gauge bias bars... SKIPPED (no gauge data)")
    
    # 8. FSS plot
    fss_data = metrics_dict.get('fss')
    if fss_data is not None:
        print("[7/8] Fractions Skill Score...")
        plot_fss(fss_data, event_label=event_label,
                 save_path=os.path.join(save_dir, f"{prefix}_fss.png"))
    else:
        print("[7/8] FSS... SKIPPED (not computed)")
    
    # 8. Stats panel (PDF / CDF / Scatter summary)
    print("[8/8] Stats panel...")
    plot_stats_panel(metrics_dict, gauge_accum_df=gauge_accum_df,
                     event_label=event_label, accum_hours=accum_hours,
                     save_path=os.path.join(save_dir, f"{prefix}_stats_panel.png"))
    
    # Generate transparent map overlays for Leaflet
    print("\n[+] Generating map overlays for interactive viewer...")
    overlays = generate_all_overlays(
        hrrr_accum, aorc_accum, lat, lon, metrics_dict,
        save_dir=save_dir, prefix=prefix
    )
    
    print("\n" + "=" * 60)
    print(f"ALL PLOTS GENERATED ✓")
    print(f"  Saved to: {save_dir}")
    print("=" * 60)
    
    if show:
        plt.show()
    
    return overlays

# =============================================================================
if __name__ == "__main__":
    print("Visualization module loaded.")
    print(f"Cartopy available: {HAS_CARTOPY}")
    print(f"Figures directory: {FIGURES_DIR}")
