"""
=============================================================================
HRRR vs AORC Precipitation Verification Dashboard — Metrics Engine
=============================================================================
Computes all verification metrics for comparing HRRR forecasts against 
AORC analysis and rain gauge observations.

Metric categories:
  1. CONTINUOUS metrics (Bias, MAE, RMSE, Correlation)
  2. CATEGORICAL metrics (POD, FAR, CSI, ETS) at configurable thresholds
  3. DISTRIBUTIONAL analysis (PDFs and CDFs) for comparing precipitation
     distributions between HRRR, AORC, and gauges
  4. GAUGE-SPECIFIC metrics (point-based comparison at station locations)
  5. SPATIAL metrics (per-grid-cell statistics for map overlays)

Future: Fractions Skill Score (FSS) and ML-based bias correction model
will be added in later phases.
=============================================================================
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# =============================================================================
# 1. CONTINUOUS VERIFICATION METRICS (Domain-wide)
# =============================================================================

def compute_continuous_metrics(hrrr, aorc, mask=None):
    """
    Compute continuous (non-threshold) verification metrics over the full domain.
    
    These metrics tell you about the overall magnitude and pattern accuracy
    of the HRRR forecast compared to the AORC analysis.
    
    Parameters
    ----------
    hrrr : np.ndarray or xr.DataArray
        2D array of HRRR accumulated precipitation (y, x)
    aorc : np.ndarray or xr.DataArray
        2D array of AORC accumulated precipitation on the same grid (y, x)
    mask : np.ndarray, optional
        Boolean mask — True where data should be included, False where excluded.
        Useful for masking out ocean cells, boundary cells, or sub-regions.
    
    Returns
    -------
    dict
        Dictionary of metric names → values
    """
    # Convert to numpy arrays
    h = np.asarray(hrrr, dtype=float).ravel()
    a = np.asarray(aorc, dtype=float).ravel()
    
    # Apply mask if provided
    if mask is not None:
        m = np.asarray(mask).ravel()
        h = h[m]
        a = a[m]
    
    # Remove NaN pairs
    valid = ~(np.isnan(h) | np.isnan(a))
    h = h[valid]
    a = a[valid]
    
    n = len(h)
    if n == 0:
        return {k: np.nan for k in ['n_cells', 'mean_hrrr', 'mean_aorc',
                'mult_bias', 'add_bias', 'mae', 'rmse', 'correlation',
                'std_hrrr', 'std_aorc', 'coeff_variation_hrrr', 'coeff_variation_aorc']}
    
    # --- Core statistics ---
    mean_h = np.mean(h)
    mean_a = np.mean(a)
    std_h = np.std(h, ddof=1) if n > 1 else 0
    std_a = np.std(a, ddof=1) if n > 1 else 0
    
    # --- Multiplicative Bias ---
    # Ratio of forecast mean to observed mean. Perfect = 1.0
    # > 1 means HRRR over-predicts on average; < 1 means under-predicts
    mult_bias = mean_h / mean_a if mean_a > 0 else np.nan
    
    # --- Additive (Mean) Bias ---
    # Average difference (HRRR - AORC). Perfect = 0
    add_bias = mean_h - mean_a
    
    # --- Mean Absolute Error (MAE) ---
    # Average magnitude of errors, regardless of sign. Lower is better.
    mae = np.mean(np.abs(h - a))
    
    # --- Root Mean Square Error (RMSE) ---
    # Like MAE but penalizes large errors more heavily. Lower is better.
    rmse = np.sqrt(np.mean((h - a) ** 2))
    
    # --- Pearson Correlation ---
    # Measures how well the spatial pattern matches. Perfect = 1.0
    # Note: correlation can be high even if magnitudes are completely wrong,
    # so it should always be used alongside bias/MAE/RMSE.
    if std_h > 0 and std_a > 0:
        correlation = np.corrcoef(h, a)[0, 1]
    else:
        correlation = np.nan
    
    # --- Coefficient of Variation ---
    # Normalized measure of spread — useful for comparing variability
    cv_h = std_h / mean_h if mean_h > 0 else np.nan
    cv_a = std_a / mean_a if mean_a > 0 else np.nan
    
    return {
        'n_cells': n,
        'mean_hrrr': round(mean_h, 2),
        'mean_aorc': round(mean_a, 2),
        'mult_bias': round(mult_bias, 3),
        'add_bias': round(add_bias, 2),
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'correlation': round(correlation, 4),
        'std_hrrr': round(std_h, 2),
        'std_aorc': round(std_a, 2),
        'coeff_variation_hrrr': round(cv_h, 3) if not np.isnan(cv_h) else np.nan,
        'coeff_variation_aorc': round(cv_a, 3) if not np.isnan(cv_a) else np.nan,
    }

# =============================================================================
# 2. CATEGORICAL VERIFICATION METRICS (Threshold-based)
# =============================================================================

def compute_categorical_metrics(hrrr, aorc, threshold_mm, mask=None):
    """
    Compute categorical (yes/no) verification metrics at a given 
    precipitation threshold.
    
    This answers the question: "For rain events exceeding X mm, how well
    did HRRR predict their occurrence?"
    
    The 2×2 contingency table:
                        AORC ≥ threshold    AORC < threshold
    HRRR ≥ threshold:      HIT (a)           FALSE ALARM (b)
    HRRR < threshold:      MISS (c)          CORRECT NEGATIVE (d)
    
    Parameters
    ----------
    hrrr, aorc : np.ndarray or xr.DataArray
        2D accumulated precipitation arrays (must be on same grid)
    threshold_mm : float
        Precipitation threshold in mm (e.g., 2.54 for 0.1 inch)
    mask : np.ndarray, optional
        Boolean mask for sub-region analysis
    
    Returns
    -------
    dict
        Categorical metrics including POD, FAR, CSI, ETS, and the
        raw contingency table counts
    """
    h = np.asarray(hrrr, dtype=float).ravel()
    a = np.asarray(aorc, dtype=float).ravel()
    
    if mask is not None:
        m = np.asarray(mask).ravel()
        h = h[m]
        a = a[m]
    
    valid = ~(np.isnan(h) | np.isnan(a))
    h = h[valid]
    a = a[valid]
    n = len(h)
    
    if n == 0:
        return {k: np.nan for k in ['threshold_mm', 'hits', 'misses', 
                'false_alarms', 'correct_negatives', 'pod', 'far', 'csi',
                'ets', 'frequency_bias', 'accuracy']}
    
    # Build contingency table
    h_yes = h >= threshold_mm
    a_yes = a >= threshold_mm
    
    hits = np.sum(h_yes & a_yes)               # Both say rain
    misses = np.sum(~h_yes & a_yes)             # AORC says rain, HRRR missed it
    false_alarms = np.sum(h_yes & ~a_yes)       # HRRR says rain, AORC doesn't
    correct_negatives = np.sum(~h_yes & ~a_yes) # Both say no rain
    
    total = hits + misses + false_alarms + correct_negatives
    
    # --- Probability of Detection (POD / Hit Rate) ---
    # Of all the times it actually rained (AORC), what fraction did HRRR catch?
    # Range: 0 to 1. Perfect = 1. Also called "sensitivity" or "recall".
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    
    # --- False Alarm Ratio (FAR) ---
    # Of all times HRRR predicted rain, what fraction was wrong?
    # Range: 0 to 1. Perfect = 0. (Not the same as False Alarm Rate!)
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    
    # --- Critical Success Index (CSI / Threat Score) ---
    # Overall skill at predicting rain events, ignoring correct negatives.
    # Range: 0 to 1. Perfect = 1.
    # This is the "go-to" metric for precipitation verification.
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    
    # --- Equitable Threat Score (ETS / Gilbert Skill Score) ---
    # Like CSI but adjusted for hits expected by random chance.
    # Range: -1/3 to 1. Perfect = 1. ETS = 0 means no skill beyond chance.
    hits_random = (hits + misses) * (hits + false_alarms) / total if total > 0 else 0
    denom = hits + misses + false_alarms - hits_random
    ets = (hits - hits_random) / denom if denom > 0 else np.nan
    
    # --- Frequency Bias ---
    # Ratio of predicted events to observed events.
    # > 1 means HRRR over-predicts the frequency of rain.
    # < 1 means HRRR under-predicts the frequency of rain.
    freq_bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else np.nan
    
    # --- Overall Accuracy ---
    # Fraction of all cells correctly classified. Can be misleading for rare events.
    accuracy = (hits + correct_negatives) / total if total > 0 else np.nan
    
    return {
        'threshold_mm': threshold_mm,
        'hits': int(hits),
        'misses': int(misses),
        'false_alarms': int(false_alarms),
        'correct_negatives': int(correct_negatives),
        'pod': round(pod, 4) if not np.isnan(pod) else np.nan,
        'far': round(far, 4) if not np.isnan(far) else np.nan,
        'csi': round(csi, 4) if not np.isnan(csi) else np.nan,
        'ets': round(ets, 4) if not np.isnan(ets) else np.nan,
        'frequency_bias': round(freq_bias, 3) if not np.isnan(freq_bias) else np.nan,
        'accuracy': round(accuracy, 4) if not np.isnan(accuracy) else np.nan,
    }

def compute_categorical_multi_threshold(hrrr, aorc, thresholds=None, mask=None):
    """
    Compute categorical metrics at multiple thresholds.
    This is useful for understanding how skill varies with rain intensity.
    
    Parameters
    ----------
    thresholds : list of float, optional
        Thresholds in mm. Defaults to [0.254, 2.54, 6.35, 12.7, 25.4]
        which correspond to 0.01", 0.1", 0.25", 0.5", 1.0" in inches.
    
    Returns
    -------
    pd.DataFrame
        One row per threshold with all categorical metrics
    """
    from config import PRECIP_THRESHOLDS
    
    if thresholds is None:
        thresholds = PRECIP_THRESHOLDS
    
    results = []
    for thresh in thresholds:
        metrics = compute_categorical_metrics(hrrr, aorc, thresh, mask)
        metrics['threshold_inches'] = round(thresh / 25.4, 3)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    return df

# =============================================================================
# 3. PROBABILITY DENSITY FUNCTIONS (PDFs) AND CUMULATIVE DISTRIBUTION 
#    FUNCTIONS (CDFs)
# =============================================================================

def compute_pdf(data, bins=None, min_val=0.1, max_val=None, n_bins=50,
                log_bins=True, density=True):
    """
    Compute the Probability Density Function (PDF) of precipitation values.
    
    The PDF shows the relative frequency of different precipitation amounts.
    For precipitation, log-spaced bins are often more informative because 
    the distribution is highly skewed (lots of light rain, little heavy rain).
    
    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Precipitation values (1D or 2D, will be flattened)
    bins : np.ndarray, optional
        Custom bin edges. If None, bins are auto-generated.
    min_val : float
        Minimum precipitation value to include (filters out zeros/trace)
    max_val : float, optional
        Maximum value. If None, uses data max.
    n_bins : int
        Number of bins
    log_bins : bool
        If True, use logarithmically-spaced bins (better for precip)
    density : bool
        If True, normalize to probability density. If False, raw counts.
    
    Returns
    -------
    dict with keys:
        'bin_centers': np.ndarray — center of each bin
        'bin_edges': np.ndarray — edges of bins (len = n_bins + 1)
        'pdf': np.ndarray — probability density (or counts if density=False)
        'n_samples': int — total number of valid samples
        'n_zeros': int — number of zero/trace values excluded
    """
    values = np.asarray(data, dtype=float).ravel()
    
    # Count and remove zeros/trace amounts
    n_total = len(values)
    n_nan = np.isnan(values).sum()
    values = values[~np.isnan(values)]
    n_zeros = np.sum(values < min_val)
    values = values[values >= min_val]
    
    if len(values) == 0:
        return {'bin_centers': np.array([]), 'bin_edges': np.array([]),
                'pdf': np.array([]), 'n_samples': 0, 'n_zeros': int(n_zeros)}
    
    if max_val is None:
        max_val = np.percentile(values, 99.5)  # Trim extreme outliers
    
    values = values[values <= max_val]
    
    # Generate bins
    if bins is None:
        if log_bins:
            bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)
        else:
            bins = np.linspace(min_val, max_val, n_bins + 1)
    
    # Compute histogram
    counts, bin_edges = np.histogram(values, bins=bins, density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return {
        'bin_centers': bin_centers,
        'bin_edges': bin_edges,
        'pdf': counts,
        'n_samples': len(values),
        'n_zeros': int(n_zeros),
    }

def compute_cdf(data, min_val=0.0, max_val=None, n_points=200):
    """
    Compute the Cumulative Distribution Function (CDF) of precipitation values.
    
    The CDF shows the probability that precipitation is less than or equal to
    a given value. Comparing HRRR and AORC CDFs reveals systematic differences
    in the precipitation distribution — for example, if HRRR's CDF is shifted
    right, it produces heavier rainfall more often than AORC.
    
    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Precipitation values
    min_val : float
        Minimum value for the CDF x-axis
    max_val : float, optional
        Maximum value. If None, uses data max.
    n_points : int
        Number of points to evaluate the CDF at
    
    Returns
    -------
    dict with keys:
        'values': np.ndarray — precipitation values (x-axis)
        'cdf': np.ndarray — cumulative probability (0 to 1)
        'n_samples': int — number of valid samples
        'median': float — 50th percentile
        'p90': float — 90th percentile
        'p95': float — 95th percentile
        'p99': float — 99th percentile
    """
    values = np.asarray(data, dtype=float).ravel()
    values = values[~np.isnan(values)]
    values = values[values >= min_val]
    
    if len(values) == 0:
        return {'values': np.array([]), 'cdf': np.array([]),
                'n_samples': 0, 'median': np.nan, 'p90': np.nan,
                'p95': np.nan, 'p99': np.nan}
    
    # Sort for empirical CDF
    sorted_vals = np.sort(values)
    
    if max_val is None:
        max_val = sorted_vals[-1]
    
    # Create evenly-spaced evaluation points
    x = np.linspace(min_val, max_val, n_points)
    
    # Empirical CDF: fraction of values ≤ x
    cdf = np.searchsorted(sorted_vals, x, side='right') / len(sorted_vals)
    
    # Key percentiles
    percentiles = np.percentile(values, [50, 90, 95, 99])
    
    return {
        'values': x,
        'cdf': cdf,
        'n_samples': len(values),
        'median': round(percentiles[0], 2),
        'p90': round(percentiles[1], 2),
        'p95': round(percentiles[2], 2),
        'p99': round(percentiles[3], 2),
    }

def compute_pdf_cdf_comparison(hrrr, aorc, gauges_accum=None, 
                                min_val=0.1, n_bins=50):
    """
    Compute and compare PDFs and CDFs between HRRR, AORC, and optionally gauges.
    
    This is the main distributional analysis function. It returns everything
    needed to plot comparative PDF and CDF figures.
    
    Parameters
    ----------
    hrrr : np.ndarray or xr.DataArray
        HRRR accumulated precipitation
    aorc : np.ndarray or xr.DataArray
        AORC accumulated precipitation (on HRRR grid)
    gauges_accum : pd.DataFrame, optional
        Accumulated gauge data with 'total_precip_mm' column
    min_val : float
        Minimum precip to include in distributions (filter out dry cells)
    n_bins : int
        Number of bins for the PDF
    
    Returns
    -------
    dict with keys:
        'hrrr_pdf', 'aorc_pdf', 'gauge_pdf' — PDF results
        'hrrr_cdf', 'aorc_cdf', 'gauge_cdf' — CDF results
        'ks_statistic' — Kolmogorov-Smirnov test statistic (HRRR vs AORC)
        'ks_pvalue' — p-value of the KS test
    """
    # Compute PDFs
    # First pass: determine common bin edges from combined data range
    all_vals = np.concatenate([
        np.asarray(hrrr).ravel()[np.asarray(hrrr).ravel() >= min_val],
        np.asarray(aorc).ravel()[np.asarray(aorc).ravel() >= min_val],
    ])
    if len(all_vals) > 0:
        max_val = np.percentile(all_vals, 99.5)
        common_bins = np.logspace(np.log10(min_val), np.log10(max(max_val, min_val * 10)), 
                                  n_bins + 1)
    else:
        common_bins = np.logspace(np.log10(min_val), np.log10(100), n_bins + 1)
    
    hrrr_pdf = compute_pdf(hrrr, bins=common_bins, min_val=min_val)
    aorc_pdf = compute_pdf(aorc, bins=common_bins, min_val=min_val)
    
    # Compute CDFs
    max_cdf = np.percentile(all_vals, 99.9) if len(all_vals) > 0 else 100
    hrrr_cdf = compute_cdf(hrrr, min_val=0, max_val=max_cdf)
    aorc_cdf = compute_cdf(aorc, min_val=0, max_val=max_cdf)
    
    # Kolmogorov-Smirnov test: are the two distributions significantly different?
    # A small p-value (< 0.05) means the distributions are statistically different.
    h_flat = np.asarray(hrrr).ravel()
    a_flat = np.asarray(aorc).ravel()
    h_valid = h_flat[~np.isnan(h_flat)]
    a_valid = a_flat[~np.isnan(a_flat)]
    
    if len(h_valid) > 0 and len(a_valid) > 0:
        ks_stat, ks_pval = stats.ks_2samp(h_valid, a_valid)
    else:
        ks_stat, ks_pval = np.nan, np.nan
    
    result = {
        'hrrr_pdf': hrrr_pdf,
        'aorc_pdf': aorc_pdf,
        'hrrr_cdf': hrrr_cdf,
        'aorc_cdf': aorc_cdf,
        'ks_statistic': round(ks_stat, 4),
        'ks_pvalue': ks_pval,
    }
    
    # Gauge distributions (if available)
    if gauges_accum is not None and 'total_precip_mm' in gauges_accum.columns:
        gauge_vals = gauges_accum['total_precip_mm'].dropna().values
        if len(gauge_vals) > 3:
            result['gauge_pdf'] = compute_pdf(gauge_vals, bins=common_bins, min_val=min_val)
            result['gauge_cdf'] = compute_cdf(gauge_vals, min_val=0, max_val=max_cdf)
            # Store raw values so the visualization can use KDE for small samples
            result['_gauge_raw_values'] = gauge_vals
        else:
            result['gauge_pdf'] = None
            result['gauge_cdf'] = None
            result['_gauge_raw_values'] = None
            print("  [INFO] Too few gauge values for distribution analysis")
    
    return result

# =============================================================================
# 4. GAUGE-SPECIFIC METRICS (Point comparison)
# =============================================================================

def compute_gauge_metrics(hrrr, aorc, gauge_accum_df, hrrr_lat, hrrr_lon):
    """
    Compare HRRR and AORC to rain gauge observations at each station location.
    
    For each gauge, we find the nearest HRRR/AORC grid cell and compare
    the gridded value to the observed gauge total. This gives us "ground truth"
    validation that doesn't rely on AORC being correct.
    
    Parameters
    ----------
    hrrr : np.ndarray or xr.DataArray
        2D HRRR accumulated precipitation (y, x)
    aorc : np.ndarray or xr.DataArray
        2D AORC accumulated precipitation on HRRR grid (y, x)
    gauge_accum_df : pd.DataFrame
        Accumulated gauge data with columns: station, lat, lon, total_precip_mm
    hrrr_lat, hrrr_lon : np.ndarray
        2D arrays of HRRR grid cell latitudes and longitudes
    
    Returns
    -------
    pd.DataFrame
        One row per gauge station with observed, HRRR, AORC values and errors
    dict
        Domain-aggregate gauge metrics (mean bias, MAE, correlation)
    """
    h = np.asarray(hrrr, dtype=float)
    a = np.asarray(aorc, dtype=float)
    lat = np.asarray(hrrr_lat, dtype=float)
    lon = np.asarray(hrrr_lon, dtype=float)
    
    results = []
    
    for _, gauge in gauge_accum_df.iterrows():
        g_lat = gauge['lat']
        g_lon = gauge['lon']
        g_obs = gauge['total_precip_mm']
        
        if np.isnan(g_obs):
            continue
        
        # Find nearest grid cell using Euclidean distance on lat/lon
        # (accurate enough for ~100km domains)
        if lat.ndim == 2:
            dist = np.sqrt((lat - g_lat)**2 + (lon - g_lon)**2)
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            h_val = h[idx]
            a_val = a[idx]
            cell_lat = lat[idx]
            cell_lon = lon[idx]
        else:
            # 1D lat/lon (regular grid after regridding)
            lat_idx = np.argmin(np.abs(lat - g_lat))
            lon_idx = np.argmin(np.abs(lon - g_lon))
            h_val = h[lat_idx, lon_idx]
            a_val = a[lat_idx, lon_idx]
            cell_lat = lat[lat_idx]
            cell_lon = lon[lon_idx]
        
        results.append({
            'station': gauge.get('station', gauge.name),
            'gauge_lat': g_lat,
            'gauge_lon': g_lon,
            'cell_lat': float(cell_lat),
            'cell_lon': float(cell_lon),
            'observed_mm': round(g_obs, 1),
            'hrrr_mm': round(float(h_val), 1),
            'aorc_mm': round(float(a_val), 1),
            'hrrr_error_mm': round(float(h_val) - g_obs, 1),
            'aorc_error_mm': round(float(a_val) - g_obs, 1),
            'hrrr_bias_pct': round((float(h_val) - g_obs) / g_obs * 100, 1) if g_obs > 0 else np.nan,
            'aorc_bias_pct': round((float(a_val) - g_obs) / g_obs * 100, 1) if g_obs > 0 else np.nan,
            'qc_flag': gauge.get('qc_flag', 'GOOD'),
        })
    
    gauge_results = pd.DataFrame(results)
    
    # Compute aggregate gauge metrics
    if len(gauge_results) > 0:
        good_gauges = gauge_results[gauge_results['qc_flag'] == 'GOOD']
        if len(good_gauges) == 0:
            good_gauges = gauge_results  # Fall back to all gauges
        
        obs = good_gauges['observed_mm'].values
        h_vals = good_gauges['hrrr_mm'].values
        a_vals = good_gauges['aorc_mm'].values
        
        agg_metrics = {
            'n_gauges': len(good_gauges),
            'hrrr_mean_bias_mm': round(np.mean(h_vals - obs), 2),
            'hrrr_mae_mm': round(np.mean(np.abs(h_vals - obs)), 2),
            'hrrr_rmse_mm': round(np.sqrt(np.mean((h_vals - obs)**2)), 2),
            'aorc_mean_bias_mm': round(np.mean(a_vals - obs), 2),
            'aorc_mae_mm': round(np.mean(np.abs(a_vals - obs)), 2),
            'aorc_rmse_mm': round(np.sqrt(np.mean((a_vals - obs)**2)), 2),
        }
        
        # Correlation (need > 2 gauges)
        if len(good_gauges) > 2:
            agg_metrics['hrrr_gauge_corr'] = round(np.corrcoef(h_vals, obs)[0, 1], 4)
            agg_metrics['aorc_gauge_corr'] = round(np.corrcoef(a_vals, obs)[0, 1], 4)
        else:
            agg_metrics['hrrr_gauge_corr'] = np.nan
            agg_metrics['aorc_gauge_corr'] = np.nan
    else:
        agg_metrics = {}
    
    return gauge_results, agg_metrics

# =============================================================================
# 5. SPATIAL METRICS (Per-grid-cell for map overlays)
# =============================================================================

def compute_spatial_difference(hrrr, aorc):
    """
    Compute the grid-cell-by-grid-cell difference (HRRR - AORC).
    
    This is the simplest spatial metric — it shows where HRRR over-predicts
    (positive values, red on map) and under-predicts (negative, blue).
    
    Returns
    -------
    np.ndarray
        2D difference field (same shape as inputs)
    """
    return np.asarray(hrrr) - np.asarray(aorc)

def compute_spatial_bias_ratio(hrrr, aorc, min_aorc=0.5):
    """
    Compute the grid-cell-by-grid-cell bias ratio (HRRR / AORC).
    
    Values > 1 mean HRRR is wetter than AORC at that cell.
    Values < 1 mean HRRR is drier.
    Cells where AORC < min_aorc are masked to avoid division by tiny numbers.
    
    Parameters
    ----------
    min_aorc : float
        Minimum AORC value (mm) to include in the ratio calculation
    
    Returns
    -------
    np.ndarray
        2D bias ratio field (NaN where AORC is below threshold)
    """
    h = np.asarray(hrrr, dtype=float)
    a = np.asarray(aorc, dtype=float)
    
    ratio = np.full_like(h, np.nan)
    valid = a >= min_aorc
    ratio[valid] = h[valid] / a[valid]
    
    return ratio

def compute_spatial_absolute_error(hrrr, aorc):
    """Per-cell absolute error |HRRR - AORC|."""
    return np.abs(np.asarray(hrrr) - np.asarray(aorc))

# =============================================================================
# 6. FRACTIONS SKILL SCORE (FSS)
# =============================================================================

def compute_fss(hrrr, aorc, thresholds=None, neighborhood_sizes=None, grid_spacing_km=3.0):
    """
    Compute Fractions Skill Score (FSS) across multiple spatial scales.
    
    FSS (Roberts & Lean, 2008) answers: "At what spatial scale does HRRR
    become skillful at predicting precipitation exceeding a given threshold?"
    
    The idea is that a forecast might place rain in the wrong exact grid cell
    but get the neighborhood right. FSS quantifies this by:
      1. Converting both fields to binary (>= threshold)
      2. Averaging the binary fields over neighborhoods of increasing size
      3. Comparing the resulting fraction fields
    
    FSS = 1 means perfect overlap of fractions at that scale.
    FSS = 0 means no skill (worse than random).
    FSS >= 0.5 is considered "useful" skill (the conventional target).
    
    Parameters
    ----------
    hrrr : np.ndarray
        2D HRRR accumulated precipitation (y, x)
    aorc : np.ndarray
        2D AORC accumulated precipitation on HRRR grid (y, x)
    thresholds : list of float, optional
        Precipitation thresholds in mm. Defaults to PRECIP_THRESHOLDS from config.
    neighborhood_sizes : list of int, optional
        Neighborhood widths in grid cells (must be odd). If None, auto-generates
        from 1 (single cell, ~3km) up to the domain size.
    grid_spacing_km : float
        Approximate grid spacing in km (HRRR ≈ 3km)
    
    Returns
    -------
    dict with:
        'thresholds_mm': list of threshold values
        'neighborhood_cells': list of neighborhood widths (grid cells)
        'neighborhood_km': list of neighborhood widths (km)
        'fss': dict mapping threshold → list of FSS values (one per neighborhood)
        'useful_scale_km': dict mapping threshold → km where FSS first >= 0.5
    """
    from scipy.ndimage import uniform_filter
    from config import PRECIP_THRESHOLDS
    
    if thresholds is None:
        thresholds = PRECIP_THRESHOLDS
    
    h = np.asarray(hrrr, dtype=float)
    a = np.asarray(aorc, dtype=float)
    
    # Replace NaN with 0 for the fraction computation
    h = np.nan_to_num(h, nan=0.0)
    a = np.nan_to_num(a, nan=0.0)
    
    ny, nx = h.shape
    max_dim = min(ny, nx)
    
    # Auto-generate neighborhood sizes if not provided
    # Go from 1 cell (~3km) up to the full domain, using roughly logarithmic spacing
    if neighborhood_sizes is None:
        # Build a list of odd sizes from 1 to max_dim
        raw = [1, 3, 5, 7, 9, 13, 17, 21, 27, 33, 41, 49]
        neighborhood_sizes = [n for n in raw if n <= max_dim]
        # Always include the full domain (make it odd)
        full = max_dim if max_dim % 2 == 1 else max_dim - 1
        if full not in neighborhood_sizes:
            neighborhood_sizes.append(full)
        neighborhood_sizes = sorted(set(neighborhood_sizes))
    
    neighborhood_km = [round(n * grid_spacing_km, 1) for n in neighborhood_sizes]
    
    result = {
        'thresholds_mm': [float(t) for t in thresholds],
        'neighborhood_cells': neighborhood_sizes,
        'neighborhood_km': neighborhood_km,
        'fss': {},
        'useful_scale_km': {},
    }
    
    for thresh in thresholds:
        # Binary fields: 1 where precip >= threshold, 0 otherwise
        h_binary = (h >= thresh).astype(float)
        a_binary = (a >= thresh).astype(float)
        
        # Check if this threshold is meaningful:
        # Need at least 1% of cells exceeding threshold in at least one field
        n_cells = h.size
        h_wet = h_binary.sum()
        a_wet = a_binary.sum()
        min_cells = max(5, n_cells * 0.01)  # At least 5 cells or 1%
        
        # If neither field has meaningful coverage, FSS is undefined
        if h_wet < min_cells and a_wet < min_cells:
            fss_values = [float('nan')] * len(neighborhood_sizes)
            result['fss'][float(thresh)] = fss_values
            result['useful_scale_km'][float(thresh)] = None
            print(f"  {thresh}mm threshold: N/A (too few cells exceed threshold in both fields)")
            continue
        
        fss_values = []
        
        for n in neighborhood_sizes:
            # Compute fraction fields using a uniform (box) filter
            # uniform_filter computes the mean over an n×n window
            h_frac = uniform_filter(h_binary, size=n, mode='constant', cval=0.0)
            a_frac = uniform_filter(a_binary, size=n, mode='constant', cval=0.0)
            
            # MSE of the fraction fields
            mse_frac = np.mean((h_frac - a_frac) ** 2)
            
            # Reference MSE (worst possible = no overlap)
            # This is mean(h_frac²) + mean(a_frac²)
            mse_ref = np.mean(h_frac ** 2) + np.mean(a_frac ** 2)
            
            if mse_ref == 0:
                # Both fields are entirely zero — FSS is undefined
                fss_values.append(float('nan'))
            else:
                fss = 1.0 - (mse_frac / mse_ref)
                fss_values.append(round(float(fss), 4))
        
        result['fss'][float(thresh)] = fss_values
        
        # Find the "useful scale" — smallest neighborhood where FSS >= 0.5
        useful_km = None
        for fss_val, km in zip(fss_values, neighborhood_km):
            if not np.isnan(fss_val) and fss_val >= 0.5:
                useful_km = km
                break
        result['useful_scale_km'][float(thresh)] = useful_km
    
    return result

# =============================================================================
# 7. MASTER METRICS FUNCTION
# =============================================================================

def compute_all_metrics(hrrr_accum, aorc_accum, gauge_accum_df=None,
                        hrrr_lat=None, hrrr_lon=None, thresholds=None):
    """
    Compute ALL verification metrics for a single event/accumulation window.
    This is the main entry point for the metrics engine.
    
    Parameters
    ----------
    hrrr_accum : np.ndarray or xr.DataArray
        2D HRRR accumulated precipitation
    aorc_accum : np.ndarray or xr.DataArray
        2D AORC accumulated precipitation (regridded to HRRR grid)
    gauge_accum_df : pd.DataFrame, optional
        Accumulated gauge observations
    hrrr_lat, hrrr_lon : np.ndarray, optional
        HRRR grid coordinates (needed for gauge matching)
    thresholds : list of float, optional
        Precipitation thresholds for categorical metrics (mm)
    
    Returns
    -------
    dict with keys:
        'continuous': dict — domain-wide continuous metrics
        'categorical': pd.DataFrame — categorical metrics at multiple thresholds
        'distributions': dict — PDFs and CDFs for HRRR, AORC, and gauges
        'gauge_detail': pd.DataFrame — per-station comparison
        'gauge_summary': dict — aggregate gauge metrics
        'spatial': dict — 2D spatial metric fields
    """
    print("=" * 60)
    print("COMPUTING VERIFICATION METRICS")
    print("=" * 60)
    
    # 1. Continuous metrics
    print("\n[1/5] Continuous metrics...")
    continuous = compute_continuous_metrics(hrrr_accum, aorc_accum)
    print(f"  Bias: {continuous['mult_bias']}, MAE: {continuous['mae']} mm, "
          f"RMSE: {continuous['rmse']} mm, r: {continuous['correlation']}")
    
    # 2. Categorical metrics
    print("\n[2/5] Categorical metrics...")
    categorical = compute_categorical_multi_threshold(hrrr_accum, aorc_accum, thresholds)
    for _, row in categorical.iterrows():
        print(f"  Threshold {row['threshold_mm']:.1f}mm: "
              f"POD={row['pod']:.3f}, FAR={row['far']:.3f}, CSI={row['csi']:.3f}")
    
    # 3. Distributional analysis (PDFs and CDFs)
    print("\n[3/5] PDF/CDF analysis...")
    distributions = compute_pdf_cdf_comparison(
        hrrr_accum, aorc_accum, gauge_accum_df
    )
    print(f"  KS statistic: {distributions['ks_statistic']} "
          f"(p={distributions['ks_pvalue']:.4e})")
    print(f"  HRRR P90: {distributions['hrrr_cdf']['p90']} mm, "
          f"AORC P90: {distributions['aorc_cdf']['p90']} mm")
    
    # 4. Gauge comparison
    gauge_detail = pd.DataFrame()
    gauge_summary = {}
    if gauge_accum_df is not None and hrrr_lat is not None:
        print("\n[4/5] Gauge comparison...")
        gauge_detail, gauge_summary = compute_gauge_metrics(
            hrrr_accum, aorc_accum, gauge_accum_df, hrrr_lat, hrrr_lon
        )
        if gauge_summary:
            print(f"  {gauge_summary['n_gauges']} gauges compared")
            print(f"  HRRR vs gauges: bias={gauge_summary['hrrr_mean_bias_mm']} mm, "
                  f"MAE={gauge_summary['hrrr_mae_mm']} mm")
            print(f"  AORC vs gauges: bias={gauge_summary['aorc_mean_bias_mm']} mm, "
                  f"MAE={gauge_summary['aorc_mae_mm']} mm")
    else:
        print("\n[4/5] Gauge comparison... SKIPPED (no gauge data or coordinates)")
    
    # 5. Spatial fields
    print("\n[5/6] Spatial metric fields...")
    spatial = {
        'difference': compute_spatial_difference(hrrr_accum, aorc_accum),
        'bias_ratio': compute_spatial_bias_ratio(hrrr_accum, aorc_accum),
        'absolute_error': compute_spatial_absolute_error(hrrr_accum, aorc_accum),
    }
    
    # 6. Fractions Skill Score
    print("\n[6/6] Fractions Skill Score (FSS)...")
    fss = compute_fss(hrrr_accum, aorc_accum, thresholds=thresholds)
    for thresh, scale in fss['useful_scale_km'].items():
        scale_str = f"{scale} km" if scale is not None else "> domain"
        print(f"  {thresh}mm threshold: useful scale = {scale_str}")
    
    print("\n" + "=" * 60)
    print("ALL METRICS COMPUTED SUCCESSFULLY ✓")
    print("=" * 60)
    
    return {
        'continuous': continuous,
        'categorical': categorical,
        'distributions': distributions,
        'gauge_detail': gauge_detail,
        'gauge_summary': gauge_summary,
        'spatial': spatial,
        'fss': fss,
    }

def print_metrics_report(metrics_dict, event_label=""):
    """
    Print a formatted text report of all metrics.
    Useful for quick inspection in Spyder console.
    """
    print("\n" + "=" * 70)
    print(f"  VERIFICATION REPORT: {event_label}")
    print("=" * 70)
    
    c = metrics_dict['continuous']
    print(f"\n  CONTINUOUS METRICS ({c['n_cells']} grid cells)")
    print(f"  {'─' * 50}")
    print(f"  Mean HRRR:           {c['mean_hrrr']:8.2f} mm")
    print(f"  Mean AORC:           {c['mean_aorc']:8.2f} mm")
    print(f"  Multiplicative Bias: {c['mult_bias']:8.3f}   (perfect = 1.000)")
    print(f"  Additive Bias:       {c['add_bias']:8.2f} mm (perfect = 0.00)")
    print(f"  MAE:                 {c['mae']:8.2f} mm")
    print(f"  RMSE:                {c['rmse']:8.2f} mm")
    print(f"  Correlation:         {c['correlation']:8.4f}   (perfect = 1.0000)")
    
    print(f"\n  CATEGORICAL METRICS")
    print(f"  {'─' * 50}")
    cat = metrics_dict['categorical']
    print(f"  {'Threshold':>10s}  {'POD':>6s}  {'FAR':>6s}  {'CSI':>6s}  {'ETS':>6s}  {'FBias':>6s}")
    for _, row in cat.iterrows():
        ets_str = f"{row['ets']:6.3f}" if not np.isnan(row['ets']) else "   N/A"
        print(f"  {row['threshold_mm']:8.1f}mm  {row['pod']:6.3f}  {row['far']:6.3f}  "
              f"{row['csi']:6.3f}  {ets_str}  {row['frequency_bias']:6.3f}")
    
    dist = metrics_dict['distributions']
    print(f"\n  DISTRIBUTION COMPARISON")
    print(f"  {'─' * 50}")
    print(f"  KS Statistic:    {dist['ks_statistic']}")
    print(f"  KS p-value:      {dist['ks_pvalue']:.4e}")
    print(f"  HRRR Percentiles: P50={dist['hrrr_cdf']['median']}mm, "
          f"P90={dist['hrrr_cdf']['p90']}mm, P95={dist['hrrr_cdf']['p95']}mm")
    print(f"  AORC Percentiles: P50={dist['aorc_cdf']['median']}mm, "
          f"P90={dist['aorc_cdf']['p90']}mm, P95={dist['aorc_cdf']['p95']}mm")
    
    gs = metrics_dict.get('gauge_summary', {})
    if gs:
        print(f"\n  GAUGE COMPARISON ({gs['n_gauges']} stations)")
        print(f"  {'─' * 50}")
        print(f"  HRRR vs Gauges: Bias={gs['hrrr_mean_bias_mm']:+.2f}mm, "
              f"MAE={gs['hrrr_mae_mm']:.2f}mm, r={gs.get('hrrr_gauge_corr', 'N/A')}")
        print(f"  AORC vs Gauges: Bias={gs['aorc_mean_bias_mm']:+.2f}mm, "
              f"MAE={gs['aorc_mae_mm']:.2f}mm, r={gs.get('aorc_gauge_corr', 'N/A')}")
    
    print("\n" + "=" * 70)

# =============================================================================
if __name__ == "__main__":
    # Quick self-test with synthetic data
    print("Running metrics self-test with synthetic data...")
    np.random.seed(42)
    
    # Create synthetic HRRR and AORC fields
    y, x = np.mgrid[0:30, 0:30]
    aorc_synth = 15 * np.exp(-((y-15)**2 + (x-15)**2) / 100) + np.random.rand(30, 30) * 3
    hrrr_synth = 18 * np.exp(-((y-14)**2 + (x-16)**2) / 120) + np.random.rand(30, 30) * 4
    
    continuous = compute_continuous_metrics(hrrr_synth, aorc_synth)
    print(f"\nContinuous: {continuous}")
    
    categorical = compute_categorical_metrics(hrrr_synth, aorc_synth, 2.54)
    print(f"\nCategorical (2.54mm): {categorical}")
    
    pdf = compute_pdf(hrrr_synth)
    cdf = compute_cdf(hrrr_synth)
    print(f"\nPDF: {pdf['n_samples']} samples, {len(pdf['bin_centers'])} bins")
    print(f"CDF: median={cdf['median']}mm, P90={cdf['p90']}mm, P95={cdf['p95']}mm")
    
    print("\nSelf-test PASSED ✓")
