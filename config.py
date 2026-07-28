"""
=============================================================================
HRRR vs AORC Precipitation Verification Dashboard — Configuration
=============================================================================
Central configuration for domain bounds, data paths, and analysis parameters.
Modify DOMAIN_CENTER and DOMAIN_SIZE_KM to change the analysis region.

Future: NY-uHMT integration will add additional gauge networks and 
high-resolution urban observations to this configuration.
=============================================================================
"""

import os
import numpy as np
from datetime import datetime

# =============================================================================
# DOMAIN CONFIGURATION
# Change these to move/resize the analysis domain
# =============================================================================
DOMAIN_CENTER_LAT = 40.7128    # NYC latitude
DOMAIN_CENTER_LON = -74.0060   # NYC longitude
DOMAIN_SIZE_KM = 100           # Width and height in km (adjustable)

# Convert km to approximate degrees (at NYC latitude)
# 1 degree latitude ≈ 111 km everywhere
# 1 degree longitude ≈ 111 * cos(lat) km ≈ 84.4 km at 40.7°N
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(DOMAIN_CENTER_LAT))

HALF_HEIGHT_DEG = (DOMAIN_SIZE_KM / 2.0) / KM_PER_DEG_LAT
HALF_WIDTH_DEG = (DOMAIN_SIZE_KM / 2.0) / KM_PER_DEG_LON

# Bounding box derived from center + size
DOMAIN_BBOX = {
    'south': DOMAIN_CENTER_LAT - HALF_HEIGHT_DEG,
    'north': DOMAIN_CENTER_LAT + HALF_HEIGHT_DEG,
    'west': DOMAIN_CENTER_LON - HALF_WIDTH_DEG,
    'east': DOMAIN_CENTER_LON + HALF_WIDTH_DEG,
}

# Add a buffer for regridding edge effects (10 km ≈ 0.09°)
BUFFER_DEG = 0.1
DOMAIN_BBOX_BUFFERED = {
    'south': DOMAIN_BBOX['south'] - BUFFER_DEG,
    'north': DOMAIN_BBOX['north'] + BUFFER_DEG,
    'west': DOMAIN_BBOX['west'] - BUFFER_DEG,
    'east': DOMAIN_BBOX['east'] + BUFFER_DEG,
}

# --- Tight NYC Metro Domain (from SOM verification work) ---
# This is the domain used in the frequency_bias_csi_analysis SOM project.
# It shows up as a BLACK BOX OUTLINE on all maps to indicate which part of
# the larger domain is considered "NYC metro" for research purposes.
NYC_TIGHT_BBOX = {
    'south': 40.48,
    'north': 40.92,
    'west': -74.26,
    'east': -73.70,
}

# --- Context Overlay Extent ---
# Controls how large the precipitation overlay on the map is.
# Default 600km gives DC to Boston, western PA to offshore — 
# enough to see the broader storm context while keeping downloads fast.
CONTEXT_EXTENT_KM = 600

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# --- AORC via FTP (PSL 4km product) ---
# AORC 4km precipitation data from NOAA Physical Sciences Laboratory.
# This is the SAME source used in the SOM project (6.5_AORC_4km_Downloader).
#
# Server: ftp2.psl.noaa.gov
# Path:   /Projects/AORC_CONUS_4km/{YYYY}/prate.aorc.{YYYYMMDD}.nc
#
# Each daily file contains 24 hourly timesteps of precipitation RATE.
# Variable: 'prate', Units: mm/hr (verified from file metadata).
# Since each timestep = 1 hour, the value IS the hourly accumulation in mm.
# No unit conversion is needed (load_aorc() auto-detects this from attrs).
#
# ⚠ This is NOT the OWP 1km AORC product from hydrology.nws.noaa.gov.
AORC_FTP_HOST = "ftp2.psl.noaa.gov"
AORC_FTP_BASE_DIR = "/Projects/AORC_CONUS_4km"
AORC_FILE_PATTERN = "prate.aorc.{date}.nc"  # {date} = YYYYMMDD

# --- HRRR via AWS S3 ---
# HRRR GRIB2 files on the Big Data Program S3 bucket (no auth needed)
# Path pattern: hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf{FH}.grib2
# Also available via HTTPS (no boto3 needed):
#   https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/...
HRRR_S3_BUCKET = "noaa-hrrr-bdp-pds"
HRRR_S3_REGION = "us-east-1"
HRRR_S3_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

# --- HRRR FORECAST STRATEGY ---
#
# This controls HOW we sample HRRR forecasts, which directly affects the
# verification results. Different strategies represent different operational
# scenarios. Getting this wrong silently corrupts the entire analysis.
#
# STRATEGY OPTIONS:
#
#   'rolling_1h' (default, most common in research):
#       For each valid hour, use that hour's own initialization at fxx=1.
#       Example for 00Z-24Z accumulation:
#         Hour 01Z valid → init=00Z, fxx=1  (1h lead time)
#         Hour 02Z valid → init=01Z, fxx=1  (1h lead time)
#         ...
#         Hour 00Z+1 valid → init=23Z, fxx=1 (1h lead time)
#       Pros: All forecasts have identical 1h lead time. Most skillful.
#       Cons: Not operationally realistic — you'd need to wait for each
#             hourly run before issuing the "forecast."
#       Use when: You want to evaluate HRRR's best-case precipitation skill,
#                 isolating model physics errors from lead-time degradation.
#
#   'single_init' (operationally realistic):
#       Use ONE initialization and take consecutive forecast hours.
#       Example for 00Z init, 24h accumulation:
#         Hour 01Z valid → init=00Z, fxx=1   (1h lead)
#         Hour 02Z valid → init=00Z, fxx=2   (2h lead)
#         ...
#         Hour 00Z+1 valid → init=00Z, fxx=24 (24h lead)
#       Pros: Represents what a forecaster actually had available at init time.
#       Cons: Skill degrades with lead time — later hours are less accurate.
#       Use when: You want to evaluate operational forecast utility.
#       INIT_HOUR controls which run to use (default=0 for 00Z).
#
#   'day_ahead_12z' (day-ahead operational):
#       Use the PREVIOUS DAY's 12Z run for tomorrow's weather.
#       Example for next-day 00Z-24Z accumulation:
#         Hour 01Z valid → init=prev_day 12Z, fxx=13  (13h lead)
#         Hour 02Z valid → init=prev_day 12Z, fxx=14  (14h lead)
#         ...
#         Hour 00Z+1 valid → init=prev_day 12Z, fxx=36 (36h lead)
#       Pros: True day-ahead forecast — what forecasters brief in the morning.
#       Cons: Longest lead times, weakest skill. Only available from 
#             extended HRRR runs (00Z, 06Z, 12Z, 18Z go out to 48h).
#       Use when: Evaluating HRRR's value for next-day flood planning.
#
HRRR_FORECAST_STRATEGY = 'day_ahead_12z'  # 'rolling_1h', 'single_init', 'day_ahead_12z'
HRRR_INIT_HOUR = 0  # Only used for 'single_init' strategy (0-23)
HRRR_FXX = 1        # Forecast hour for 'rolling_1h' strategy

# --- ASOS/AWOS via Iowa Environmental Mesonet ---
# IEM provides the best unified API for ASOS/AWOS/COOP observations
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# Network codes for states in our domain
IEM_NETWORKS = ["NY_ASOS", "NJ_ASOS", "CT_ASOS", "PA_ASOS"]

# --- NY-uHMT (New York Urban Hydrometeorological Testbed) ---
# Dense urban rain gauge network across all five NYC boroughs.
# 15-minute tipping bucket data from NOAA CREST, served as CSV files.
# Timestamps are in Eastern local time (EDT/EST) — loader converts to UTC.
NY_UHMT_ENABLED = True
NY_UHMT_BASE_URL = "https://datadb.noaacrest.org/public/uhmt/Processed_Data"
NY_UHMT_LOCAL_DIR = os.path.join(os.path.dirname(__file__), 'nyuhmt_data')  # Local CSV cache folder

# Station metadata: site_num → (name, code, lat, lon, csv_filename)
# Coordinates verified from NOAA CREST project documentation.
# Sites 4,17,19,20 have no CSV on the server (decommissioned/not deployed).
# Site 22 on server = Site 21 in metadata (New York Harbor School, renumbered).
NY_UHMT_STATIONS = {
    1:  ("Queens Botanical Garden",          "QBG", 40.75062,  -73.82873, "Site1_Queens_Botanical_Garden_Fifteen.csv"),
    2:  ("Queensborough Community College",  "QCC", 40.75685,  -73.75620, "Site2_Queensborough_Community_College_Fifteen.csv"),
    3:  ("Ronald Edmonds Learning Ctr",      "REM", 40.68843,  -73.97116, "Site3_Ronald_Edmonds_Learning_Center_Fifteen.csv"),
    5:  ("Middletown Plaza NYCHA",           "MTP", 40.84375,  -73.83107, "Site5_Middletown_Houses_Fifteen.csv"),
    6:  ("Dyckman Houses NYCHA",             "DKN", 40.86086,  -73.92219, "Site6_Dyckman_Houses_Fifteen.csv"),
    7:  ("Williamsburg Houses",              "WBG", 40.71064,  -73.94253, "Site7_Williamsburg_Houses_Fifteen.csv"),
    8:  ("Eagle Academy Harlem",             "PGD", 40.81738,  -73.94751, "Site8_Polo_Ground_Fifteen.csv"),
    9:  ("Far Rockaway NYCHA",               "FRW", 40.59684,  -73.77221, "Site9_Far_Rockaway_Fifteen.csv"),
    10: ("BayView NYCHA",                    "BAY", 40.63361,  -73.88611, "Site10_BayView_Fifteen.csv"),
    11: ("Baisley Park",                     "BPK", 40.68531,  -73.78296, "Site11_Baisley_Park_Fifteen.csv"),
    12: ("East River Houses",                "ERV", 40.78849,  -73.94018, "Site12_East_River_Fifteen.csv"),
    13: ("Astoria Houses",                   "AST", 40.77306,  -73.93289, "Site13_Astoria_Fifteen.csv"),
    14: ("Haber Houses Coney Island",        "HBR", 40.57338,  -73.99098, "Site14_Haber_Coney_Island_Fifteen.csv"),
    15: ("Walt Whitman Middle School",       "WWM", 40.64836,  -73.95343, "Site15_Walt_Whitman_MS_Fifteen.csv"),
    16: ("JHS 14 Shellbank",                 "JHS", 40.59292,  -73.93793, "Site16_JHS_High_School_Fifteen.csv"),
    18: ("Mary D Carter School",             "MDC", 40.75761,  -73.90819, "Site18_MDC_School_Fifteen.csv"),
    21: ("New York Harbor School",           "NHH", 40.69135,  -74.02008, "Site22_New_York_Harbor_School_Fifteen.csv"),
}

# --- MRMS (Multi-Radar Multi-Sensor) ---
# NOAA MRMS QPE provides radar+gauge merged precipitation at ~1km/2-min.
# We use the hourly GaugeCorr_QPE_01H_Pass2 product.
# Data is in GRIB2 format — requires cfgrib or pygrib (optional dependency).
MRMS_ENABLED = True
MRMS_ARCHIVE_URL = "https://mtarchive.geol.iastate.edu"
# Path pattern: /YYYY/MM/DD/mrms/ncep/MultiSensor_QPE_01H_Pass2/
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2"

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Accumulation windows available to the user (hours)
ACCUM_WINDOWS = [6, 12, 18, 24]

# Precipitation thresholds for categorical metrics (mm)
PRECIP_THRESHOLDS = [0.254, 2.54, 6.35, 12.7, 25.4, 50.8, 76.2, 101.6]  # WPC standard: 0.01", 0.10", 0.25", 0.50", 1.00", 2.00", 3.00", 4.00"
DEFAULT_THRESHOLD_MM = 2.54  # 0.1 inch — standard light rain threshold

# Colormap settings for precipitation
PRECIP_CMAP_LEVELS = [0, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100]
DIFF_CMAP_LEVELS = [-30, -20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30]

# =============================================================================
# FILE PATHS
# =============================================================================
import os

# Base project directory — change this to your preferred location
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
HRRR_RAW_DIR = os.path.join(RAW_DIR, "hrrr")
AORC_RAW_DIR = os.path.join(RAW_DIR, "aorc")
GAUGE_RAW_DIR = os.path.join(RAW_DIR, "gauges")
MRMS_RAW_DIR = os.path.join(RAW_DIR, "mrms")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# --- SOM Project Directory (user's existing HRRR verification work) ---
# This is where the paired_nyc_WITH_TOTALS.csv and AORC/HRRR grids live
# from the SOM-based frequency bias & CSI analysis.
# UPDATE THIS to match your local machine path:
SOM_BASE_DIR = r"C:\Users\Sebastian Makrides\OneDrive - The George Washington University\Desktop\HRRR\FINALIZED TOP TIER SCRIPTS\SOM"
SOM_PAIRED_CSV = os.path.join(SOM_BASE_DIR, "paired_nyc_WITH_TOTALS.csv")
SOM_OUTPUT_DIR = os.path.join(SOM_BASE_DIR, "frequency_bias_csi_analysis")

# --- AORC Data Directory ---
# AORC daily .nc files from the PSL 4km product (ftp2.psl.noaa.gov).
# Files are named: prate.aorc.YYYYMMDD.nc
# Each file contains 24 hourly timesteps of precipitation rate.
#
# This should point to wherever you downloaded AORC files using
# 6.5_AORC_4km_Downloader_From_FTP.py — typically the aorc_data_pull_4km dir.
# UPDATE THIS to match your local machine:
AORC_LOCAL_DIR = os.path.join(SOM_BASE_DIR, "aorc_data_pull_4km")

# --- HRRR Data Directory ---
# HRRR GRIB2 or processed files. If you already have HRRR grids from the
# SOM project, point this there. Otherwise we'll download from S3.
HRRR_LOCAL_DIR = os.path.join(SOM_BASE_DIR, "hrrr")

# Create directories if they don't exist (only for project dirs, not SOM)
for d in [DATA_DIR, RAW_DIR, HRRR_RAW_DIR, AORC_RAW_DIR, GAUGE_RAW_DIR, MRMS_RAW_DIR,
          PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# DISPLAY / PRINT CONFIG
# =============================================================================
def print_config():
    """Print current configuration for verification."""
    print("=" * 60)
    print("HRRR vs AORC Verification Dashboard — Configuration")
    print("=" * 60)
    print(f"Domain Center:  {DOMAIN_CENTER_LAT:.4f}°N, {DOMAIN_CENTER_LON:.4f}°W")
    print(f"Domain Size:    {DOMAIN_SIZE_KM} km × {DOMAIN_SIZE_KM} km")
    print(f"Bounding Box:")
    print(f"  South: {DOMAIN_BBOX['south']:.4f}°N")
    print(f"  North: {DOMAIN_BBOX['north']:.4f}°N")
    print(f"  West:  {DOMAIN_BBOX['west']:.4f}°W")
    print(f"  East:  {DOMAIN_BBOX['east']:.4f}°W")
    print(f"NYC Tight Box (SOM domain):")
    print(f"  {NYC_TIGHT_BBOX['south']:.2f}°N–{NYC_TIGHT_BBOX['north']:.2f}°N, "
          f"{NYC_TIGHT_BBOX['west']:.2f}°W–{NYC_TIGHT_BBOX['east']:.2f}°W")
    print(f"Approx grid cells (HRRR 3km): ~{int(DOMAIN_SIZE_KM/3)}×{int(DOMAIN_SIZE_KM/3)}")
    print(f"Approx grid cells (AORC 1km): ~{DOMAIN_SIZE_KM}×{DOMAIN_SIZE_KM}")
    print(f"Accum windows: {ACCUM_WINDOWS} hours")
    print(f"NY-uHMT enabled: {NY_UHMT_ENABLED}")
    print(f"Data directory: {DATA_DIR}")
    print(f"SOM base dir:   {SOM_BASE_DIR}")
    print(f"AORC local dir: {AORC_LOCAL_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
