HRRR Precipitation Verification Dashboard — v2 (April 2026)
============================================================

DEPLOYMENT:
  1. Copy all .py files to:  web_app\web_app\
  2. Copy static\index.html to:  web_app\web_app\static\index.html
  3. Clear cache:  rmdir /s /q cache
  4. Launch:  python app.py
  5. Open:  http://localhost:8050

BUG FIXES IN THIS BUILD:
  - MRMS hour offset: files now fetched by accumulation-end time (01Z-24Z not 00Z-23Z)
  - MRMS 75% coverage guard: partial returns → None instead of misleading low totals
  - Hourly slider: station traces now use analysis-domain indices (were reading from displaced context-extent cells)
  - HRRR hourly mask: corrected to hour-ending convention (> start, <= end)
  - Eastern Time: now uses zoneinfo for proper EDT/EST (was hardcoded UTC-5)
  - Gauge nearest-cell: cos(lat) correction applied (was overweighting longitude at 40°N)
  - Custom bbox: validation added (rejects too-large, reversed, or incomplete bounds)
  - Dead code removed: visualization.py 1,354 → 1,113 lines (4 unused functions deleted)

UI CHANGES:
  - Controls retitled: "Accumulation Window", "Forecast Strategy", "Statistics Area"
  - ⓘ info icons with hover tooltips explaining each control
  - Strategy dropdown shows "Single Init 00Z" (was just "Single Init")

FILES:  7 Python + 1 HTML = 7,185 lines total
