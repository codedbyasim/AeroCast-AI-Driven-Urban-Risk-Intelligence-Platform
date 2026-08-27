"""
AeroCast Spatial Interpolation & Map GeoJSON API Routes.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timezone

from config import settings
from spatial.interface import get_interpolated_grid
from ml.interface import get_all_aqi_forecasts, get_all_heat_island_risk
from flood.interface import get_all_zones_flood_risk

router = APIRouter(prefix="/api/v1/spatial", tags=["Spatial & Map Layers"])


@router.get("/grid", summary="Get Full 241-Zone Interpolated Grid")
def get_spatial_grid(
    variable: str = Query(
        "aqi_pm25",
        description="Variable to retrieve (aqi_pm25, aqi_pm10, no2_ppb, temperature_c, relative_humidity_percent, wind_speed_kmh, rainfall_mm_forecast, surface_pressure_hpa)"
    ),
    allow_cache: bool = Query(True, description="Allow reading from latest spatial snapshot cache"),
) -> Dict[str, Any]:
    """
    Returns the continuous 241-zone Universal/Ordinary Kriging interpolation matrix
    for a specific environmental variable, including variance and confidence scores.
    """
    try:
        grid = get_interpolated_grid(variable=variable, allow_cache=allow_cache)
        return {
            "variable": variable,
            "total_zones": len(grid),
            "canonical_count": 241,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "zones": grid,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate spatial grid: {e}")


@router.get("/geojson", summary="Get Enriched 241-Zone GeoJSON Map Layer")
def get_spatial_geojson(
    allow_cache: bool = Query(True, description="Allow reading from cache"),
) -> Dict[str, Any]:
    """
    Returns the complete 241-zone polygon GeoJSON FeatureCollection dynamically enriched with:
    - Live Kriging PM2.5 and confidence score (M2)
    - 24-hour advance PM2.5 forecast & hazard tier (M3)
    - Urban Heat Island (UHI) anomaly score (M3)
    - Deterministic Flash Flood runoff risk score (M4)
    
    Directly pluggable into Leaflet, Mapbox GL JS, Deck.gl, or QGIS.
    """
    zone_geojson_path = Path(settings.OSM_HDX_ZONE_GRID_PATH)
    if not zone_geojson_path.exists():
        raise HTTPException(status_code=500, detail="Canonical zone grid GeoJSON file not found.")

    with open(zone_geojson_path, "r", encoding="utf-8") as f:
        raw_geojson = json.load(f)

    raw_features = raw_geojson.get("features", [])

    # Pre-fetch grids
    try:
        pm25_grid = get_interpolated_grid("aqi_pm25", allow_cache=allow_cache)
    except Exception:
        pm25_grid = {}

    try:
        forecast_grid = get_all_aqi_forecasts(horizon_hours=24, allow_cache=allow_cache)
    except Exception:
        forecast_grid = {}

    try:
        uhi_grid = get_all_heat_island_risk(allow_cache=allow_cache)
    except Exception:
        uhi_grid = {}

    try:
        flood_grid = get_all_zones_flood_risk(horizon_hours=24, allow_cache=allow_cache)
    except Exception:
        flood_grid = {}

    enriched_features = []
    for feat in raw_features:
        props = dict(feat.get("properties", {}))
        zid = props.get("zone_id")

        m2_info = pm25_grid.get(zid, {})
        m3_fc = forecast_grid.get(zid, {})
        m3_uhi = uhi_grid.get(zid, {})
        m4_fl = flood_grid.get(zid, {})

        props["pm25_current"] = m2_info.get("value")
        props["pm25_confidence"] = m2_info.get("confidence_score")
        props["is_direct_sensor"] = m2_info.get("is_direct_sensor", False)

        props["forecast_pm25_24h"] = m3_fc.get("forecasted_pm25")
        props["hazard_category_24h"] = m3_fc.get("hazard_category")
        props["uncertainty_interval_80"] = m3_fc.get("uncertainty_interval_80")

        props["heat_island_score"] = m3_uhi.get("heat_island_score") or m3_uhi.get("uhi_risk_score") or m3_uhi.get("heat_island_risk_score")
        props["heat_risk_category"] = m3_uhi.get("risk_category")

        props["flood_risk_score"] = m4_fl.get("flood_risk_score")
        props["flood_risk_category"] = m4_fl.get("risk_category")
        props["flood_alert_level"] = m4_fl.get("alert_level")

        enriched_features.append({
            "type": "Feature",
            "geometry": feat.get("geometry"),
            "properties": props,
        })

    return {
        "type": "FeatureCollection",
        "total_features": len(enriched_features),
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": enriched_features,
    }
