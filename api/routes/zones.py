"""
AeroCast Zone Directory & Geolocation API Routes.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timezone

from spatial.covariates import CovariateManager
from ingestion.osm_hdx_client import OSMHDXClient
from ingestion.interface import get_latest_data
from spatial.interface import get_zone_interpolated
from ml.interface import get_aqi_forecast, get_heat_island_risk
from flood.interface import get_zone_flood_risk

router = APIRouter(prefix="/api/v1/zones", tags=["Zones & Geolocation"])

_osm_client: Optional[OSMHDXClient] = None
_cov_mgr: Optional[CovariateManager] = None


def _get_osm() -> OSMHDXClient:
    global _osm_client
    if _osm_client is None:
        _osm_client = OSMHDXClient()
    return _osm_client


def _get_cov() -> CovariateManager:
    global _cov_mgr
    if _cov_mgr is None:
        _cov_mgr = CovariateManager()
    return _cov_mgr


@router.get("", summary="List All 241 Canonical Zones")
def list_zones() -> Dict[str, Any]:
    """
    Returns the complete directory of all 241 canonical zones in Lahore District,
    including geographic centroids, surface area, and terrain covariates.
    """
    osm = _get_osm()
    cov_mgr = _get_cov()
    all_covs = cov_mgr.get_all_covariates()
    zones = osm.get_all_zones()

    zone_list = []
    for z in zones:
        zid = str(z.get("zone_id"))
        cov = all_covs.get(zid, {})

        zone_list.append({
            "zone_id": zid,
            "zone_name": z.get("zone_name"),
            "grid_row": z.get("grid_row"),
            "grid_col": z.get("grid_col"),
            "area_sqkm": z.get("area_sqkm"),
            "centroid_lat": z.get("centroid_lat"),
            "centroid_lon": z.get("centroid_lon"),
            "district": z.get("district", "Lahore District"),
            "terrain_context": {
                "elevation_m": cov.get("elevation_m"),
                "slope_percent": cov.get("slope_percent"),
                "ndvi_index": cov.get("ndvi_index"),
                "road_density_km_per_sqkm": cov.get("road_density_km_per_sqkm"),
                "population_density_per_sqkm": cov.get("population_density_per_sqkm"),
                "impervious_surface_ratio": cov.get("impervious_surface_ratio"),
            },
        })

    # Sort canonically
    zone_list.sort(key=lambda z: str(z["zone_id"]))

    return {
        "total_zones": len(zone_list),
        "canonical_grid_count": 241,
        "zones": zone_list,
    }


@router.get("/lookup", summary="Lookup Zone by GPS Coordinates")
def lookup_zone_by_coordinates(
    lat: float = Query(..., description="Latitude in decimal degrees (e.g. 31.5204)"),
    lon: float = Query(..., description="Longitude in decimal degrees (e.g. 74.3587)"),
) -> Dict[str, Any]:
    """
    Performs spatial point-in-polygon (or nearest centroid fallback) geolocation lookup
    mapping any GPS coordinate in Lahore to its canonical Zone ID and multi-hazard profile.
    """
    osm = _get_osm()
    zone_dict = osm.find_zone_by_coordinates(lat=lat, lon=lon)
    if not zone_dict:
        raise HTTPException(
            status_code=404,
            detail=f"Coordinates ({lat}, {lon}) are outside the Lahore District computational grid.",
        )

    zone_id = zone_dict.get("zone_id")
    return get_zone_snapshot(zone_id=zone_id)


@router.get("/{zone_id}", summary="Get Unified Multi-Hazard Snapshot for Zone")
def get_zone_snapshot(zone_id: str) -> Dict[str, Any]:
    """
    Returns a unified multi-hazard risk snapshot for a single zone, consolidating:
    1. Ingestion status and live metrics (M1)
    2. Spatial Kriging continuous PM2.5 estimate and confidence score (M2)
    3. 24-hour advance PM2.5 forecast, uncertainty bounds, and hazard tier (M3)
    4. Urban Heat Island (UHI) anomaly score (M3)
    5. Deterministic Flash Flood runoff risk score and WASA advisory (M4)
    """
    cov_mgr = _get_cov()
    all_covs = cov_mgr.get_all_covariates()
    if zone_id not in all_covs:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{zone_id}' not found in canonical 241-zone grid.",
        )

    cov = all_covs.get(zone_id, {})
    osm = _get_osm()
    all_z = {z["zone_id"]: z for z in osm.get_all_zones()}
    z_meta = all_z.get(zone_id, {})

    # 1. M1 Live Record
    live_rec = get_latest_data(zone_id)
    metrics = live_rec.get("metrics", {}) if live_rec else {}
    data_quality = live_rec.get("data_quality", {}) if live_rec else {
        "interpolated": True,
        "confidence_score": 0.55,
        "stale": False,
        "notes": "Spatial Kriging estimate",
    }

    # 2. M2 Kriging Snapshot
    m2_data = get_zone_interpolated(zone_id, allow_cache=True) or {}
    pm25_spatial = m2_data.get("aqi_pm25", {})

    # 3. M3 24h AQI Forecast
    try:
        m3_forecast = get_aqi_forecast(zone_id, horizon_hours=24)
    except Exception as e:
        m3_forecast = {"error": str(e), "forecasted_pm25": None}

    # 4. M3 UHI
    try:
        m3_uhi = get_heat_island_risk(zone_id, allow_cache=True)
    except Exception as e:
        m3_uhi = {"error": str(e), "heat_island_risk_score": None}

    # 5. M4 Flash Flood
    try:
        m4_flood = get_zone_flood_risk(zone_id, horizon_hours=24, allow_cache=True)
    except Exception as e:
        m4_flood = {"error": str(e), "flood_risk_score": None}

    return {
        "zone_id": zone_id,
        "zone_name": z_meta.get("zone_name", f"Zone {zone_id}"),
        "grid_row": z_meta.get("grid_row", 1),
        "grid_col": z_meta.get("grid_col", 1),
        "centroid": {
            "latitude": z_meta.get("centroid_lat", 31.5),
            "longitude": z_meta.get("centroid_lon", 74.3),
        },
        "area_sqkm": z_meta.get("area_sqkm", 9.0),
        "data_quality": data_quality,
        "current_conditions": {
            "pm25_current_ug_m3": metrics.get("pm25") or pm25_spatial.get("value"),
            "temperature_c": metrics.get("temperature_c"),
            "relative_humidity_percent": metrics.get("relative_humidity_percent"),
            "wind_speed_kmh": metrics.get("wind_speed_kmh"),
            "spatial_kriging_confidence": pm25_spatial.get("confidence_score"),
            "is_direct_sensor": pm25_spatial.get("is_direct_sensor", False),
        },
        "forecast_24h_aqi": m3_forecast,
        "urban_heat_island": m3_uhi,
        "flash_flood_risk": m4_flood,
        "terrain_covariates": cov,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
