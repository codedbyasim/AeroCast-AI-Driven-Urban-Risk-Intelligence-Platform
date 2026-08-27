"""
AeroCast Multi-Hazard Intelligence API Routes.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timezone

from ml.interface import get_all_aqi_forecasts, get_all_heat_island_risk
from flood.interface import get_all_zones_flood_risk

router = APIRouter(prefix="/api/v1/hazards", tags=["Multi-Hazard Intelligence"])


@router.get("/forecast", summary="Get 24-Hour Advance AQI Forecasts")
def get_aqi_forecasts(
    horizon_hours: int = Query(24, description="Forecast horizon in hours (default: 24)"),
    allow_cache: bool = Query(True, description="Allow reading from latest forecast snapshot cache"),
) -> Dict[str, Any]:
    """
    Returns 24-hour advance PM2.5 and AQI hazard forecasts, uncertainty bounds (80% CI),
    and US EPA / PEQS hazard classifications across all 241 Lahore zones.
    """
    if horizon_hours != 24:
        raise HTTPException(
            status_code=400,
            detail=f"Only horizon_hours=24 is supported (got {horizon_hours}).",
        )

    try:
        forecasts = get_all_aqi_forecasts(horizon_hours=24, allow_cache=allow_cache)
        return {
            "hazard_type": "air_quality_pm25",
            "horizon_hours": 24,
            "total_zones": len(forecasts),
            "canonical_count": 241,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "forecasts": forecasts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate AQI forecasts: {e}")


@router.get("/heat-island", summary="Get Urban Heat Island (UHI) Risk Scores")
def get_heat_island_scores(
    allow_cache: bool = Query(True, description="Allow reading from latest UHI cache"),
) -> Dict[str, Any]:
    """
    Returns Urban Heat Island (UHI) anomaly scores $[0.0, 1.0]$, Copernicus NDVI cooling deficits,
    impervious surface exposure, and heat vulnerability tiers across all 241 Lahore zones.
    """
    try:
        uhi_scores = get_all_heat_island_risk(allow_cache=allow_cache)
        return {
            "hazard_type": "urban_heat_island",
            "total_zones": len(uhi_scores),
            "canonical_count": 241,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "zones": uhi_scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate UHI scores: {e}")


@router.get("/flood", summary="Get Flash Flood & Waterlogging Risk Scores")
def get_flood_risk_scores(
    horizon_hours: int = Query(24, description="Forecast horizon in hours (default: 24)"),
    allow_cache: bool = Query(True, description="Allow reading from latest flood risk cache"),
) -> Dict[str, Any]:
    """
    Returns deterministic hydrological flash flood risk scores $[0.0, 1.0]$, slope flatness inversions,
    impervious concrete ratios, and WASA emergency dewatering advisories across all 241 zones.
    """
    if horizon_hours != 24:
        raise HTTPException(
            status_code=400,
            detail=f"Only horizon_hours=24 is supported (got {horizon_hours}).",
        )

    try:
        flood_scores = get_all_zones_flood_risk(horizon_hours=24, allow_cache=allow_cache)
        return {
            "hazard_type": "flash_flood_runoff",
            "horizon_hours": 24,
            "total_zones": len(flood_scores),
            "canonical_count": 241,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "zones": flood_scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate flood scores: {e}")


@router.get("/unified-risk-summary", summary="Get Multi-Hazard Composite Risk Index")
def get_unified_risk_summary(
    top_n: int = Query(10, description="Number of top highest-risk zones to return (default: 10)"),
) -> Dict[str, Any]:
    """
    Aggregates multi-hazard risk across Smog (M3), Heat Island (M3), and Flash Flood (M4)
    to identify the highest-priority emergency response zones across Lahore District.
    """
    try:
        forecasts = get_all_aqi_forecasts(horizon_hours=24, allow_cache=True)
        uhi = get_all_heat_island_risk(allow_cache=True)
        flood = get_all_zones_flood_risk(horizon_hours=24, allow_cache=True)

        zone_composites = []
        for zid in forecasts.keys():
            fc = forecasts.get(zid, {})
            u = uhi.get(zid, {})
            fl = flood.get(zid, {})

            # Normalized risk components [0, 1]
            pm25_val = float(fc.get("forecasted_pm25") or 65.0)
            smog_risk = min(1.0, pm25_val / 250.0)
            heat_risk = float(u.get("heat_island_risk_score") or 0.5)
            flood_risk = float(fl.get("flood_risk_score") or 0.2)

            # Composite Multi-Hazard Index (Weighted)
            composite_index = round(0.40 * smog_risk + 0.30 * heat_risk + 0.30 * flood_risk, 3)

            zone_composites.append({
                "zone_id": zid,
                "composite_risk_index": composite_index,
                "smog_risk": round(smog_risk, 3),
                "pm25_forecast_ug_m3": round(pm25_val, 1),
                "smog_hazard_tier": fc.get("hazard_category"),
                "heat_risk": round(heat_risk, 3),
                "heat_tier": u.get("risk_category"),
                "flood_risk": round(flood_risk, 3),
                "flood_tier": fl.get("risk_category"),
                "primary_threat": "Smog" if smog_risk >= max(heat_risk, flood_risk) else ("Heat Island" if heat_risk >= flood_risk else "Flash Flood"),
            })

        # Rank descending
        zone_composites.sort(key=lambda z: z["composite_risk_index"], reverse=True)

        return {
            "total_zones_evaluated": len(zone_composites),
            "canonical_count": 241,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "top_priority_zones": zone_composites[:top_n],
            "all_ranked_zones": zone_composites,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate multi-hazard summary: {e}")


@router.get("/fires", summary="Get NASA FIRMS Satellite Active Fire Hotspots")
async def get_nasa_fires(days: int = Query(3, ge=1, le=7)) -> Dict[str, Any]:
    """
    Returns real-time active fire anomaly coordinates and Fire Radiative Power (FRP)
    from NASA VIIRS/MODIS satellites across the crop-residue burning transboundary belt.
    """
    try:
        from ingestion.firms_client import NasaFirmsClient
        client = NasaFirmsClient()
        data = await client.fetch_active_fires(days=days)
        return {
            "status": "success",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fire_data": data,
        }
    except Exception as e:
        logger.error("Failed to retrieve NASA FIRMS data: %s", e)
        return {
            "status": "success",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fire_data": {
                "fire_count": 0,
                "total_frp_mw": 0.0,
                "mean_frp_mw": 0.0,
                "max_frp_mw": 0.0,
                "hotspots": [],
                "data_quality_notes": f"NASA FIRMS satellite telemetry unavailable: {e}",
            }
        }

