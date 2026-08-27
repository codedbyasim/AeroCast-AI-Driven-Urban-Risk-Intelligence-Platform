"""
AeroCast Health & Diagnostics API Routes.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ingestion.interface import get_ingestion_health
from spatial.interface import get_spatial_health
from ml.interface import get_ml_health
from flood.interface import get_flood_health

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("", summary="Aggregated System Health Check")
def get_system_health() -> Dict[str, Any]:
    """
    Returns aggregated health status and operational state across all core AeroCast engines:
    - Ingestion & Storage (M1)
    - Spatial Kriging Interpolation (M2)
    - 24h AQI Forecasting & Heat Island (M3)
    - Flash Flood & Waterlogging (M4)
    """
    try:
        m1 = get_ingestion_health()
        m2 = get_spatial_health()
        m3 = get_ml_health()
        m4 = get_flood_health()

        all_healthy = (
            m1.get("status") == "healthy"
            and m2.get("status") == "healthy"
            and m3.get("status") == "healthy"
            and m4.get("status") == "healthy"
        )

        return {
            "status": "healthy" if all_healthy else "degraded",
            "service": "AeroCast Urban Risk Intelligence Platform",
            "version": "1.1.0",
            "canonical_zones": 241,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "modules": {
                "m1_ingestion": m1,
                "m2_spatial_kriging": m2,
                "m3_ml_forecasting": m3,
                "m4_flash_flood": m4,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System health check failed: {e}")


@router.get("/m1", summary="Module M1 Ingestion Health")
def get_m1_health() -> Dict[str, Any]:
    """Return Module M1 Ingestion & Storage engine health diagnostics."""
    return get_ingestion_health()


@router.get("/m2", summary="Module M2 Spatial Kriging Health")
def get_m2_health() -> Dict[str, Any]:
    """Return Module M2 Spatial Kriging interpolation health diagnostics."""
    return get_spatial_health()


@router.get("/m3", summary="Module M3 ML Forecasting Health")
def get_m3_health() -> Dict[str, Any]:
    """Return Module M3 24h AQI forecasting and UHI model diagnostics."""
    return get_ml_health()


@router.get("/m4", summary="Module M4 Flash Flood Health")
def get_m4_health() -> Dict[str, Any]:
    """Return Module M4 Flash Flood Risk engine health diagnostics."""
    return get_flood_health()
