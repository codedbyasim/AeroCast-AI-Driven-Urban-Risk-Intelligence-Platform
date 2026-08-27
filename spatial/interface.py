"""
Public Interface Facade for AeroCast M2 Spatial Interpolation Engine
====================================================================
Provides a unified geostatistical interpolation API for downstream predictive
analytics (M3 Smog Forecasting, M4 Flash Flood Risk Engine) and GIS visualization (M6).
Keyed by canonical Zone ID (ZONE-LHR-####).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import threading

from .kriging_engine import KrigingEngine
from .covariates import CovariateManager
from ingestion.cache import LocalDataCache
from config import settings

logger = logging.getLogger("aerocast.spatial.interface")

_engine: Optional[KrigingEngine] = None
_lock = threading.RLock()
SPATIAL_CACHE_DIR = Path(settings.CACHE_DIR) / "spatial"


def _get_engine() -> KrigingEngine:
    """Singleton getter for KrigingEngine."""
    global _engine
    if _engine is None:
        cache = LocalDataCache()
        cov_mgr = CovariateManager()
        _engine = KrigingEngine(cache=cache, covariate_manager=cov_mgr)
    return _engine


def get_interpolated_grid(
    variable: str = "aqi_pm25",
    allow_cache: bool = True,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Downstream Facade API: Retrieve interpolated 241-zone grid values for a specific variable.

    :param variable: Environmental variable (e.g. 'aqi_pm25', 'temperature_c', etc.)
    :param allow_cache: Whether to read from cached spatial snapshot if available.
    :param force_recompute: If True, forces re-running Kriging interpolation.
    :return: Mapping of zone_id -> {value, variance, confidence_score, method, ...}
    """
    SPATIAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = SPATIAL_CACHE_DIR / f"{variable}_latest.json"

    if allow_cache and not force_recompute and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return cached_data.get("zones", {})
        except Exception as e:
            logger.warning("Failed to load cached spatial grid for %s: %s. Recomputing...", variable, e)

    engine = _get_engine()
    results = engine.interpolate_variable(variable=variable)

    # Cache output to .cache/spatial/{variable}_latest.json
    try:
        with _lock:
            payload = {
                "variable": variable,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "total_zones": len(results),
                "zones": results,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
    except Exception as e:
        logger.error("Failed to write spatial cache for %s: %s", variable, e)

    return results


def get_all_interpolated_grid(
    allow_cache: bool = True,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Downstream Facade API: Retrieve the complete 241-zone table for ALL variables at once.
    This is what Module M3 (ML Forecasting) and Module M4 (Flash Flood) consume.

    :return: Nested dictionary: zone_id -> variable -> {value, variance, confidence_score, ...}
    """
    SPATIAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = SPATIAL_CACHE_DIR / "all_variables_latest.json"

    if allow_cache and not force_recompute and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return cached_data.get("zones", {})
        except Exception as e:
            logger.warning("Failed to load cached complete spatial grid: %s. Recomputing...", e)

    engine = _get_engine()
    results = engine.interpolate_all_variables()

    try:
        with _lock:
            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "total_zones": len(results),
                "zones": results,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
    except Exception as e:
        logger.error("Failed to write complete spatial cache: %s", e)

    return results


def get_zone_interpolated(zone_id: str, allow_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Retrieve all interpolated variables and confidence scores for a single zone.
    """
    full_grid = get_all_interpolated_grid(allow_cache=allow_cache)
    return full_grid.get(zone_id)


def trigger_spatial_interpolation() -> Dict[str, Any]:
    """
    Synchronously trigger full spatial kriging recomputation across all variables
    and persist results into the spatial cache.
    """
    all_grid = get_all_interpolated_grid(allow_cache=False, force_recompute=True)
    return {
        "status": "success",
        "zones_interpolated": len(all_grid),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def get_spatial_health() -> Dict[str, Any]:
    """
    Check operational status and metrics of the Spatial Interpolation Engine.
    """
    cov_mgr = CovariateManager()
    is_synthetic = cov_mgr.is_synthetic_ndvi()

    # Inspect PM2.5 grid
    pm25_grid = get_interpolated_grid("aqi_pm25", allow_cache=True)
    direct_sensors = sum(1 for z in pm25_grid.values() if z.get("is_direct_sensor"))
    avg_confidence = round(
        float(sum(z.get("confidence_score", 0.0) for z in pm25_grid.values()) / max(1, len(pm25_grid))), 2
    )

    return {
        "status": "healthy" if len(pm25_grid) > 0 else "uninitialized",
        "total_zones_covered": len(pm25_grid),
        "direct_sensor_zones": direct_sensors,
        "interpolated_zones": len(pm25_grid) - direct_sensors,
        "mean_pm25_confidence": avg_confidence,
        "synthetic_covariates_active": is_synthetic,
        "road_density_cached": cov_mgr.road_density_path.exists(),
    }
