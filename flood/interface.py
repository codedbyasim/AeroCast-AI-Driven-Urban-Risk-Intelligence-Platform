"""
AeroCast Module M4: Flash Flood Risk Interface.
Provides cached and on-demand flash flood risk scoring endpoints for downstream REST API & GIS Dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from .engine import FlashFloodScorer
from config import settings

logger = logging.getLogger("aerocast.flood.interface")

_FLOOD_SCORER: Optional[FlashFloodScorer] = None
_CACHE_DIR = Path(settings.CACHE_DIR) / "flood"
_CACHE_TTL_MINUTES = 30


def _get_flood_scorer() -> FlashFloodScorer:
    global _FLOOD_SCORER
    if _FLOOD_SCORER is None:
        _FLOOD_SCORER = FlashFloodScorer()
    return _FLOOD_SCORER


def _read_cache(cache_file: Path) -> Optional[Dict[str, Any]]:
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = datetime.fromisoformat(data.get("timestamp_utc"))
            if datetime.now(timezone.utc) - cached_at < timedelta(minutes=_CACHE_TTL_MINUTES):
                return data.get("payload")
        except Exception as e:
            logger.warning("Failed to read flood cache %s: %s", cache_file, e)
    return None


def _write_cache(cache_file: Path, payload: Dict[str, Any]):
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write flood cache %s: %s", cache_file, e)


def get_zone_flood_risk(
    zone_id: str,
    horizon_hours: int = 24,
    current_weather: Optional[Dict[str, Any]] = None,
    allow_cache: bool = True,
) -> Dict[str, Any]:
    """
    Get 24-hour advance flash flood and urban waterlogging risk score for a specific zone.
    """
    if horizon_hours != 24:
        raise ValueError(f"Module M4 currently only supports horizon_hours=24 (got {horizon_hours}).")

    cache_file = _CACHE_DIR / f"flood_{zone_id}_{horizon_hours}h.json"
    if allow_cache and current_weather is None:
        cached = _read_cache(cache_file)
        if cached:
            return cached

    scorer = _get_flood_scorer()
    result = scorer.calculate_zone_flood_risk(
        zone_id, horizon_hours=horizon_hours, current_weather=current_weather
    )

    if allow_cache and current_weather is None:
        _write_cache(cache_file, result)

    return result


def get_all_zones_flood_risk(
    horizon_hours: int = 24,
    weather_by_zone: Optional[Dict[str, Dict[str, Any]]] = None,
    allow_cache: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Get 24-hour advance flash flood risk for all 241 zones across Lahore District.
    """
    if horizon_hours != 24:
        raise ValueError(f"Module M4 currently only supports horizon_hours=24 (got {horizon_hours}).")

    cache_file = _CACHE_DIR / f"all_zones_flood_{horizon_hours}h.json"
    if allow_cache and weather_by_zone is None:
        cached = _read_cache(cache_file)
        if cached:
            return cached

    scorer = _get_flood_scorer()
    results = scorer.calculate_all_zones_flood_risk(
        horizon_hours=horizon_hours, weather_by_zone=weather_by_zone
    )

    if allow_cache and weather_by_zone is None:
        _write_cache(cache_file, results)

    return results


def get_flood_health() -> Dict[str, Any]:
    """Return health diagnostic status of Module M4 Flash Flood Risk Engine."""
    scorer = _get_flood_scorer()
    zone_count = len(scorer._zone_covariates)
    return {
        "status": "healthy",
        "module": "M4_Flash_Flood_Engine",
        "total_zones_covered": zone_count,
        "is_241_zones_canonical": (zone_count == 241),
        "weights": {
            "precipitation": scorer.WEIGHT_PRECIPITATION,
            "imperviousness": scorer.WEIGHT_IMPERVIOUSNESS,
            "slope_flatness": scorer.WEIGHT_SLOPE_FLATNESS,
            "elevation_depression": scorer.WEIGHT_ELEVATION_SINK,
            "antecedent_wetness": scorer.WEIGHT_ANTECEDENT_WETNESS,
        },
        "elevation_bounds_m": {
            "min": round(scorer.min_elevation, 1),
            "max": round(scorer.max_elevation, 1),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
