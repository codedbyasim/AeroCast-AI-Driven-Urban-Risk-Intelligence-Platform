"""
Public API Facade for AeroCast Module M3 (ML & Predictive Analytics).
Provides high-level programmatic access to 24-hour advance AQI forecasts, UHI risk scores,
and health diagnostic checks. Includes disk caching to prevent redundant execution.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from config import settings
from .aqi_forecast import AQIForecastModel
from .heat_island import HeatIslandScorer

logger = logging.getLogger("aerocast.ml.interface")

_FORECAST_MODEL: Optional[AQIForecastModel] = None
_UHI_SCORER: Optional[HeatIslandScorer] = None
_ML_CACHE_DIR = Path(settings.CACHE_DIR) / "ml"
_CACHE_TTL_MINUTES = 30


def _get_forecast_model(version: str = "v4.0") -> AQIForecastModel:
    global _FORECAST_MODEL
    if _FORECAST_MODEL is None or _FORECAST_MODEL.model_version != version:
        _FORECAST_MODEL = AQIForecastModel(model_version=version)
    return _FORECAST_MODEL


def _get_uhi_scorer() -> HeatIslandScorer:
    global _UHI_SCORER
    if _UHI_SCORER is None:
        _UHI_SCORER = HeatIslandScorer()
    return _UHI_SCORER


def _read_cache(cache_file: Path) -> Optional[Dict[str, Any]]:
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = datetime.fromisoformat(data.get("timestamp_utc"))
            if datetime.now(timezone.utc) - cached_at < timedelta(minutes=_CACHE_TTL_MINUTES):
                return data.get("payload")
        except Exception:
            pass
    return None


def _write_cache(cache_file: Path, payload: Any):
    _ML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write ML cache: %s", e)


def get_aqi_forecast(zone_id: str, horizon_hours: int = 24) -> Dict[str, Any]:
    """
    Get 24-hour advance AQI prediction for a specific zone.
    """
    if horizon_hours != 24:
        raise ValueError(
            f"AeroCast v1.0 currently only supports horizon_hours=24 (got {horizon_hours}). "
            "The 48-hour forecast has been removed due to data precision limitations."
        )
    model = _get_forecast_model()
    return model.predict_zone(zone_id=zone_id, horizon_hours=24)


def get_all_aqi_forecasts(horizon_hours: int = 24, allow_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Generate 24-hour advance AQI forecasts for all 241 Lahore zones.
    """
    if horizon_hours != 24:
        raise ValueError(
            f"AeroCast v1.0 currently only supports horizon_hours=24 (got {horizon_hours}). "
            "The 48-hour forecast has been removed due to data precision limitations."
        )
    cache_file = _ML_CACHE_DIR / "forecast_24h.json"
    if allow_cache:
        cached = _read_cache(cache_file)
        if cached:
            return cached

    model = _get_forecast_model()
    from spatial.interface import get_interpolated_grid
    pm25_grid = get_interpolated_grid("aqi_pm25", allow_cache=True)

    results = {}
    for z_id in pm25_grid.keys():
        curr_pm25 = pm25_grid[z_id].get("value", 65.0)
        results[z_id] = model.predict_zone(zone_id=z_id, horizon_hours=24, current_pm25=curr_pm25)

    _write_cache(cache_file, results)
    return results


def get_heat_island_risk(zone_id: str, allow_cache: bool = True) -> Dict[str, Any]:
    """
    Get Urban Heat Island risk score and metrics for a specific zone.
    """
    if allow_cache:
        all_uhi = get_all_heat_island_risk(allow_cache=True)
        if zone_id in all_uhi:
            return all_uhi[zone_id]

    scorer = _get_uhi_scorer()
    return scorer.score_zone(zone_id)


def get_all_heat_island_risk(allow_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Get Urban Heat Island risk scores for all 241 Lahore zones.
    """
    cache_file = _ML_CACHE_DIR / "heat_island_scores.json"
    if allow_cache:
        cached = _read_cache(cache_file)
        if cached:
            return cached

    scorer = _get_uhi_scorer()
    results = scorer.score_all_zones()
    _write_cache(cache_file, results)
    return results


def train_and_save_models() -> Dict[str, Any]:
    """
    Trigger end-to-end training of 24h XGBoost model.
    """
    model = _get_forecast_model()
    return model.train_and_evaluate()


def get_ml_health() -> Dict[str, Any]:
    """
    Health diagnostic reporting ML model status, artifact availability, and metrics.
    """
    model = _get_forecast_model()
    models_dir = model.models_dir
    xgb_24h_ok = (models_dir / f"aqi_xgb_24h_{model.model_version}.joblib").exists()

    return {
        "status": "healthy" if xgb_24h_ok else "uninitialized",
        "models_directory": str(models_dir),
        "artifacts": {
            "xgb_24h": xgb_24h_ok,
        },
        "model_version": model.model_version,
        "evaluation_metrics": model.metrics,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
