"""
AeroCast Module M3 — Machine Learning & Predictive Analytics Engine.
Provides 24-hour Advance AQI Forecasting (XGBoost Regressor)
and Zonal Urban Heat Island (UHI) Risk Scoring for the 241 Lahore zones.
"""

from .feature_engineering import FeatureEngineer
from .aqi_forecast import AQIForecastModel
from .heat_island import HeatIslandScorer
from .interface import (
    get_aqi_forecast,
    get_all_aqi_forecasts,
    get_heat_island_risk,
    get_all_heat_island_risk,
    get_ml_health,
    train_and_save_models,
)

__all__ = [
    "FeatureEngineer",
    "AQIForecastModel",
    "HeatIslandScorer",
    "get_aqi_forecast",
    "get_all_aqi_forecasts",
    "get_heat_island_risk",
    "get_all_heat_island_risk",
    "get_ml_health",
    "train_and_save_models",
]
