"""
Unit Tests for Module M3 AQI Forecasting Model (Two-Stage Probabilistic XGBoost 24-hour Advance).
"""

import pytest
import numpy as np
import pandas as pd
from ml.aqi_forecast import AQIForecastModel
from ml.interface import get_aqi_forecast, get_all_aqi_forecasts, get_ml_health


@pytest.fixture
def aqi_model(tmp_path):
    return AQIForecastModel(models_dir=tmp_path, model_version="v4.0")


def test_aqi_forecast_train_and_evaluate(aqi_model):
    """Test 24h two-stage model training pipeline with 44 physics & FIRMS features."""
    dates = [f"2026-06-{i:02d}" for i in range(1, 30)]
    records = []
    for d in dates:
        for z in ["ZONE-LHR-0001", "ZONE-LHR-0002"]:
            p_c = 50.0 + np.random.uniform(-5, 5)
            records.append({
                "date": d,
                "zone_id": z,
                "pm25_current": p_c,
                "pm25_lag_24h": 48.0,
                "pm25_lag_48h": 46.0,
                "pm25_lag_7d": 45.0,
                "pm25_diff_24h": 2.0,
                "pm25_trajectory_ratio": 1.04,
                "pm25_acceleration": 0.0,
                "pm25_rolling_mean_24h": 49.0,
                "pm25_rolling_mean_72h": 48.0,
                "pm25_rolling_max_24h": 52.0,
                "pm25_rolling_max_72h": 54.0,
                "pm25_rolling_std_72h": 2.5,
                "consecutive_elevated_days": 0,
                "temp_max": 35.0,
                "temp_min": 25.0,
                "temp_mean": 30.0,
                "temp_diurnal_range": 10.0,
                "precipitation_sum": 0.0,
                "wind_speed_max": 10.0,
                "relative_humidity": 55.0,
                "stagnation_index": 0.95,
                "ventilation_factor": 400.0,
                "stagnation_smog_interaction": 0.0,
                "atmospheric_ventilation_index": 0.4,
                "thermal_inversion_trapping_index": 0.5,
                "nasa_firms_fire_count": 5,
                "nasa_firms_total_frp_mw": 45.0,
                "month": 6,
                "day_of_year": 170,
                "sin_day_of_year": 0.2,
                "cos_day_of_year": -0.9,
                "max_adjacent_sensor_pm25": 55.0,
                "mean_adjacent_sensor_pm25": 52.0,
                "spatial_pollution_gradient": -5.0,
                "wind_downwind_pm25_transport": 5.0,
                "kriging_variance_uncertainty": 0.3,
                "nearest_sensor_distance_km": 1.8,
                "pm25_target_24h": 52.0 + np.random.uniform(-3, 3),
            })
    df = pd.DataFrame(records)

    metrics = aqi_model.train_and_evaluate(df, train_ratio=0.70, val_ratio=0.15)

    assert "combined_annual_holdout" in metrics
    assert metrics["combined_annual_holdout"]["mae"] > 0
    assert (aqi_model.models_dir / f"aqi_xgb_24h_{aqi_model.model_version}.joblib").exists()
    assert (aqi_model.models_dir / f"aqi_xgb_classifier_{aqi_model.model_version}.joblib").exists()
    assert (aqi_model.models_dir / f"aqi_xgb_p90_{aqi_model.model_version}.joblib").exists()
    assert (aqi_model.models_dir / "metadata_v4.json").exists()


def test_aqi_forecast_predict_zone():
    """Test 24h zone probabilistic inference, quantiles, and extreme spike probability."""
    pred_24h = get_aqi_forecast("ZONE-LHR-0075", horizon_hours=24)

    assert pred_24h["zone_id"] == "ZONE-LHR-0075"
    assert pred_24h["horizon_hours"] == 24
    assert pred_24h["forecasted_pm25"] > 0.0
    assert "probabilistic_quantiles" in pred_24h
    assert "p10_lower_bound_ug_m3" in pred_24h["probabilistic_quantiles"]
    assert "p90_worst_case_ceiling_ug_m3" in pred_24h["probabilistic_quantiles"]
    assert "extreme_spike_probability" in pred_24h
    assert 0.0 <= pred_24h["extreme_spike_probability"] <= 1.0
    assert "physics_drivers" in pred_24h

    # 48-hour forecast is explicitly removed and must raise ValueError
    with pytest.raises(ValueError, match="horizon_hours=24"):
        get_aqi_forecast("ZONE-LHR-0075", horizon_hours=48)


def test_aqi_forecast_all_zones():
    """Test batch forecasting across all 241 zones."""
    forecasts = get_all_aqi_forecasts(horizon_hours=24, allow_cache=True)
    assert len(forecasts) == 241
    assert "ZONE-LHR-0001" in forecasts
    assert "ZONE-LHR-0241" in forecasts

    with pytest.raises(ValueError, match="horizon_hours=24"):
        get_all_aqi_forecasts(horizon_hours=48)


def test_ml_health():
    """Test ML diagnostic health status endpoint."""
    health = get_ml_health()
    assert health["status"] in ("healthy", "uninitialized")
    assert "artifacts" in health
    assert "xgb_24h" in health["artifacts"]
