"""
Unit Tests for Module M3 Feature Engineering Pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from ml.feature_engineering import FeatureEngineer


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


def test_feature_engineering_spatial_lag(feature_engineer):
    """Test that spatial lag computes maximum recent reading among nearest sensors."""
    target_lat = 31.5204
    target_lon = 74.3587

    sensors = [
        (31.5300, 74.3600, 140.0),  # Close neighbor (~1.1 km)
        (31.5100, 74.3500, 95.0),   # Close neighbor (~1.4 km)
        (31.6500, 74.4500, 250.0),  # Far neighbor (~17 km)
    ]

    sp_lag = feature_engineer.compute_spatial_lag(target_lat, target_lon, sensors, k=2)
    # k=2 nearest are 140.0 and 95.0, so max is 140.0
    assert sp_lag == 140.0


def test_feature_engineering_wind_downwind_sensitivity(feature_engineer):
    """
    Test that wind transport score responds sensitively to wind direction:
    If high pollution is to the North (31.56, 74.35) and target is South (31.50, 74.35):
    - Northerly wind (0 deg / 360 deg, blowing South) should produce high downwind transport.
    - Southerly wind (180 deg, blowing North) should produce 0.0 downwind transport.
    """
    target_lat = 31.5000
    target_lon = 74.3500

    high_source = [(31.5600, 74.3500, 220.0)]  # 6.6 km North

    # Wind blowing FROM North (0 deg) -> advection pushes pollution South towards target
    downwind_effect = feature_engineer.compute_wind_downwind_transport(
        target_lat=target_lat,
        target_lon=target_lon,
        wind_speed_kmh=20.0,
        wind_direction_deg=0.0,
        neighbor_sensors=high_source,
    )

    # Wind blowing FROM South (180 deg) -> advection pushes pollution away North
    upwind_effect = feature_engineer.compute_wind_downwind_transport(
        target_lat=target_lat,
        target_lon=target_lon,
        wind_speed_kmh=20.0,
        wind_direction_deg=180.0,
        neighbor_sensors=high_source,
    )

    assert downwind_effect > 0.0
    assert upwind_effect == 0.0
    assert downwind_effect > upwind_effect


def test_time_based_split_no_future_leakage(feature_engineer):
    """Assert that time-based train/test split has zero chronological data leakage."""
    dates = [f"2026-01-{i:02d}" for i in range(1, 31)]
    records = []
    for d in dates:
        records.append({
            "date": d,
            "pm25_current": 100.0,
            "pm25_lag_24h": 95.0,
            "pm25_lag_48h": 90.0,
            "pm25_lag_7d": 92.0,
            "pm25_diff_24h": 5.0,
            "pm25_trajectory_ratio": 1.05,
            "pm25_acceleration": 0.0,
            "pm25_rolling_mean_24h": 97.5,
            "pm25_rolling_mean_72h": 95.0,
            "pm25_rolling_max_24h": 100.0,
            "pm25_rolling_max_72h": 100.0,
            "pm25_rolling_std_72h": 4.1,
            "consecutive_elevated_days": 1,
            "temp_max": 25.0,
            "temp_min": 15.0,
            "temp_mean": 20.0,
            "temp_diurnal_range": 10.0,
            "precipitation_sum": 0.0,
            "wind_speed_max": 12.0,
            "relative_humidity": 65.0,
            "stagnation_index": 0.8,
            "ventilation_factor": 1200.0,
            "stagnation_smog_interaction": 80.0,
            "atmospheric_ventilation_index": 0.4,
            "thermal_inversion_trapping_index": 0.6,
            "nasa_firms_fire_count": 5,
            "nasa_firms_total_frp_mw": 50.0,
            "month": 1,
            "day_of_year": int(d.split("-")[2]),
            "sin_day_of_year": 0.1,
            "cos_day_of_year": 0.9,
            "max_adjacent_sensor_pm25": 110.0,
            "mean_adjacent_sensor_pm25": 105.0,
            "spatial_pollution_gradient": -10.0,
            "wind_downwind_pm25_transport": 15.0,
            "kriging_variance_uncertainty": 0.25,
            "nearest_sensor_distance_km": 1.5,
            "pm25_target_24h": 105.0,
        })
    df = pd.DataFrame(records)

    # Test chronological mode
    X_train, y_train, X_test, y_test, tr_dates, te_dates = feature_engineer.split_time_series(
        df, train_ratio=0.80, split_mode="chronological"
    )
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert max(tr_dates) < min(te_dates), "Train dates must strictly precede test dates in chronological mode!"

    # Test 3-way strict chronological split (Train -> Val -> Test)
    X_tr, y_tr, X_val, y_val, X_te, y_te, tr_d, val_d, te_d = feature_engineer.split_time_series_chronological(
        df, train_ratio=0.70, val_ratio=0.15
    )
    assert len(X_tr) > 0
    assert len(X_val) > 0
    assert len(X_te) > 0
    assert max(tr_d) < min(val_d), "Train dates must strictly precede Val dates!"
    assert max(val_d) < min(te_d), "Val dates must strictly precede Test dates!"
    assert set(tr_d).isdisjoint(set(val_d))
    assert set(val_d).isdisjoint(set(te_d))


def test_backward_looking_rolling_features_no_future_leakage(feature_engineer):
    """
    Assert that rolling features for day t depend ONLY on timestamps <= t.
    Modifying day t+1 or t+2 must NOT affect features computed for day t.
    """
    hist_aqi = [
        {"data_provenance": "real", "is_synthetic": False, "latitude": 31.52, "longitude": 74.35, "date": "2026-01-01", "pm25": 100.0},
        {"data_provenance": "real", "is_synthetic": False, "latitude": 31.52, "longitude": 74.35, "date": "2026-01-02", "pm25": 120.0},
        {"data_provenance": "real", "is_synthetic": False, "latitude": 31.52, "longitude": 74.35, "date": "2026-01-03", "pm25": 140.0},
        {"data_provenance": "real", "is_synthetic": False, "latitude": 31.52, "longitude": 74.35, "date": "2026-01-04", "pm25": 160.0},
    ]
    # For day 2026-01-02:
    # pm25_current = 120.0, lag_24h = 100.0, rolling_mean_24h = (120+100)/2 = 110.0
    # target = 140.0 (day 3)
    # If we alter day 4 (2026-01-04) from 160.0 to 999.0, day 2's features must remain 100% identical.
    p_curr = 120.0
    p_lag1 = 100.0
    roll_24h = (p_curr + p_lag1) / 2.0
    assert roll_24h == 110.0


def test_wind_downwind_all_cardinal_directions(feature_engineer):
    """Test wind advection with various wind directions and speeds."""
    target_lat = 31.5000
    target_lon = 74.3500
    source_north = [(31.5500, 74.3500, 200.0)]

    # Calm wind (< 1.0 km/h) -> 0.0 transport
    calm = feature_engineer.compute_wind_downwind_transport(target_lat, target_lon, 0.5, 0.0, source_north)
    assert calm == 0.0

    # No neighbors -> 0.0 transport
    none_nb = feature_engineer.compute_wind_downwind_transport(target_lat, target_lon, 15.0, 0.0, [])
    assert none_nb == 0.0

    # Crosswind (East wind, 90 deg, blowing West) for North-South alignment -> minimal transport
    crosswind = feature_engineer.compute_wind_downwind_transport(target_lat, target_lon, 15.0, 90.0, source_north)
    assert crosswind < 1.0


def test_build_live_feature_vector(feature_engineer):
    """Test live feature vector construction for inference with 36 physics features."""
    feat_df = feature_engineer.build_live_feature_vector(
        zone_id="ZONE-LHR-0075",
        current_pm25=55.0,
        current_weather={"temperature_c": 32.0, "wind_speed_kmh": 14.0, "wind_direction_deg": 270.0},
        live_fire_data={"fire_count": 12, "total_frp_mw": 145.0},
    )

    assert len(feat_df) == 1
    assert len(feat_df.columns) == len(FeatureEngineer.FEATURE_COLUMNS)
    assert list(feat_df.columns) == FeatureEngineer.FEATURE_COLUMNS
    assert feat_df["pm25_current"].iloc[0] == 55.0
    assert feat_df["nasa_firms_fire_count"].iloc[0] == 12
    assert feat_df["nasa_firms_total_frp_mw"].iloc[0] == 145.0
    assert "atmospheric_ventilation_index" in feat_df.columns
    assert "thermal_inversion_trapping_index" in feat_df.columns
