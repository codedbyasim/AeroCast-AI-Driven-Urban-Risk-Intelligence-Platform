"""
Unit Tests for Module M8: Backtesting & Continuous Model Evaluation Engine.
Validates continuous metrics, extreme event metrics, drift detectors, and REST API endpoints.
"""

import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from api.app import create_app

from backtesting.metrics import (
    calculate_continuous_metrics,
    calculate_extreme_event_metrics,
    calculate_directional_accuracy,
    detect_model_drift,
)
from backtesting.interface import (
    run_backtest,
    get_latest_backtest_results,
    get_drift_status,
    get_backtesting_health,
)


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_continuous_metrics_calculation():
    """Verify regression metrics calculation accuracy and safeguards."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0, 52.0])

    metrics = calculate_continuous_metrics(y_true, y_pred)
    assert metrics["sample_count"] == 5
    assert metrics["mae"] == 2.2
    assert metrics["rmse"] > 0
    assert metrics["r2"] > 0.90
    assert metrics["median_ae"] == 2.0


def test_extreme_event_metrics_calculation():
    """Verify binary confusion matrix and classification metrics for extreme events."""
    y_true = np.array([50.0, 120.0, 160.0, 40.0, 180.0])
    y_pred = np.array([45.0, 110.0, 140.0, 95.0, 175.0])

    # Threshold = 100.0: Actual Positives = 3 (120, 160, 180)
    # Predicted Positives = 3 (110, 140, 175) -> All 3 are True Positives
    metrics_100 = calculate_extreme_event_metrics(y_true, y_pred, threshold=100.0)
    assert metrics_100["true_positives"] == 3
    assert metrics_100["false_positives"] == 0
    assert metrics_100["false_negatives"] == 0
    assert metrics_100["recall_sensitivity"] == 1.0
    assert metrics_100["f1_score"] == 1.0


def test_directional_accuracy_calculation():
    """Verify trajectory directional accuracy calculation."""
    y_current = np.array([50.0, 50.0, 50.0, 50.0])
    y_true = np.array([60.0, 40.0, 70.0, 30.0])    # [+1, -1, +1, -1]
    y_pred = np.array([55.0, 45.0, 48.0, 20.0])    # [+1, -1, -1, -1] -> 3/4 correct

    acc = calculate_directional_accuracy(y_true, y_pred, y_current)
    assert acc == 0.75


def test_detect_model_drift_thresholds():
    """Verify model drift detector status transitions."""
    # Healthy baseline
    d_healthy = detect_model_drift(current_mae=21.5, current_r2=0.74, baseline_mae=20.93)
    assert d_healthy["drift_status"] == "HEALTHY"
    assert d_healthy["is_retraining_required"] is False

    # Degraded
    d_deg = detect_model_drift(current_mae=26.0, current_r2=0.60, baseline_mae=20.93)
    assert d_deg["drift_status"] == "DEGRADED"

    # Critical Drift
    d_crit = detect_model_drift(current_mae=32.0, current_r2=0.15, baseline_mae=20.93)
    assert d_crit["drift_status"] == "CRITICAL_DRIFT"
    assert d_crit["is_retraining_required"] is True


def test_backtest_facade_and_reporting():
    """Verify end-to-end backtest execution and artifact persistence."""
    res = run_backtest(export_csv=True)
    assert res["canonical_zones_evaluated"] == 241
    assert "validation_holdout_results" in res
    assert "future_test_holdout_results" in res
    assert "spatial_kriging_cross_validation" in res
    assert "model_drift_monitoring" in res

    # Verify CSV export exists
    csv_file = Path("reports/BACKTEST_EVALUATION_REPORT.csv")
    assert csv_file.exists()


def test_backtest_rest_api_endpoints(client):
    """Test Module M8 REST API endpoints."""
    # 1. Health
    r_health = client.get("/api/v1/backtest/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    # 2. Latest
    r_lat = client.get("/api/v1/backtest/latest")
    assert r_lat.status_code == 200
    assert "validation_holdout_results" in r_lat.json()

    # 3. Drift Status
    r_drift = client.get("/api/v1/backtest/drift-status")
    assert r_drift.status_code == 200
    assert "drift_status" in r_drift.json()

    # 4. Trigger Run
    r_run = client.post("/api/v1/backtest/run?export_csv=true")
    assert r_run.status_code == 200
    assert r_run.json()["canonical_zones_evaluated"] == 241
