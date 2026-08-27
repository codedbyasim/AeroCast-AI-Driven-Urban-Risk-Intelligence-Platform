"""
Unit tests for AeroCast Module M2 (Spatial Interpolation & Kriging Engine).
Validates Ordinary Kriging, Universal Kriging, IDW fallback, statistical confidence,
and EPA optical PM2.5 humidity calibration.
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from spatial.kriging_engine import KrigingEngine
from spatial.calibration import calibrate_pm25_optical
from spatial.covariates import CovariateManager
from ingestion.schema import NormalizedRecord, Metrics, SpatialContext, DataQuality


def test_optical_pm25_calibration():
    """Test EPA relative humidity correction for optical PM2.5 sensors."""
    # Under dry conditions (<= 30% RH), no calibration change
    assert calibrate_pm25_optical(100.0, 25.0) == 100.0
    assert calibrate_pm25_optical(100.0, None) == 100.0

    # At 80% RH, optical reading overestimation should be scaled down
    calibrated_80 = calibrate_pm25_optical(100.0, 80.0)
    # Expected factor: 1 + 0.24 * (0.8^2) = 1.1536 -> 100 / 1.1536 ≈ 86.68
    assert calibrated_80 < 100.0
    assert 84.0 <= calibrated_80 <= 88.0

    # At 100% RH, maximum hygroscopic growth scaled down
    calibrated_100 = calibrate_pm25_optical(100.0, 100.0)
    assert calibrated_100 < calibrated_80
    assert 78.0 <= calibrated_100 <= 82.0


def test_kriging_synthetic_gradient_accuracy():
    """
    Test Ordinary Kriging accuracy on a synthetic gradient field with known ground truth.
    Field: z = 100 + 50 * (lon - 74.0) + 30 * (lat - 31.0)
    """
    engine = KrigingEngine()

    # Generate 10 synthetic control points
    lons = [74.1, 74.2, 74.3, 74.4, 74.5, 74.25, 74.35, 74.45, 74.15, 74.48]
    lats = [31.2, 31.3, 31.4, 31.5, 31.6, 31.35, 31.45, 31.55, 31.25, 31.58]

    records = {}
    for i, (lon, lat) in enumerate(zip(lons, lats), start=1):
        z_id = f"ZONE-LHR-{i:04d}"
        true_val = 100.0 + 50.0 * (lon - 74.0) + 30.0 * (lat - 31.0)
        records[z_id] = NormalizedRecord(
            source="TestAQI",
            zone_id=z_id,
            metrics=Metrics(aqi_pm25=true_val),
            spatial_context=SpatialContext(centroid_lat=lat, centroid_lon=lon),
            data_quality=DataQuality(interpolated=False, confidence_score=1.0),
        )

    # Interpolate
    results = engine.interpolate_variable("aqi_pm25", records_map=records)
    assert len(results) >= 200, "Interpolation did not cover the full zone grid"

    # Verify that zone predictions follow the synthetic gradient closely
    covs = engine.covariate_manager.get_all_covariates()
    residuals = []
    for z_id, res in results.items():
        c_lon = covs[z_id]["centroid_lon"]
        c_lat = covs[z_id]["centroid_lat"]
        true_gradient_val = 100.0 + 50.0 * (c_lon - 74.0) + 30.0 * (c_lat - 31.0)
        pred = res["value"]
        residuals.append(abs(pred - true_gradient_val))

    mean_absolute_error = np.mean(residuals)
    assert mean_absolute_error < 10.0, f"Mean absolute error {mean_absolute_error:.2f} is higher than expected"


def test_low_control_points_idw_fallback():
    """
    Test that < 4 control points gracefully triggers IDW fallback
    and assigns lower confidence scores without crashing.
    """
    engine = KrigingEngine()

    # Provide only 2 control points
    records = {
        "ZONE-LHR-0010": NormalizedRecord(
            source="TestAQI",
            zone_id="ZONE-LHR-0010",
            metrics=Metrics(no2_ppb=45.0),
            spatial_context=SpatialContext(centroid_lat=31.50, centroid_lon=74.30),
            data_quality=DataQuality(interpolated=False),
        ),
        "ZONE-LHR-0020": NormalizedRecord(
            source="TestAQI",
            zone_id="ZONE-LHR-0020",
            metrics=Metrics(no2_ppb=25.0),
            spatial_context=SpatialContext(centroid_lat=31.55, centroid_lon=74.40),
            data_quality=DataQuality(interpolated=False),
        ),
    }

    results = engine.interpolate_variable("no2_ppb", records_map=records)
    assert len(results) >= 200

    sample_zone = next(iter(results.values()))
    assert sample_zone["method"] == "idw_fallback"
    assert sample_zone["confidence_score"] <= 0.60, "IDW fallback confidence should be penalized"
    assert 20.0 <= sample_zone["value"] <= 50.0


def test_confidence_gradient_direct_vs_interpolated():
    """
    Test FR-SPATIAL-03: Direct sensor zones should have higher confidence
    than distant unmonitored zones.
    """
    engine = KrigingEngine()

    records = {
        f"ZONE-LHR-{i:04d}": NormalizedRecord(
            source="TestAQI",
            zone_id=f"ZONE-LHR-{i:04d}",
            metrics=Metrics(temperature_c=28.0 + (i % 5)),
            spatial_context=SpatialContext(centroid_lat=31.50 + 0.02 * i, centroid_lon=74.30 + 0.02 * i),
            data_quality=DataQuality(interpolated=False),
        )
        for i in range(1, 8)
    }

    results = engine.interpolate_variable("temperature_c", records_map=records)

    direct_confidences = [
        res["confidence_score"] for res in results.values() if res["is_direct_sensor"]
    ]
    interpolated_confidences = [
        res["confidence_score"] for res in results.values() if not res["is_direct_sensor"]
    ]

    assert len(direct_confidences) > 0
    assert len(interpolated_confidences) > 0
    assert np.mean(direct_confidences) > np.mean(interpolated_confidences)


def test_confidence_discriminates_across_variance_gradient():
    """
    Regression test: Ensure confidence scores strictly discriminate across zones
    with meaningfully different kriging variances (prevents saturation/flat scores bug).
    """
    engine = KrigingEngine()

    # Create 8 scattered control points
    lons = [74.20, 74.25, 74.30, 74.35, 74.40, 74.45, 74.22, 74.38]
    lats = [31.40, 31.45, 31.50, 31.55, 31.60, 31.42, 31.52, 31.58]

    records = {
        f"ZONE-LHR-{i:04d}": NormalizedRecord(
            source="TestAQI",
            zone_id=f"ZONE-LHR-{i:04d}",
            metrics=Metrics(aqi_pm25=35.0 + 5.0 * i),
            spatial_context=SpatialContext(centroid_lat=lat, centroid_lon=lon),
            data_quality=DataQuality(interpolated=False),
        )
        for i, (lon, lat) in enumerate(zip(lons, lats), start=1)
    }

    results = engine.interpolate_variable("aqi_pm25", records_map=records)

    # Filter unmonitored zones
    unmonitored = [r for r in results.values() if not r["is_direct_sensor"]]
    assert len(unmonitored) >= 50

    # Sort unmonitored zones by variance
    sorted_by_var = sorted(unmonitored, key=lambda x: x["variance"])
    low_var_zone = sorted_by_var[0]
    high_var_zone = sorted_by_var[-1]

    # Verify meaningful variance spread
    assert high_var_zone["variance"] > low_var_zone["variance"] * 1.1

    # Verify confidence scores are NOT identical / saturated
    conf_diff = low_var_zone["confidence_score"] - high_var_zone["confidence_score"]
    assert conf_diff >= 0.05, f"Confidence scores did not discriminate variance spread (diff: {conf_diff:.4f})"

    # Check distinct confidence scores exist across the grid
    unique_confidences = len(set(r["confidence_score"] for r in unmonitored))
    assert unique_confidences >= 5, f"Only {unique_confidences} unique confidence scores across unmonitored zones"
