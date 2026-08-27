"""
Unit tests for AeroCast Module M2 Covariates (NDVI & Road Density).
Validates road network density calculation, GeoJSON caching, and synthetic provenance flags.
"""

import json
import pytest
from pathlib import Path
from spatial.covariates import CovariateManager, ROAD_DENSITY_PATH


def test_road_density_computation_and_cache(tmp_path):
    """Test road density calculation and caching to GeoJSON."""
    test_road_path = tmp_path / "test_lahore_road_density.geojson"
    mgr = CovariateManager(road_density_path=test_road_path)

    # Compute
    density_map = mgr.load_or_compute_road_density(force_refresh=True)

    assert test_road_path.exists(), "Road density GeoJSON was not created"
    assert len(density_map) >= 200, f"Expected >= 200 zones, got {len(density_map)}"

    # Check properties of a sample zone
    sample = next(iter(density_map.values()))
    assert "road_density_km_per_sqkm" in sample
    assert "estimated_road_length_km" in sample
    assert sample["road_density_km_per_sqkm"] > 0.5

    # Check reload from cache
    reloaded_map = mgr.load_or_compute_road_density(force_refresh=False)
    assert len(reloaded_map) == len(density_map)


def test_all_covariates_structure():
    """Test get_all_covariates returns complete context for all 241 zones."""
    mgr = CovariateManager()
    covs = mgr.get_all_covariates()

    assert len(covs) >= 200
    required_keys = {
        "zone_id",
        "grid_row",
        "grid_col",
        "area_sqkm",
        "centroid_lat",
        "centroid_lon",
        "ndvi_index",
        "elevation_m",
        "slope_percent",
        "road_density_km_per_sqkm",
        "is_synthetic_covariates",
    }

    for z_id, data in covs.items():
        assert z_id.startswith("ZONE-LHR-")
        for k in required_keys:
            assert k in data, f"Missing key {k} in zone {z_id}"
        assert -1.0 <= data["ndvi_index"] <= 1.0
        assert data["road_density_km_per_sqkm"] > 0.0


def test_synthetic_ndvi_provenance():
    """Test that synthetic raster status is correctly exposed."""
    mgr = CovariateManager()
    is_synth = mgr.is_synthetic_ndvi()
    assert isinstance(is_synth, bool)
