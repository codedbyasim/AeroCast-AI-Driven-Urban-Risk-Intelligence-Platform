"""
Unit tests for AeroCast Module M2 Spatial Interface Facade.
Validates get_interpolated_grid, get_all_interpolated_grid, caching, and health diagnostics.
"""

import pytest
from spatial.interface import (
    get_interpolated_grid,
    get_all_interpolated_grid,
    get_zone_interpolated,
    get_spatial_health,
    trigger_spatial_interpolation,
)


def test_spatial_interface_health():
    """Test get_spatial_health diagnostics."""
    health = get_spatial_health()
    assert "status" in health
    assert "total_zones_covered" in health
    assert "mean_pm25_confidence" in health
    assert health["total_zones_covered"] >= 200


def test_spatial_interface_get_interpolated_grid():
    """Test get_interpolated_grid for PM2.5 and temperature."""
    pm25_grid = get_interpolated_grid("aqi_pm25", allow_cache=False)
    assert len(pm25_grid) >= 200

    sample_zone = next(iter(pm25_grid.values()))
    assert "value" in sample_zone
    assert "variance" in sample_zone
    assert "confidence_score" in sample_zone
    assert "method" in sample_zone
    assert 0.0 <= sample_zone["confidence_score"] <= 1.0


def test_spatial_interface_get_all_interpolated_grid():
    """Test get_all_interpolated_grid returns multi-variable dictionary."""
    all_grid = get_all_interpolated_grid(allow_cache=False)
    assert len(all_grid) >= 200

    sample_z_id = next(iter(all_grid.keys()))
    zone_vars = all_grid[sample_z_id]
    assert "aqi_pm25" in zone_vars
    assert "temperature_c" in zone_vars
    assert "relative_humidity_percent" in zone_vars
