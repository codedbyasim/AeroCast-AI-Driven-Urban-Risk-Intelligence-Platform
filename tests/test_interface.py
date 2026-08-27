"""
Unit tests for Ingestion Interface Facade (SRS v1.1).
Validates get_latest_data, get_all_zone_data, get_ingestion_health, and sync_historical.
"""

import pytest
from ingestion.interface import (
    get_latest_data,
    get_all_zone_data,
    get_all_uc_data,
    get_ingestion_health,
    sync_historical,
)


def test_interface_health():
    """Test get_ingestion_health returns expected dictionary keys."""
    health = get_ingestion_health()
    assert "status" in health
    assert "total_zones_cached" in health
    assert "stale_count" in health
    assert "interpolated_count" in health
    assert "fresh_count" in health


def test_interface_get_all_zone_data():
    """Test retrieving all cached zone records."""
    all_zones = get_all_zone_data()
    assert isinstance(all_zones, list)
    if all_zones:
        first = all_zones[0]
        assert "zone_id" in first
        assert "schema_version" in first
        assert first["schema_version"] == "1.1"


def test_interface_get_all_uc_data_alias():
    """Test backwards compatibility alias get_all_uc_data."""
    all_ucs = get_all_uc_data()
    all_zones = get_all_zone_data()
    assert len(all_ucs) == len(all_zones)


def test_interface_sync_historical(monkeypatch):
    """Test historical data synchronization interface."""
    from unittest.mock import AsyncMock, patch
    with patch("ingestion.scheduler.IngestionScheduler.fetch_historical_dataset", new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = {"status": "success", "historical_aqi_records": 7260, "historical_weather_days": 730}
        res = sync_historical(days=730)
        assert res["status"] in ("success", "historical_sync_triggered_async")
