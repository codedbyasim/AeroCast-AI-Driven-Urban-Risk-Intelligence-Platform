"""
Unit tests for IngestionScheduler and trigger_full_sync (SRS v1.1).
Validates nearest-neighbor weather matching (Issue 1), interpolation diagnostic notes (Issue 2),
and synthetic raster flagging in data_quality (Issue 4).
"""

from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, MagicMock
from ingestion.scheduler import IngestionScheduler
from ingestion.schema import NormalizedRecord, Metrics, SpatialContext, DataQuality
from ingestion.cache import LocalDataCache


@pytest.fixture
def isolated_cache(tmp_path):
    return LocalDataCache(cache_dir=tmp_path / "cache")


@pytest.mark.asyncio
async def test_full_sync_nearest_weather_and_interpolation_notes(isolated_cache):
    """
    Test trigger_full_sync():
    1. Zone without direct AQI gets interpolated=True and explanatory note (Issue 2).
    2. Zone without direct weather match gets geographically nearest grid point (Issue 1).
    3. Synthetic raster flagging in notes and confidence score (Issue 4).
    """
    # Create two weather records at different locations
    # Point A (North): lat 31.70, lon 74.35
    weather_north = NormalizedRecord(
        source="Open-Meteo",
        zone_id="ZONE-LHR-NORTH",
        metrics=Metrics(temperature_c=25.0, rainfall_mm_forecast=0.0),
        spatial_context=SpatialContext(
            zone_name="Lahore North Grid Point",
            centroid_lat=31.70,
            centroid_lon=74.35,
        ),
        data_quality=DataQuality(confidence_score=1.0),
    )

    # Point B (South): lat 31.35, lon 74.35
    weather_south = NormalizedRecord(
        source="Open-Meteo",
        zone_id="ZONE-LHR-SOUTH",
        metrics=Metrics(temperature_c=38.0, rainfall_mm_forecast=15.0),
        spatial_context=SpatialContext(
            zone_name="Lahore South Grid Point",
            centroid_lat=31.35,
            centroid_lon=74.35,
        ),
        data_quality=DataQuality(confidence_score=1.0),
    )

    # Note order in meteo_records: weather_north is first (meteo_records[0])
    mock_meteo = MagicMock()
    mock_meteo.fetch_grid_weather = AsyncMock(return_value=[])

    mock_openaq = MagicMock()
    mock_openaq.fetch_latest_measurements = AsyncMock(return_value=[])

    # WorldPop and Copernicus clients reporting synthetic rasters
    mock_worldpop = MagicMock()
    mock_worldpop.compute_all_zone_densities.return_value = {"ZONE-LHR-0001": 10000.0, "ZONE-LHR-0002": 5000.0}
    mock_worldpop.is_synthetic_raster.return_value = True

    mock_copernicus = MagicMock()
    mock_copernicus.compute_all_zone_terrain.return_value = {
        "ZONE-LHR-0001": {"elevation_m": 220.0, "slope_percent": 2.0, "ndvi_index": 0.3},
        "ZONE-LHR-0002": {"elevation_m": 208.0, "slope_percent": 1.5, "ndvi_index": 0.4},
    }
    mock_copernicus.is_synthetic_raster.return_value = True

    # Scheduler instance with isolated cache
    scheduler = IngestionScheduler(
        cache=isolated_cache,
        worldpop_client=mock_worldpop,
        copernicus_client=mock_copernicus,
    )

    # Mock static context for two zones:
    # Zone 1 (South zone): lat 31.36, lon 74.35 -> closer to weather_south (31.35) than weather_north (31.70)
    # Zone 2 (North zone): lat 31.69, lon 74.35 -> closer to weather_north (31.70)
    scheduler._static_context = {
        "ZONE-LHR-0001": {
            "zone_name": "Far South Zone",
            "grid_row": 18,
            "grid_col": 10,
            "centroid_lat": 31.36,
            "centroid_lon": 74.35,
            "elevation_m": 208.0,
        },
        "ZONE-LHR-0002": {
            "zone_name": "Far North Zone",
            "grid_row": 2,
            "grid_col": 10,
            "centroid_lat": 31.69,
            "centroid_lon": 74.35,
            "elevation_m": 220.0,
        },
    }
    scheduler._is_initialized = True
    scheduler._is_synthetic_raster = True

    # Mock poll jobs
    scheduler.poll_openaq_job = AsyncMock(return_value=[])  # No direct AQI stations
    scheduler.poll_openmeteo_job = AsyncMock(return_value=[weather_north, weather_south])

    # Run full sync
    res = await scheduler.trigger_full_sync()
    assert res["status"] == "success"
    assert res["zones_synced"] == 2

    # Check Zone 1 (Far South Zone)
    z1_rec = isolated_cache.get_latest_record("ZONE-LHR-0001")
    assert z1_rec is not None
    # Issue 1 verification: Must match weather_south (38.0 C), NOT weather_north (25.0 C) which was index 0
    assert z1_rec.metrics.temperature_c == 38.0
    assert z1_rec.metrics.rainfall_mm_forecast == 15.0

    # Issue 2 verification: Interpolated zone must have non-null explanatory note
    assert z1_rec.data_quality.interpolated is True
    assert z1_rec.data_quality.notes is not None
    assert "no direct AQI station" in z1_rec.data_quality.notes
    assert "Lahore South Grid Point" in z1_rec.data_quality.notes

    # Issue 4 verification: Synthetic raster note and confidence score factor
    assert "synthetic placeholder rasters" in z1_rec.data_quality.notes
    assert z1_rec.data_quality.confidence_score <= 0.70

    # Check Zone 2 (Far North Zone)
    z2_rec = isolated_cache.get_latest_record("ZONE-LHR-0002")
    assert z2_rec is not None
    # Must match weather_north (25.0 C)
    assert z2_rec.metrics.temperature_c == 25.0
    assert "Lahore North Grid Point" in z2_rec.data_quality.notes
