"""
Unit tests for LocalDataCache (SRS v1.1).
Validates read/write, batch operations, stale fallback retrieval, and confidence degradation.
"""

from datetime import datetime, timezone
import pytest
from ingestion.cache import LocalDataCache
from ingestion.schema import NormalizedRecord, Metrics, SpatialContext, DataQuality


@pytest.fixture
def temp_cache(tmp_path):
    """Fixture providing an isolated temporary cache instance."""
    return LocalDataCache(cache_dir=tmp_path / "cache")


def create_sample_record(zone_id: str = "ZONE-LHR-0001", confidence: float = 1.0) -> NormalizedRecord:
    """Helper to create a sample NormalizedRecord."""
    return NormalizedRecord(
        schema_version="1.1",
        source="OpenAQ | Open-Meteo",
        zone_id=zone_id,
        timestamp_utc=datetime.now(timezone.utc),
        metrics=Metrics(temperature_c=32.0, aqi_pm25=120.0),
        spatial_context=SpatialContext(grid_row=1, grid_col=1, elevation_m=210.0),
        data_quality=DataQuality(confidence_score=confidence, stale=False),
    )


def test_save_and_get_latest_record(temp_cache):
    """Test saving a single record and retrieving it."""
    rec = create_sample_record("ZONE-LHR-0005")
    assert temp_cache.save_record(rec) is True

    loaded = temp_cache.get_latest_record("ZONE-LHR-0005")
    assert loaded is not None
    assert loaded.zone_id == "ZONE-LHR-0005"
    assert loaded.metrics.temperature_c == 32.0
    assert loaded.metrics.aqi_pm25 == 120.0


def test_save_records_batch(temp_cache):
    """Test batch caching multiple records."""
    records = [
        create_sample_record(f"ZONE-LHR-{i:04d}")
        for i in range(1, 11)
    ]
    saved_count = temp_cache.save_records(records)
    assert saved_count == 10

    all_latest = temp_cache.get_all_latest()
    assert len(all_latest) == 10
    assert "ZONE-LHR-0003" in all_latest


def test_stale_fallback_degrades_confidence(temp_cache):
    """Test that retrieving stale fallback marks stale=True and lowers confidence."""
    rec = create_sample_record("ZONE-LHR-0010", confidence=1.0)
    temp_cache.save_record(rec)

    stale_rec = temp_cache.get_stale_fallback("ZONE-LHR-0010", reason="API timeout")
    assert stale_rec is not None
    assert stale_rec.data_quality.stale is True
    assert stale_rec.data_quality.notes == "API timeout"
    assert stale_rec.data_quality.confidence_score < 1.0


def test_clear_cache(temp_cache):
    """Test clearing all cache files."""
    rec = create_sample_record("ZONE-LHR-0001")
    temp_cache.save_record(rec)
    assert len(temp_cache.get_all_latest()) == 1

    temp_cache.clear_cache()
    assert len(temp_cache.get_all_latest()) == 0
