"""
Unit tests for AeroCast Pydantic v2 schemas (SRS v1.1 Section 6.1).
Validates NormalizedRecord, Metrics, SpatialContext, and DataQuality models.
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from ingestion.schema import NormalizedRecord, Metrics, SpatialContext, DataQuality


def test_normalized_record_valid():
    """Test valid NormalizedRecord creation and canonical dict export."""
    rec = NormalizedRecord(
        schema_version="1.1",
        source="OpenAQ | Open-Meteo",
        zone_id="ZONE-LHR-0042",
        timestamp_utc=datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc),
        metrics=Metrics(
            aqi_pm25=187.4,
            aqi_pm10=210.1,
            temperature_c=41.2,
            rainfall_mm_forecast=62.5,
            wind_speed_kmh=11.3,
        ),
        spatial_context=SpatialContext(
            elevation_m=214.0,
            slope_percent=3.1,
            impervious_surface_ratio=0.68,
            ndvi_index=0.21,
            grid_row=8,
            grid_col=14,
            centroid_lat=31.5120,
            centroid_lon=74.3310,
        ),
        data_quality=DataQuality(
            interpolated=True,
            confidence_score=0.82,
            stale=False,
            notes=None,
        ),
    )

    canonical = rec.to_canonical_dict()
    assert canonical["schema_version"] == "1.1"
    assert canonical["zone_id"] == "ZONE-LHR-0042"
    assert canonical["timestamp_utc"] == "2026-08-22T12:00:00Z"
    assert canonical["metrics"]["aqi_pm25"] == 187.4
    assert canonical["spatial_context"]["grid_row"] == 8
    assert canonical["spatial_context"]["grid_col"] == 14
    assert canonical["data_quality"]["confidence_score"] == 0.82
    assert canonical["data_quality"]["interpolated"] is True


def test_normalized_record_iso_string_timestamp():
    """Test timestamp parsing from ISO string with Z suffix."""
    rec = NormalizedRecord(
        source="OpenAQ",
        zone_id="ZONE-LHR-0001",
        timestamp_utc="2026-08-22T06:00:00Z",
    )
    assert rec.timestamp_utc.tzinfo == timezone.utc
    assert rec.timestamp_utc.hour == 6


def test_schema_missing_zone_id():
    """Test that missing zone_id triggers validation error."""
    with pytest.raises(ValidationError):
        NormalizedRecord(
            source="OpenAQ",
            # missing zone_id
        )


def test_spatial_context_bounds():
    """Test boundary validation for impervious ratio and NDVI."""
    # Invalid impervious > 1.0
    with pytest.raises(ValidationError):
        SpatialContext(impervious_surface_ratio=1.5)

    # Invalid NDVI < -1.0
    with pytest.raises(ValidationError):
        SpatialContext(ndvi_index=-2.0)
