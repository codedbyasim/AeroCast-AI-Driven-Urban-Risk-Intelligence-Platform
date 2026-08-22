"""
Unit tests for DataNormalizer (SRS v1.1).
Validates normalization of OpenAQ and Open-Meteo observations into canonical Zone records,
and verifies FR-INGEST-09 fallback data-quality flagging and notes preservation.
"""

from datetime import datetime, timezone
import pytest
from ingestion.normalizer import DataNormalizer
from ingestion.osm_hdx_client import OSMHDXClient
from ingestion.schema import NormalizedRecord, Metrics, SpatialContext, DataQuality


@pytest.fixture
def normalizer():
    return DataNormalizer()


def test_normalize_openaq_real_station(normalizer):
    """Test normalizer on real OpenAQ station data."""
    raw_station = {
        "station_id": "LHR-AQ-01",
        "station_name": "US Consulate Lahore / Gulberg",
        "latitude": 31.5165,
        "longitude": 74.3496,
        "pm25": 145.2,
        "pm10": 180.5,
        "no2": 32.0,
        "timestamp_utc": "2026-08-22T10:00:00Z",
        "source": "OpenAQ",
        "is_fallback": False,
    }

    norm = normalizer.normalize_openaq_record(raw_station)
    assert norm is not None
    assert norm.schema_version == "1.1"
    assert norm.zone_id.startswith("ZONE-LHR-")
    assert norm.metrics.aqi_pm25 == 145.2
    assert norm.metrics.aqi_pm10 == 180.5
    assert norm.data_quality.confidence_score == 1.0
    assert norm.data_quality.notes is None


def test_normalize_openaq_fallback_tagging(normalizer):
    """Test that synthetic fallback OpenAQ data is properly flagged (FR-INGEST-09 / R-01)."""
    raw_fallback = {
        "station_id": "LHR-AQ-01",
        "latitude": 31.5165,
        "longitude": 74.3496,
        "pm25": 165.0,
        "source": "OpenAQ-Synthetic",
        "is_fallback": True,
        "fallback_reason": "synthetic fallback — OpenAQ API unreachable",
    }

    norm = normalizer.normalize_openaq_record(raw_fallback)
    assert norm is not None
    assert norm.data_quality.notes == "synthetic fallback — OpenAQ API unreachable"
    assert norm.data_quality.confidence_score <= 0.50
    assert norm.data_quality.interpolated is True


def test_normalize_openmeteo_record(normalizer):
    """Test normalizer on Open-Meteo forecast data."""
    raw_weather = {
        "latitude": 31.5204,
        "longitude": 74.3587,
        "temperature_c": 35.5,
        "rainfall_mm_forecast": 12.4,
        "wind_speed_kmh": 14.0,
        "relative_humidity_percent": 65.0,
        "surface_pressure_hpa": 1010.0,
        "source": "Open-Meteo",
        "is_fallback": False,
    }

    norm = normalizer.normalize_openmeteo_record(raw_weather)
    assert norm is not None
    assert norm.zone_id.startswith("ZONE-LHR-")
    assert norm.metrics.temperature_c == 35.5
    assert norm.metrics.rainfall_mm_forecast == 12.4
    assert norm.data_quality.confidence_score == 1.0


def test_merge_zone_snapshot(normalizer):
    """Test merging AQ and weather records into a single canonical Zone snapshot."""
    aq_rec = NormalizedRecord(
        source="OpenAQ",
        zone_id="ZONE-LHR-0020",
        metrics=Metrics(aqi_pm25=150.0),
        data_quality=DataQuality(confidence_score=1.0),
    )
    meteo_rec = NormalizedRecord(
        source="Open-Meteo",
        zone_id="ZONE-LHR-0020",
        metrics=Metrics(temperature_c=36.0, rainfall_mm_forecast=0.0),
        data_quality=DataQuality(confidence_score=1.0),
    )
    spatial_ctx = {
        "elevation_m": 218.0,
        "slope_percent": 2.8,
        "population_density_per_sqkm": 15000.0,
        "grid_row": 5,
        "grid_col": 7,
    }

    merged = normalizer.merge_zone_snapshot(
        zone_id="ZONE-LHR-0020",
        aq_record=aq_rec,
        weather_record=meteo_rec,
        spatial_context=spatial_ctx,
    )

    assert merged is not None
    assert merged.zone_id == "ZONE-LHR-0020"
    assert "OpenAQ" in merged.source and "Open-Meteo" in merged.source
    assert merged.metrics.aqi_pm25 == 150.0
    assert merged.metrics.temperature_c == 36.0
    assert merged.spatial_context.elevation_m == 218.0
    assert merged.spatial_context.grid_row == 5
    assert merged.spatial_context.grid_col == 7


def test_merge_zone_snapshot_interpolated_aq_none(normalizer):
    """Test merging when aq_record is None (interpolated zone) ensures notes are preserved (Issue 2)."""
    meteo_rec = NormalizedRecord(
        source="Open-Meteo",
        zone_id="ZONE-LHR-0099",
        metrics=Metrics(temperature_c=34.0, rainfall_mm_forecast=5.0),
        data_quality=DataQuality(confidence_score=1.0),
    )
    spatial_ctx = {
        "elevation_m": 212.0,
        "slope_percent": 2.1,
        "grid_row": 10,
        "grid_col": 12,
    }
    explicit_note = "no direct AQI station in this zone — using nearest-zone weather context only, AQI pending M2 spatial interpolation"

    merged = normalizer.merge_zone_snapshot(
        zone_id="ZONE-LHR-0099",
        aq_record=None,
        weather_record=meteo_rec,
        spatial_context=spatial_ctx,
        is_interpolated=True,
        confidence=0.85,
        notes=explicit_note,
    )

    assert merged is not None
    assert merged.zone_id == "ZONE-LHR-0099"
    assert merged.data_quality.interpolated is True
    assert merged.data_quality.confidence_score == 0.85
    assert merged.data_quality.notes is not None
    assert "no direct AQI station" in merged.data_quality.notes
    assert merged.metrics.aqi_pm25 is None
    assert merged.metrics.temperature_c == 34.0
