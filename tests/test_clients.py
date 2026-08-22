"""
Unit tests for external API and raster ingestion clients (SRS v1.1).
Validates OpenAQ, Open-Meteo, Copernicus, WorldPop, and OSM clients.
"""

import pytest
from shapely.geometry import Polygon
from ingestion.openaq_client import OpenAQClient
from ingestion.openmeteo_client import OpenMeteoClient
from ingestion.copernicus_client import CopernicusClient
from ingestion.worldpop_client import WorldPopClient
from ingestion.osm_hdx_client import OSMHDXClient


def test_osm_hdx_client_zone_lookup():
    """Test OSM client spatial point-in-polygon lookup."""
    client = OSMHDXClient()
    zone_info = client.find_zone_by_coordinates(31.5204, 74.3587)

    assert zone_info is not None
    assert "zone_id" in zone_info
    assert zone_info["zone_id"].startswith("ZONE-LHR-")
    assert "grid_row" in zone_info
    assert "grid_col" in zone_info
    assert "centroid_lat" in zone_info
    assert "centroid_lon" in zone_info


def test_openaq_client_fallback_flagging():
    """Test that OpenAQ fallback observations include proper flagging."""
    client = OpenAQClient(api_key="")  # Unset key forces fallback
    records = client.generate_fallback_observations()

    assert len(records) > 0
    for rec in records:
        assert rec["is_fallback"] is True
        assert "synthetic fallback" in rec["fallback_reason"]
        assert rec["pm25"] is not None
        assert rec["pm10"] is not None


@pytest.mark.asyncio
async def test_openaq_client_historical_fetch():
    """Test fetching multi-year historical AQI series."""
    client = OpenAQClient()
    history = await client.fetch_historical_measurements(days=60)

    assert len(history) == 60
    assert "pm25" in history[0]
    assert "date" in history[0]
    assert history[0]["is_fallback"] is True


def test_openmeteo_client_fallback_flagging():
    """Test that Open-Meteo fallback snapshots are properly tagged."""
    client = OpenMeteoClient()
    fallback = client._generate_fallback_weather(31.5204, 74.3587)

    assert fallback["is_fallback"] is True
    assert "synthetic fallback" in fallback["fallback_reason"]
    assert fallback["temperature_c"] is not None
    assert fallback["rainfall_mm_forecast"] is not None


def test_worldpop_client_density():
    """Test WorldPop client density sampling."""
    client = WorldPopClient()
    density = client.get_population_density_at_point(31.545, 74.325)
    assert density > 1000.0


def test_copernicus_client_terrain():
    """Test Copernicus client terrain sampling."""
    client = CopernicusClient()
    poly = Polygon([(74.30, 31.50), (74.33, 31.50), (74.33, 31.53), (74.30, 31.53), (74.30, 31.50)])
    metrics = client.calculate_zone_terrain_metrics(poly)

    assert "elevation_m" in metrics
    assert "slope_percent" in metrics
    assert "ndvi_index" in metrics
    assert 180.0 <= metrics["elevation_m"] <= 300.0
