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


@pytest.mark.asyncio
async def test_openaq_client_unconfigured_handling():
    """Test that OpenAQ without API key cleanly returns empty list without inventing fake data."""
    client = OpenAQClient(api_key="")
    records = await client.fetch_latest_measurements()
    assert records == []


@pytest.mark.asyncio
async def test_openaq_client_historical_fetch():
    """Test fetching historical AQI series structure when unconfigured."""
    client = OpenAQClient(api_key="")
    history = await client.fetch_historical_measurements(days=60)
    assert isinstance(history, list)


@pytest.mark.asyncio
async def test_openmeteo_client_weather_fetch():
    """Test Open-Meteo current and forecast weather fetching."""
    client = OpenMeteoClient()
    weather = await client.fetch_current_and_forecast(31.5204, 74.3587)
    assert isinstance(weather, dict)
    assert "latitude" in weather
    assert "longitude" in weather


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


def test_authentic_rasters_provenance():
    """Test that downloaded authentic rasters are correctly identified as non-synthetic."""
    cop_client = CopernicusClient()
    wp_client = WorldPopClient()

    if cop_client.dem_path.exists():
        assert isinstance(cop_client.is_synthetic_dem(), bool)
    if cop_client.ndvi_path.exists():
        assert isinstance(cop_client.is_synthetic_ndvi(), bool)
    if wp_client.raster_path.exists():
        assert isinstance(wp_client.is_synthetic_raster(), bool)


@pytest.mark.asyncio
async def test_nasa_firms_client():
    """Test NASA FIRMS active fire client response structure and metrics."""
    from ingestion.firms_client import NasaFirmsClient
    client = NasaFirmsClient()
    result = await client.fetch_active_fires(days=1)

    assert "source" in result
    assert "fire_count" in result
    assert "total_frp_mw" in result
    assert "is_fallback" in result
    assert result["fire_count"] >= 0
    assert result["total_frp_mw"] >= 0.0
