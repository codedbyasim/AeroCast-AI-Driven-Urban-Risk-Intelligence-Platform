"""
Unit Tests for Module M9: AeroCast REST API Layer (FastAPI).
Validates endpoint responses, schema consistency, status codes, and 241-zone payload structure.
"""

import pytest
from fastapi.testclient import TestClient
from api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_api_info_endpoint(client):
    """Test API info JSON endpoint."""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["canonical_zones"] == 241


def test_system_health_endpoint(client):
    """Test aggregated health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["canonical_zones"] == 241
    assert "modules" in data
    assert "m1_ingestion" in data["modules"]
    assert "m2_spatial_kriging" in data["modules"]
    assert "m3_ml_forecasting" in data["modules"]
    assert "m4_flash_flood" in data["modules"]


def test_individual_module_health_endpoints(client):
    """Test dedicated module diagnostic endpoints."""
    for mod in ["m1", "m2", "m3", "m4"]:
        resp = client.get(f"/health/{mod}")
        assert resp.status_code == 200
        assert resp.json().get("status") == "healthy"


def test_list_zones_endpoint(client):
    """Test 241-zone directory listing."""
    response = client.get("/api/v1/zones")
    assert response.status_code == 200
    data = response.json()
    assert data["total_zones"] == 241
    assert data["canonical_grid_count"] == 241
    assert len(data["zones"]) == 241
    assert data["zones"][0]["zone_id"] == "ZONE-LHR-0001"
    assert data["zones"][-1]["zone_id"] == "ZONE-LHR-0241"


def test_get_zone_snapshot_endpoint(client):
    """Test single zone multi-hazard drill-down."""
    response = client.get("/api/v1/zones/ZONE-LHR-0075")
    assert response.status_code == 200
    data = response.json()
    assert data["zone_id"] == "ZONE-LHR-0075"
    assert "current_conditions" in data
    assert "forecast_24h_aqi" in data
    assert "urban_heat_island" in data
    assert "flash_flood_risk" in data
    assert "terrain_covariates" in data


def test_get_invalid_zone_404(client):
    """Test that requesting a non-existent zone ID returns 404."""
    response = client.get("/api/v1/zones/ZONE-LHR-9999")
    assert response.status_code == 404


def test_lookup_zone_by_coordinates(client):
    """Test GPS point-in-polygon lookup for Lahore coordinates."""
    # Mall Road / Central Lahore coordinates
    response = client.get("/api/v1/zones/lookup?lat=31.5204&lon=74.3587")
    assert response.status_code == 200
    data = response.json()
    assert data["zone_id"].startswith("ZONE-LHR-")
    assert "current_conditions" in data


def test_spatial_grid_endpoint(client):
    """Test Kriging spatial interpolation grid endpoint."""
    response = client.get("/api/v1/spatial/grid?variable=aqi_pm25")
    assert response.status_code == 200
    data = response.json()
    assert data["total_zones"] == 241
    assert "ZONE-LHR-0001" in data["zones"]


def test_spatial_geojson_endpoint(client):
    """Test map-ready GeoJSON feature collection."""
    response = client.get("/api/v1/spatial/geojson")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert data["total_features"] == 241
    assert len(data["features"]) == 241

    first_feat = data["features"][0]
    props = first_feat["properties"]
    assert "zone_id" in props
    assert "pm25_current" in props
    assert "forecast_pm25_24h" in props
    assert "heat_island_score" in props
    assert "flood_risk_score" in props


def test_hazards_forecast_endpoint(client):
    """Test 24-hour advance AQI forecast endpoint."""
    response = client.get("/api/v1/hazards/forecast?horizon_hours=24")
    assert response.status_code == 200
    data = response.json()
    assert data["total_zones"] == 241
    assert data["horizon_hours"] == 24


def test_hazards_heat_island_endpoint(client):
    """Test Urban Heat Island endpoint."""
    response = client.get("/api/v1/hazards/heat-island")
    assert response.status_code == 200
    data = response.json()
    assert data["total_zones"] == 241


def test_hazards_flood_endpoint(client):
    """Test Flash Flood risk endpoint."""
    response = client.get("/api/v1/hazards/flood?horizon_hours=24")
    assert response.status_code == 200
    data = response.json()
    assert data["total_zones"] == 241


def test_unified_risk_summary_endpoint(client):
    """Test multi-hazard composite priority ranking."""
    response = client.get("/api/v1/hazards/unified-risk-summary?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["top_priority_zones"]) == 5
    assert data["total_zones_evaluated"] == 241
    top_zone = data["top_priority_zones"][0]
    assert "composite_risk_index" in top_zone
    assert "primary_threat" in top_zone
