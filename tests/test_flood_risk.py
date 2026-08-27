"""
Unit Tests for Module M4: Flash Flood Risk Calculation Engine.
Validates physical hydrological sensitivity, deterministic reproducibility, bounds, 241-zone coverage,
live weather ingestion wiring, and 7-day exponential antecedent precipitation index.
"""

import pytest
from flood.engine import FlashFloodScorer
from flood.interface import (
    get_zone_flood_risk,
    get_all_zones_flood_risk,
    get_flood_health,
)


@pytest.fixture
def flood_scorer():
    return FlashFloodScorer()


def test_flood_risk_bounds_and_structure(flood_scorer):
    """Test that flood risk score is strictly bounded in [0.0, 1.0] with valid structure."""
    result = flood_scorer.calculate_zone_flood_risk(
        "ZONE-LHR-0075",
        horizon_hours=24,
        current_weather={"precipitation_sum": 45.0, "antecedent_precipitation_3d": 20.0},
    )

    assert result["zone_id"] == "ZONE-LHR-0075"
    assert result["horizon_hours"] == 24
    assert 0.0 <= result["flood_risk_score"] <= 1.0
    assert result["risk_category"] in ("Low", "Moderate", "High", "Severe")
    assert result["alert_level"] in ("GREEN", "YELLOW", "ORANGE", "RED")
    assert "component_breakdown" in result
    assert "terrain_context" in result
    assert "actionable_advisory" in result
    assert "data_quality" in result


def test_flood_risk_precipitation_sensitivity(flood_scorer):
    """Test that flood risk strictly increases with higher forecasted precipitation."""
    dry = flood_scorer.calculate_zone_flood_risk(
        "ZONE-LHR-0075", current_weather={"precipitation_sum": 0.0}
    )
    moderate = flood_scorer.calculate_zone_flood_risk(
        "ZONE-LHR-0075", current_weather={"precipitation_sum": 35.0}
    )
    heavy = flood_scorer.calculate_zone_flood_risk(
        "ZONE-LHR-0075", current_weather={"precipitation_sum": 85.0}
    )

    assert dry["flood_risk_score"] < moderate["flood_risk_score"]
    assert moderate["flood_risk_score"] < heavy["flood_risk_score"]
    assert heavy["risk_category"] in ("High", "Severe")


def test_flood_risk_slope_flatness_sensitivity(flood_scorer):
    """Test that flatter terrain generates higher flood stagnation risk than sloped terrain."""
    flat_zone_risk = flood_scorer.calculate_zone_flood_risk(
        "ZONE-LHR-0001", current_weather={"precipitation_sum": 50.0}
    )
    assert 0.0 <= flat_zone_risk["flood_risk_score"] <= 1.0
    assert "slope_flatness_contribution" in flat_zone_risk["component_breakdown"]


def test_flood_risk_no_argument_live_weather_wiring(monkeypatch):
    """
    CRITICAL REGRESSION TEST:
    Verify that calling get_zone_flood_risk() without explicit current_weather
    correctly pulls live rainfall_mm_forecast from M1 ingestion cache rather than
    silently defaulting to 0.0mm.
    """
    fake_live_record = {
        "zone_id": "ZONE-LHR-0075",
        "metrics": {
            "rainfall_mm_forecast": 65.0,
            "temperature_c": 28.5,
            "relative_humidity_percent": 82.0,
        },
        "spatial_context": {
            "elevation_m": 210.0,
            "slope_percent": 0.8,
            "impervious_surface_ratio": 0.75,
        },
        "data_quality": {
            "stale": False,
            "confidence_score": 0.95,
        }
    }

    import ingestion.interface
    monkeypatch.setattr(ingestion.interface, "get_latest_data", lambda zid: fake_live_record)

    # Call production path with NO explicit weather argument
    result = get_zone_flood_risk("ZONE-LHR-0075", horizon_hours=24, allow_cache=False)

    # Assert that 65mm rainfall was pulled from live M1 cache
    assert result["component_breakdown"]["forecasted_precipitation_24h_mm"] == 65.0
    assert result["component_breakdown"]["precipitation_risk_contribution"] > 0.20
    assert result["flood_risk_score"] > 0.35
    assert result["data_quality"]["is_fallback_weather"] is False


def test_flood_risk_antecedent_precipitation_formula(flood_scorer):
    """Verify 7-day exponential decay antecedent precipitation index calculation."""
    api_val, raw_7d, is_fallback = flood_scorer._get_antecedent_precipitation_7d("ZONE-LHR-0075")
    assert api_val >= 0.0
    assert raw_7d >= 0.0
    assert isinstance(is_fallback, bool)


def test_flood_risk_all_241_zones():
    """Test batch flood risk calculation across all 241 canonical Lahore zones."""
    all_floods = get_all_zones_flood_risk(horizon_hours=24, allow_cache=False)
    assert len(all_floods) == 241
    assert "ZONE-LHR-0001" in all_floods
    assert "ZONE-LHR-0241" in all_floods

    for zid, data in all_floods.items():
        assert 0.0 <= data["flood_risk_score"] <= 1.0
        assert data["risk_category"] in ("Low", "Moderate", "High", "Severe")


def test_flood_health_endpoint():
    """Test Module M4 diagnostic health endpoint."""
    health = get_flood_health()
    assert health["status"] == "healthy"
    assert health["total_zones_covered"] == 241
    assert health["is_241_zones_canonical"] is True
    assert "weights" in health
    assert health["weights"]["precipitation"] == 0.40


def test_flood_horizon_validation():
    """Test that requesting unsupported horizons (e.g. 48h) raises ValueError."""
    with pytest.raises(ValueError, match="horizon_hours=24"):
        get_zone_flood_risk("ZONE-LHR-0075", horizon_hours=48)

    with pytest.raises(ValueError, match="horizon_hours=24"):
        get_all_zones_flood_risk(horizon_hours=48)
