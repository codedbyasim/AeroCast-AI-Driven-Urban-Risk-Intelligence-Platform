"""
Unit Tests for Module M3 Urban Heat Island (UHI) Risk Scoring Engine.
"""

import pytest
from ml.heat_island import HeatIslandScorer
from ml.interface import get_heat_island_risk, get_all_heat_island_risk


@pytest.fixture
def uhi_scorer():
    return HeatIslandScorer()


def test_heat_island_scoring_bounds(uhi_scorer):
    """Test that UHI risk score is bounded within [0.0, 1.0]."""
    res = uhi_scorer.score_zone("ZONE-LHR-0075", temperature_c=38.0)
    score = res["uhi_risk_score"]
    assert 0.0 <= score <= 1.0
    assert "risk_category" in res
    assert "confidence_score" in res


def test_heat_island_temperature_sensitivity(uhi_scorer):
    """Test that higher temperature increases UHI risk score."""
    cool = uhi_scorer.score_zone("ZONE-LHR-0075", temperature_c=26.0)
    hot = uhi_scorer.score_zone("ZONE-LHR-0075", temperature_c=44.0)

    assert hot["uhi_risk_score"] > cool["uhi_risk_score"]


def test_heat_island_vegetation_cooling_effect(uhi_scorer):
    """Test that higher NDVI vegetation index reduces UHI thermal risk."""
    # Temporarily modify covariate to test NDVI impact
    uhi_scorer._covariates["ZONE-TEST-GREEN"] = {
        "impervious_surface_ratio": 0.5,
        "population_density_per_sqkm": 10000.0,
        "ndvi_index": 0.85,  # High dense green canopy
    }
    uhi_scorer._covariates["ZONE-TEST-CONCRETE"] = {
        "impervious_surface_ratio": 0.5,
        "population_density_per_sqkm": 10000.0,
        "ndvi_index": 0.05,  # Bare concrete / zero greenery
    }

    green_res = uhi_scorer.score_zone("ZONE-TEST-GREEN", temperature_c=35.0)
    concrete_res = uhi_scorer.score_zone("ZONE-TEST-CONCRETE", temperature_c=35.0)

    assert green_res["uhi_risk_score"] < concrete_res["uhi_risk_score"]


def test_heat_island_all_zones():
    """Test batch UHI scoring across all 241 zones."""
    scores = get_all_heat_island_risk(allow_cache=True)
    assert len(scores) == 241
    assert "ZONE-LHR-0001" in scores
    assert "ZONE-LHR-0075" in scores
