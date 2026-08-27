"""
Unit Tests for Module M7: Early Warning Alert & Notification Dispatcher.
Validates multi-hazard threshold evaluation, multi-lingual formatting (English, Urdu, Roman Urdu),
cooldown deduplication, subscription management, and REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.app import create_app

from alerts.models import SeverityLevel, HazardType, AlertChannel, AlertSubscription
from alerts.templates import format_smog_alert, format_flood_alert, format_heat_alert
from alerts.dispatcher import AlertDispatcher
from alerts.interface import evaluate_and_dispatch_alerts, get_alerts_health


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_smog_alert_template_formatting():
    """Verify multi-lingual smog alert templates in English, Urdu (اردو), and Roman Urdu."""
    msgs = format_smog_alert(
        zone_id="ZONE-LHR-0075",
        zone_name="Gulberg / Main Boulevard",
        forecast_pm25=185.4,
        current_pm25=95.0,
        ci_range=[160.0, 210.0],
        severity=SeverityLevel.EMERGENCY,
    )

    assert "en" in msgs and "ur" in msgs and "roman_ur" in msgs
    assert "185.4" in msgs["en"]
    assert "N95" in msgs["en"]
    assert "ایرو کاسٹ ایمرجنسی الرٹ" in msgs["ur"]
    assert "185.4" in msgs["ur"]
    assert "SHADEED SMOG" in msgs["roman_ur"]


def test_flood_alert_template_formatting():
    """Verify multi-lingual flash flood alert templates."""
    msgs = format_flood_alert(
        zone_id="ZONE-LHR-0012",
        zone_name="Lakshmi Chowk / Misri Shah",
        flood_score=0.82,
        precip_mm=65.0,
        inundation_desc="15 - 30 cm street inundation",
        severity=SeverityLevel.EMERGENCY,
    )

    assert "en" in msgs and "ur" in msgs and "roman_ur" in msgs
    assert "0.82" in msgs["en"]
    assert "WASA Emergency Dewatering" in msgs["en"]
    assert "سیلاب الرٹ" in msgs["ur"]
    assert "1334" in msgs["ur"]
    assert "WASA Helpline: 1334" in msgs["roman_ur"]


def test_heat_alert_template_formatting():
    """Verify multi-lingual urban heat island alert templates."""
    msgs = format_heat_alert(
        zone_id="ZONE-LHR-0050",
        zone_name="Shahdara Industrial Core",
        uhi_score=0.74,
        temp_c=41.2,
        severity=SeverityLevel.WATCH,
    )

    assert "en" in msgs and "ur" in msgs and "roman_ur" in msgs
    assert "0.74" in msgs["en"]
    assert "41.2" in msgs["en"]
    assert "ہیٹ ویو الرٹ" in msgs["ur"]
    assert "shiddat-e-hararat" in msgs["roman_ur"]


def test_alert_cooldown_deduplication(tmp_path):
    """Verify that alert evaluation suppresses duplicate alerts within cooldown window."""
    dispatcher = AlertDispatcher(cache_dir=tmp_path)
    
    # First evaluation generates alerts
    alerts_1 = dispatcher.evaluate_all_hazards(force_reevaluate=True)
    assert len(alerts_1) > 0

    # Second immediate evaluation without force flag should be suppressed by cooldown
    alerts_2 = dispatcher.evaluate_all_hazards(force_reevaluate=False)
    assert len(alerts_2) == 0


def test_evaluate_and_dispatch_alerts_facade():
    """Test facade evaluation and dispatching."""
    result = evaluate_and_dispatch_alerts(force_reevaluate=True)
    assert "total_new_alerts_triggered" in result
    assert "total_dispatches_sent" in result
    assert result["total_new_alerts_triggered"] >= 0
    assert result["total_dispatches_sent"] >= 0


def test_alerts_health_diagnostics():
    """Test Module M7 health status."""
    health = get_alerts_health()
    assert health["status"] == "healthy"
    assert health["module"] == "M7_Alert_Dispatcher"
    assert health["cooldown_hours"] == 6
    assert "webhook" in health["supported_channels"]
    assert "ur" in health["supported_languages"]


def test_alerts_rest_api_endpoints(client):
    """Test Module M7 REST API routes (/active, /history, /dispatch, /subscribe)."""
    # 1. Dispatch
    r_disp = client.post("/api/v1/alerts/dispatch?force_reevaluate=true")
    assert r_disp.status_code == 200
    assert "total_new_alerts_triggered" in r_disp.json()

    # 2. Active Alerts
    r_act = client.get("/api/v1/alerts/active")
    assert r_act.status_code == 200
    assert "alerts" in r_act.json()

    # 3. History
    r_hist = client.get("/api/v1/alerts/history")
    assert r_hist.status_code == 200
    assert "registered_subscribers" in r_hist.json()

    # 4. Subscribe
    new_sub = {
        "subscription_id": "SUB-TEST-01",
        "agency_name": "Test Municipal Ward",
        "channel": "webhook",
        "target": "https://test.gov.pk/webhook",
        "subscribed_zones": ["ZONE-LHR-0075"],
        "subscribed_hazards": ["smog", "flash_flood"],
        "min_severity": "WATCH"
    }
    r_sub = client.post("/api/v1/alerts/subscribe", json=new_sub)
    assert r_sub.status_code == 200
    assert r_sub.json()["status"] == "success"
