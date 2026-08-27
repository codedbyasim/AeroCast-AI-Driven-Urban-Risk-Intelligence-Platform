"""
Unit and integration test suite for AeroCast AI Service & Copilot API (AIML API / Gemini 2.5 Flash).
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from api.app import create_app
from ai.service import AIAssistantService
from config import settings


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_ai_service_missing_key_raises():
    """Verify that AIAssistantService raises ValueError when API key is empty."""
    service = AIAssistantService()
    service.api_key = ""
    with pytest.raises(ValueError, match="AIML_API_KEY is not configured"):
        service._get_headers()


@pytest.mark.asyncio
async def test_ai_service_call_llm_success():
    """Verify direct AIML API call with mocked response."""
    service = AIAssistantService()
    service.api_key = "test-mock-key"
    service.model = "google/gemini-2.5-flash"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "### Tactical Directives for Zone 75\n- **WASA**: Pre-clear drains\n- **Traffic**: Reroute heavy diesel"
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        messages = [{"role": "user", "content": "Test prompt"}]
        result = await service._call_llm(messages)

        assert "Tactical Directives" in result
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == "google/gemini-2.5-flash"


def test_ai_routes_registered(client):
    """Verify all AI endpoints are exposed in OpenAPI schema."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = schema["paths"]

    assert "/api/v1/ai/zone-mitigation" in paths
    assert "/api/v1/ai/ask" in paths
    assert "/api/v1/ai/simulate-policy" in paths
    assert "/api/v1/ai/situation-report" in paths
    assert "/api/v1/hazards/fires" in paths


def test_ai_ask_endpoint_with_mock(client):
    """Test POST /api/v1/ai/ask with mocked LLM service."""
    with patch("ai.service.ai_service._call_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "Zone 75 and Zone 44 require immediate anti-smog misting."

        payload = {
            "query": "Which zones are top priority?",
            "language": "en"
        }
        resp = client.post("/api/v1/ai/ask", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "Zone 75" in data["data"]["response"]
        assert data["data"]["model_used"] == settings.AIML_MODEL


def test_ai_simulate_policy_endpoint(client):
    """Test POST /api/v1/ai/simulate-policy endpoint."""
    with patch("ai.service.ai_service._call_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "### Simulation: PM2.5 projected to drop by 28% across central Lahore."

        payload = {
            "traffic_reduction_pct": 30.0,
            "heavy_diesel_ban": True,
            "water_cannons_deployed": 10,
            "industrial_clampdown_pct": 50.0,
            "drain_preclearing": True
        }
        resp = client.post("/api/v1/ai/simulate-policy", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "Simulation:" in data["data"]["simulation_report"]


def test_ai_situation_report_endpoint(client):
    """Test GET /api/v1/ai/situation-report endpoint."""
    with patch("ai.service.ai_service._call_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "# DISTRICT ENVIRONMENTAL SITUATION REPORT (DSR)\n- Red Alert active for 18 zones."

        resp = client.get("/api/v1/ai/situation-report?language=en")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "SITUATION REPORT" in data["data"]["situation_report"]


def test_nasa_fires_endpoint(client):
    """Test GET /api/v1/hazards/fires endpoint."""
    resp = client.get("/api/v1/hazards/fires?days=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "fire_data" in data
