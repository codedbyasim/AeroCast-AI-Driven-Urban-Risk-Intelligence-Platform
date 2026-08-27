"""
Unit Tests for Module M6: Web GIS Situational Command Center Dashboard.
Validates static asset existence, HTML semantic structure, and static file serving via FastAPI.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_files_exist():
    """Verify that all core dashboard frontend assets exist on disk."""
    dashboard_dir = Path("dashboard")
    assert (dashboard_dir / "index.html").exists()
    assert (dashboard_dir / "css" / "styles.css").exists()
    assert (dashboard_dir / "js" / "app.js").exists()


def test_dashboard_served_at_root(client):
    """Verify that accessing root / serves the interactive dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "AeroCast" in response.text
    assert "gis-map" in response.text
    assert "zone-search-input" in response.text


def test_dashboard_static_assets_serving(client):
    """Verify that static CSS and JS are served under /dashboard/ mount."""
    resp_css = client.get("/dashboard/css/styles.css")
    assert resp_css.status_code == 200
    assert "--bg-app" in resp_css.text

    resp_js = client.get("/dashboard/js/app.js")
    assert resp_js.status_code == 200
    assert "initMap" in resp_js.text
