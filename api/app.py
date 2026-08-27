"""
AeroCast FastAPI Application Factory.
Configures CORS middleware, OpenAPI metadata, routers, and lifespan events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.routes.health import router as health_router
from api.routes.zones import router as zones_router
from api.routes.spatial import router as spatial_router
from api.routes.hazards import router as hazards_router
from api.routes.alerts import router as alerts_router
from api.routes.backtest import router as backtest_router
from api.routes.ai import router as ai_router

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

logger = logging.getLogger("aerocast.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown logging."""
    logger.info("==================================================")
    logger.info("  AeroCast REST API & Web GIS Dashboard Initialized")
    logger.info("  Covering 241 Canonical Zones in Lahore District")
    logger.info("==================================================")
    yield
    logger.info("AeroCast REST API Service Shutting Down.")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="AeroCast — Urban Risk Intelligence API",
        description=(
            "Predictive environmental risk intelligence platform for Lahore District. "
            "Exposes spatial Universal Kriging, 24-hour advance AQI forecasting (M3), "
            "Urban Heat Island anomaly scoring, deterministic Flash Flood risk modeling (M4), "
            "and AI Copilot urban mitigation intelligence (Gemini 2.5 Flash via AIML API) "
            "across the complete 241-zone metric grid."
        ),
        version="1.2.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Allow broad CORS for dashboard and third-party frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(health_router)
    app.include_router(zones_router)
    app.include_router(spatial_router)
    app.include_router(hazards_router)
    app.include_router(alerts_router)
    app.include_router(backtest_router)
    app.include_router(ai_router)

    # Mount Dashboard Static Files
    dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

    @app.get("/api", include_in_schema=False)
    def api_info():
        return {
            "name": "AeroCast Urban Risk Intelligence API",
            "version": "1.1.0",
            "status": "online",
            "documentation": "/docs",
            "canonical_zones": 241,
        }

    @app.get("/", include_in_schema=False)
    def root():
        index_file = dashboard_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return api_info()

    return app


app = create_app()
