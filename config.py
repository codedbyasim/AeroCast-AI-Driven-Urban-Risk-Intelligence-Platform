"""
Configuration module for AeroCast Data Ingestion Layer (SRS v1.1).
Loads settings from environment variables or .env file with robust defaults.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Settings:
    """Application settings and ingestion configurations."""

    # Project metadata
    PROJECT_NAME: str = "AeroCast — M1 Data Ingestion"
    SCHEMA_VERSION: str = "1.1"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Lahore Bounding Coordinates (Centroid & BBox for 200-Zone Grid)
    LAHORE_LATITUDE: float = float(os.getenv("LAHORE_LATITUDE", "31.5204"))
    LAHORE_LONGITUDE: float = float(os.getenv("LAHORE_LONGITUDE", "74.3587"))
    LAHORE_BBOX: Tuple[float, float, float, float] = (
        float(os.getenv("LAHORE_BBOX_MIN_LON", "74.00")),
        float(os.getenv("LAHORE_BBOX_MIN_LAT", "31.20")),
        float(os.getenv("LAHORE_BBOX_MAX_LON", "74.65")),
        float(os.getenv("LAHORE_BBOX_MAX_LAT", "31.72")),
    )

    # OpenAQ Configuration
    OPENAQ_API_KEY: str = os.getenv("OPENAQ_API_KEY", "")
    OPENAQ_BASE_URL: str = os.getenv("OPENAQ_BASE_URL", "https://api.openaq.org/v3")
    OPENAQ_POLL_INTERVAL_MINUTES: int = int(os.getenv("OPENAQ_POLL_INTERVAL_MINUTES", "30"))

    # Open-Meteo Configuration
    OPENMETEO_FORECAST_URL: str = os.getenv(
        "OPENMETEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
    )
    OPENMETEO_HISTORICAL_URL: str = os.getenv(
        "OPENMETEO_HISTORICAL_URL", "https://archive-api.open-meteo.com/v1/archive"
    )
    OPENMETEO_POLL_INTERVAL_MINUTES: int = int(os.getenv("OPENMETEO_POLL_INTERVAL_MINUTES", "60"))

    # Geospatial and Raster File Paths
    DATA_DIR: Path = BASE_DIR / "data"
    BOUNDARIES_DIR: Path = DATA_DIR / "boundaries"
    RASTERS_DIR: Path = DATA_DIR / "rasters"

    DISTRICT_BOUNDARY_PATH: Path = BOUNDARIES_DIR / "lahore_exact_osm_district.geojson"
    OSM_HDX_ZONE_GRID_PATH: Path = Path(
        os.getenv("OSM_HDX_ZONE_GRID_PATH", str(BOUNDARIES_DIR / "lahore_zone_grid.geojson"))
    )
    # Legacy alias for backward compatibility
    OSM_HDX_UC_BOUNDARY_PATH: Path = OSM_HDX_ZONE_GRID_PATH

    WORLDPOP_RASTER_PATH: Path = Path(
        os.getenv("WORLDPOP_RASTER_PATH", str(RASTERS_DIR / "worldpop_lahore_pop_density.tif"))
    )
    COPERNICUS_DEM_RASTER_PATH: Path = Path(
        os.getenv("COPERNICUS_DEM_RASTER_PATH", str(RASTERS_DIR / "copernicus_lahore_dem.tif"))
    )
    COPERNICUS_NDVI_RASTER_PATH: Path = Path(
        os.getenv("COPERNICUS_NDVI_RASTER_PATH", str(RASTERS_DIR / "copernicus_lahore_ndvi.tif"))
    )

    # Local Cache Configuration (JSON standardized format)
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", str(BASE_DIR / ".cache")))
    CACHE_FORMAT: str = "json"
    CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))

    # HTTP Client & Tenacity Retry Settings
    HTTP_TIMEOUT_SECONDS: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15.0"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF_MIN_SECONDS: float = float(os.getenv("RETRY_BACKOFF_MIN_SECONDS", "1.0"))
    RETRY_BACKOFF_MAX_SECONDS: float = float(os.getenv("RETRY_BACKOFF_MAX_SECONDS", "10.0"))

    @classmethod
    def ensure_directories(cls):
        """Ensure that data, cache, and subdirectories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
        cls.RASTERS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
