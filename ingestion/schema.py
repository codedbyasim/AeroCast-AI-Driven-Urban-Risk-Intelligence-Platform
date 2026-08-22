"""
Data schemas for AeroCast M1 Data Ingestion Layer (SRS v1.1 Compliant).
Validates normalized records and raw ingestion payloads against strict Pydantic v2 models.
Keyed by canonical Zone ID (ZONE-LHR-####).
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Metrics(BaseModel):
    """Environmental and atmospheric metrics for a Lahore Zone."""
    model_config = ConfigDict(extra="ignore")

    aqi_pm25: Optional[float] = Field(None, description="PM2.5 particulate matter (µg/m³ or AQI sub-index)")
    aqi_pm10: Optional[float] = Field(None, description="PM10 particulate matter (µg/m³ or AQI sub-index)")
    no2_ppb: Optional[float] = Field(None, description="Nitrogen dioxide concentration (ppb or µg/m³)")
    temperature_c: Optional[float] = Field(None, description="Air temperature in Celsius")
    rainfall_mm_forecast: Optional[float] = Field(None, description="Forecasted rainfall in millimeters")
    wind_speed_kmh: Optional[float] = Field(None, description="Wind speed in km/h")
    relative_humidity_percent: Optional[float] = Field(None, description="Relative humidity percentage")
    surface_pressure_hpa: Optional[float] = Field(None, description="Surface atmospheric pressure in hPa")


class SpatialContext(BaseModel):
    """Geographic, topographic, and land cover context for a Lahore Zone."""
    model_config = ConfigDict(extra="ignore")

    elevation_m: Optional[float] = Field(None, description="Mean elevation in meters above sea level (Copernicus DEM)")
    slope_percent: Optional[float] = Field(None, description="Mean topographic slope in percentage")
    impervious_surface_ratio: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Ratio of built-up/paved surface area (OSM roads & buildings)"
    )
    ndvi_index: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Normalized Difference Vegetation Index from Copernicus Sentinel-2"
    )
    population_density_per_sqkm: Optional[float] = Field(
        None, ge=0.0, description="Estimated population density per km² (WorldPop raster)"
    )
    grid_row: Optional[int] = Field(None, description="Grid row index (1-based) in uniform zone grid")
    grid_col: Optional[int] = Field(None, description="Grid col index (1-based) in uniform zone grid")
    zone_name: Optional[str] = Field(None, description="Descriptive or locality-based name of the Zone")
    centroid_lat: Optional[float] = Field(None, description="Latitude centroid of the Zone (WGS84)")
    centroid_lon: Optional[float] = Field(None, description="Longitude centroid of the Zone (WGS84)")


class DataQuality(BaseModel):
    """Metadata regarding observation freshness, interpolation, and confidence score."""
    model_config = ConfigDict(extra="ignore")

    interpolated: bool = Field(False, description="Whether this record was filled via spatial interpolation")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score of data [0.0 - 1.0]")
    stale: bool = Field(False, description="Flag indicating cached data served due to upstream API outage")
    notes: Optional[str] = Field(None, description="Optional diagnostic notes or fallback reason (e.g. synthetic fallback)")


class NormalizedRecord(BaseModel):
    """
    Standardized internal JSON schema representing environmental, spatial,
    and weather data for a single 9 sq km Zone in Lahore (SRS v1.1 Section 6.1).
    """
    model_config = ConfigDict(
        populate_by_name=True,
    )

    schema_version: str = Field(default="1.1", description="Schema version identifier")
    source: str = Field(..., description="Data source(s) identifier e.g. OpenAQ, Open-Meteo, PITB, WASA, EPA, Merged")
    zone_id: str = Field(..., description="Canonical Zone identifier e.g. ZONE-LHR-0142")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the observation / record creation"
    )
    metrics: Metrics = Field(default_factory=Metrics, description="Atmospheric and weather metrics")
    spatial_context: SpatialContext = Field(default_factory=SpatialContext, description="Topographic and land cover context")
    data_quality: DataQuality = Field(default_factory=DataQuality, description="Data provenance and quality flags")

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        """Parse string or datetime and ensure UTC timezone."""
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Output clean dictionary matching the SRS v1.1 JSON schema specification."""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "zone_id": self.zone_id,
            "timestamp_utc": self.timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in self.metrics.model_dump().items() if v is not None
            },
            "spatial_context": {
                k: round(v, 4) if (isinstance(v, float) and "lat" in k or "lon" in k)
                else (round(v, 2) if isinstance(v, float) else v)
                for k, v in self.spatial_context.model_dump().items() if v is not None
            },
            "data_quality": self.data_quality.model_dump(),
        }
