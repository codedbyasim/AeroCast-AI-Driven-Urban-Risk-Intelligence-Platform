"""
Data Normalization and Spatial Mapping Layer for AeroCast (SRS v1.1 Compliant).
Normalizes raw payloads from OpenAQ, Open-Meteo, OSM, WorldPop, and Copernicus
into the unified Pydantic v2 NormalizedRecord schema keyed by Zone ID (ZONE-LHR-####).
Properly handles fallback data quality and confidence scoring per FR-INGEST-09 (SRS Risk R-01).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from .schema import NormalizedRecord, Metrics, SpatialContext, DataQuality
from .osm_hdx_client import OSMHDXClient

logger = logging.getLogger("aerocast.normalizer")


class DataNormalizer:
    """Normalizes heterogeneous external data sources into standardized Zone-level schemas."""

    def __init__(self, osm_client: Optional[OSMHDXClient] = None):
        self.osm_client = osm_client or OSMHDXClient()

    def normalize_openaq_record(
        self,
        raw_station_data: Dict[str, Any],
        spatial_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[NormalizedRecord]:
        """
        Convert a raw OpenAQ station measurement record into a NormalizedRecord.
        Performs spatial lookup to assign the Zone ID.
        """
        lat = raw_station_data.get("latitude")
        lon = raw_station_data.get("longitude")

        if lat is None or lon is None:
            logger.warning("Rejecting OpenAQ record missing coordinates: %s", raw_station_data)
            return None

        # Map station coordinates to nearest/containing Zone
        zone_info = self.osm_client.find_zone_by_coordinates(lat, lon)
        if not zone_info:
            logger.warning("Could not map OpenAQ coordinates (%f, %f) to any Zone", lat, lon)
            return None

        zone_id = zone_info["zone_id"]

        # Parse timestamp
        raw_ts = raw_station_data.get("timestamp_utc")
        if isinstance(raw_ts, str):
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
        elif isinstance(raw_ts, datetime):
            dt = raw_ts.astimezone(timezone.utc) if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        # Build metrics
        metrics = Metrics(
            aqi_pm25=raw_station_data.get("pm25"),
            aqi_pm10=raw_station_data.get("pm10"),
            no2_ppb=raw_station_data.get("no2"),
        )

        # Build spatial context
        sp_ctx_dict = spatial_context or {}
        spatial = SpatialContext(
            zone_name=sp_ctx_dict.get("zone_name", zone_info.get("zone_name")),
            grid_row=sp_ctx_dict.get("grid_row", zone_info.get("grid_row")),
            grid_col=sp_ctx_dict.get("grid_col", zone_info.get("grid_col")),
            centroid_lat=sp_ctx_dict.get("centroid_lat", zone_info.get("centroid_lat")),
            centroid_lon=sp_ctx_dict.get("centroid_lon", zone_info.get("centroid_lon")),
            impervious_surface_ratio=sp_ctx_dict.get(
                "impervious_surface_ratio", zone_info.get("impervious_surface_ratio", 0.65)
            ),
            elevation_m=sp_ctx_dict.get("elevation_m"),
            slope_percent=sp_ctx_dict.get("slope_percent"),
            ndvi_index=sp_ctx_dict.get("ndvi_index"),
            population_density_per_sqkm=sp_ctx_dict.get("population_density_per_sqkm"),
        )

        # Determine data quality flags (FR-INGEST-09: Fallback records must self-flag)
        is_fallback = raw_station_data.get("is_fallback", False)
        fallback_reason = raw_station_data.get("fallback_reason")

        data_quality = DataQuality(
            interpolated=is_fallback,
            confidence_score=0.50 if is_fallback else 1.0,
            stale=False,
            notes=fallback_reason if is_fallback else None,
        )

        source_label = raw_station_data.get("source", "OpenAQ")

        try:
            record = NormalizedRecord(
                schema_version="1.1",
                source=source_label,
                zone_id=zone_id,
                timestamp_utc=dt,
                metrics=metrics,
                spatial_context=spatial,
                data_quality=data_quality,
            )
            return record
        except ValidationError as e:
            logger.error("Validation failed for OpenAQ record %s: %s", zone_id, e)
            return None

    def normalize_openmeteo_record(
        self,
        raw_weather: Dict[str, Any],
        zone_id: Optional[str] = None,
        spatial_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[NormalizedRecord]:
        """
        Convert a raw Open-Meteo weather record into a NormalizedRecord.
        """
        lat = raw_weather.get("latitude")
        lon = raw_weather.get("longitude")

        if zone_id:
            z_id = zone_id
            zone_info = self.osm_client.find_zone_by_coordinates(lat or settings.LAHORE_LATITUDE, lon or settings.LAHORE_LONGITUDE) or {"zone_id": z_id}
        elif lat is not None and lon is not None:
            zone_info = self.osm_client.find_zone_by_coordinates(lat, lon)
            if not zone_info:
                logger.warning("Could not map Open-Meteo coordinates (%f, %f) to any Zone", lat, lon)
                return None
            z_id = zone_info["zone_id"]
        else:
            logger.warning("Rejecting Open-Meteo record without coordinates or Zone ID")
            return None

        # Parse timestamp
        raw_ts = raw_weather.get("timestamp_utc")
        if isinstance(raw_ts, str):
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        metrics = Metrics(
            temperature_c=raw_weather.get("temperature_c"),
            rainfall_mm_forecast=raw_weather.get("rainfall_mm_forecast"),
            wind_speed_kmh=raw_weather.get("wind_speed_kmh"),
            relative_humidity_percent=raw_weather.get("relative_humidity_percent"),
            surface_pressure_hpa=raw_weather.get("surface_pressure_hpa"),
        )

        sp_ctx_dict = spatial_context or {}
        spatial = SpatialContext(
            elevation_m=sp_ctx_dict.get("elevation_m", raw_weather.get("elevation", 214.0)),
            slope_percent=sp_ctx_dict.get("slope_percent"),
            impervious_surface_ratio=sp_ctx_dict.get(
                "impervious_surface_ratio", zone_info.get("impervious_surface_ratio")
            ),
            ndvi_index=sp_ctx_dict.get("ndvi_index"),
            population_density_per_sqkm=sp_ctx_dict.get("population_density_per_sqkm"),
            zone_name=sp_ctx_dict.get("zone_name", zone_info.get("zone_name")),
            grid_row=sp_ctx_dict.get("grid_row", zone_info.get("grid_row")),
            grid_col=sp_ctx_dict.get("grid_col", zone_info.get("grid_col")),
            centroid_lat=sp_ctx_dict.get("centroid_lat", zone_info.get("centroid_lat")),
            centroid_lon=sp_ctx_dict.get("centroid_lon", zone_info.get("centroid_lon")),
        )

        # Fallback quality tracking
        is_fallback = raw_weather.get("is_fallback", False)
        fallback_reason = raw_weather.get("fallback_reason")

        data_quality = DataQuality(
            interpolated=is_fallback,
            confidence_score=0.50 if is_fallback else 1.0,
            stale=False,
            notes=fallback_reason if is_fallback else None,
        )

        source_label = raw_weather.get("source", "Open-Meteo")

        try:
            record = NormalizedRecord(
                schema_version="1.1",
                source=source_label,
                zone_id=z_id,
                timestamp_utc=dt,
                metrics=metrics,
                spatial_context=spatial,
                data_quality=data_quality,
            )
            return record
        except ValidationError as e:
            logger.error("Validation failed for Open-Meteo record %s: %s", z_id, e)
            return None

    def merge_zone_snapshot(
        self,
        zone_id: str,
        aq_record: Optional[NormalizedRecord] = None,
        weather_record: Optional[NormalizedRecord] = None,
        spatial_context: Optional[Dict[str, Any]] = None,
        timestamp_utc: Optional[datetime] = None,
        is_interpolated: bool = False,
        confidence: float = 1.0,
        is_stale: bool = False,
        notes: Optional[str] = None,
    ) -> Optional[NormalizedRecord]:
        """
        Merge multiple source records into a single unified canonical Zone record
        for downstream spatial interpolation (M2) and risk intelligence models (M3, M4).
        """
        ts = timestamp_utc or datetime.now(timezone.utc)
        sources = []
        notes_list = []

        combined_metrics = Metrics()
        combined_spatial = SpatialContext()

        if aq_record:
            sources.append(aq_record.source)
            combined_metrics.aqi_pm25 = aq_record.metrics.aqi_pm25
            combined_metrics.aqi_pm10 = aq_record.metrics.aqi_pm10
            combined_metrics.no2_ppb = aq_record.metrics.no2_ppb
            if aq_record.spatial_context:
                combined_spatial = aq_record.spatial_context.model_copy()
            if aq_record.data_quality.notes:
                notes_list.append(aq_record.data_quality.notes)
            if aq_record.data_quality.confidence_score < confidence:
                confidence = aq_record.data_quality.confidence_score

        if weather_record:
            if weather_record.source not in sources:
                sources.append(weather_record.source)
            combined_metrics.temperature_c = weather_record.metrics.temperature_c
            combined_metrics.rainfall_mm_forecast = weather_record.metrics.rainfall_mm_forecast
            combined_metrics.wind_speed_kmh = weather_record.metrics.wind_speed_kmh
            combined_metrics.relative_humidity_percent = weather_record.metrics.relative_humidity_percent
            combined_metrics.surface_pressure_hpa = weather_record.metrics.surface_pressure_hpa
            if weather_record.data_quality.notes:
                notes_list.append(weather_record.data_quality.notes)
            if weather_record.data_quality.confidence_score < confidence:
                confidence = weather_record.data_quality.confidence_score

        if notes:
            notes_list.append(notes)

        # Overlay static spatial context
        if spatial_context:
            for k, v in spatial_context.items():
                if hasattr(combined_spatial, k) and v is not None:
                    setattr(combined_spatial, k, v)

        source_label = " | ".join(sources) if sources else "AeroCast-Aggregated"
        final_notes = "; ".join(notes_list) if notes_list else None

        try:
            record = NormalizedRecord(
                schema_version="1.1",
                source=source_label,
                zone_id=zone_id,
                timestamp_utc=ts,
                metrics=combined_metrics,
                spatial_context=combined_spatial,
                data_quality=DataQuality(
                    interpolated=is_interpolated,
                    confidence_score=round(confidence, 2),
                    stale=is_stale,
                    notes=final_notes,
                ),
            )
            return record
        except ValidationError as e:
            logger.error("Validation failed during Zone snapshot merge for %s: %s", zone_id, e)
            return None

    # Backwards compatibility alias
    def merge_uc_snapshot(self, union_council_id: str, **kwargs) -> Optional[NormalizedRecord]:
        return self.merge_zone_snapshot(zone_id=union_council_id, **kwargs)
