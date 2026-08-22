"""
Scheduler Module for AeroCast M1 Data Ingestion Layer (SRS v1.1 Compliant).
Manages periodic polling of dynamic sources (OpenAQ, Open-Meteo) across the 200-Zone grid,
startup initialization of static raster datasets (OSM, WorldPop, Copernicus),
spatial nearest-neighbor weather matching, and historical data fetching for ML training.
Properly flags synthetic raster provenance and interpolation diagnostics per FR-INGEST-09 (SRS Risk R-01/R-05).
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import settings
from .schema import NormalizedRecord
from .openaq_client import OpenAQClient
from .openmeteo_client import OpenMeteoClient
from .osm_hdx_client import OSMHDXClient
from .worldpop_client import WorldPopClient
from .copernicus_client import CopernicusClient
from .normalizer import DataNormalizer
from .cache import LocalDataCache

logger = logging.getLogger("aerocast.scheduler")


class IngestionScheduler:
    """Orchestrates periodic and static data ingestion pipelines across the 200-Zone grid."""

    def __init__(
        self,
        openaq_client: Optional[OpenAQClient] = None,
        openmeteo_client: Optional[OpenMeteoClient] = None,
        osm_client: Optional[OSMHDXClient] = None,
        worldpop_client: Optional[WorldPopClient] = None,
        copernicus_client: Optional[CopernicusClient] = None,
        normalizer: Optional[DataNormalizer] = None,
        cache: Optional[LocalDataCache] = None,
    ):
        self.osm_client = osm_client or OSMHDXClient()
        self.openaq_client = openaq_client or OpenAQClient()
        self.openmeteo_client = openmeteo_client or OpenMeteoClient()
        self.worldpop_client = worldpop_client or WorldPopClient()
        self.copernicus_client = copernicus_client or CopernicusClient()
        self.normalizer = normalizer or DataNormalizer(osm_client=self.osm_client)
        self.cache = cache or LocalDataCache()

        self._scheduler = AsyncIOScheduler()
        self._static_context: Dict[str, Dict[str, Any]] = {}
        self._is_initialized = False
        self._is_synthetic_raster = False

    async def initialize_static_sources(self):
        """
        Load 200-Zone grid and compute baseline spatial and raster features
        (elevation, slope, NDVI, population density, impervious ratio) for all Zones.
        Tracks whether static raster sources are authentic or synthetic placeholders (FR-INGEST-09 / R-05).
        """
        logger.info("Initializing static datasets (200-Zone Grid, WorldPop, Copernicus DEM & NDVI)...")
        try:
            zone_gdf = self.osm_client.load_zone_grid()
            total_zones = len(zone_gdf)
            logger.info("Loaded %d Zones for Lahore district", total_zones)

            # 1. WorldPop population density & synthetic check
            pop_densities = self.worldpop_client.compute_all_zone_densities(zone_gdf)
            is_synthetic_worldpop = self.worldpop_client.is_synthetic_raster()

            # 2. Copernicus DEM & NDVI & synthetic check
            terrain_metrics = self.copernicus_client.compute_all_zone_terrain(zone_gdf)
            is_synthetic_copernicus = self.copernicus_client.is_synthetic_raster()

            self._is_synthetic_raster = bool(is_synthetic_worldpop or is_synthetic_copernicus)
            if self._is_synthetic_raster:
                logger.info("Static spatial context contains synthetic placeholder raster data (WorldPop/Copernicus)")

            # 3. Consolidate static spatial context
            self._static_context.clear()
            for _, row in zone_gdf.iterrows():
                z_id = str(row.get("zone_id", row.get("id", "ZONE-LHR-0001")))
                terrain = terrain_metrics.get(z_id, {})
                pop = pop_densities.get(z_id, 12500.0)

                self._static_context[z_id] = {
                    "zone_name": str(row.get("zone_name", f"Zone {z_id}")),
                    "grid_row": int(row.get("grid_row", 1)),
                    "grid_col": int(row.get("grid_col", 1)),
                    "area_sqkm": float(row.get("area_sqkm", 9.0)),
                    "district": str(row.get("district", "Lahore District")),
                    "centroid_lat": float(row.get("centroid_lat", row.geometry.centroid.y)),
                    "centroid_lon": float(row.get("centroid_lon", row.geometry.centroid.x)),
                    "impervious_surface_ratio": float(row.get("impervious_surface_ratio", 0.65)),
                    "population_density_per_sqkm": pop,
                    "elevation_m": terrain.get("elevation_m", 214.0),
                    "slope_percent": terrain.get("slope_percent", 2.5),
                    "ndvi_index": terrain.get("ndvi_index", 0.22),
                }

            self._is_initialized = True
            logger.info("Successfully initialized static spatial context for %d Zones", len(self._static_context))
        except Exception as e:
            logger.error("Error initializing static spatial sources: %s", e, exc_info=True)

    async def poll_openaq_job(self) -> List[NormalizedRecord]:
        """
        Periodic job: Fetch latest air quality data from OpenAQ, normalize, and cache.
        """
        logger.info("Executing OpenAQ scheduled polling job...")
        records: List[NormalizedRecord] = []
        try:
            raw_measurements = await self.openaq_client.fetch_latest_measurements()
            for raw in raw_measurements:
                lat = raw.get("latitude")
                lon = raw.get("longitude")
                zone_info = self.osm_client.find_zone_by_coordinates(lat, lon) if (lat and lon) else None
                z_id = zone_info.get("zone_id") if zone_info else None

                sp_ctx = self._static_context.get(z_id, {}) if z_id else {}
                norm_rec = self.normalizer.normalize_openaq_record(raw, spatial_context=sp_ctx)
                if norm_rec:
                    self.cache.save_record(norm_rec)
                    records.append(norm_rec)
            logger.info("OpenAQ polling completed. Processed %d records.", len(records))
        except Exception as e:
            logger.error("OpenAQ scheduled poll failed: %s", e, exc_info=True)
        return records

    async def poll_openmeteo_job(self) -> List[NormalizedRecord]:
        """
        Periodic job: Fetch latest weather forecast from Open-Meteo, normalize, and cache.
        """
        logger.info("Executing Open-Meteo scheduled polling job...")
        records: List[NormalizedRecord] = []
        try:
            grid_weather = await self.openmeteo_client.fetch_grid_weather()
            for raw in grid_weather:
                lat = raw.get("latitude")
                lon = raw.get("longitude")
                zone_info = self.osm_client.find_zone_by_coordinates(lat, lon) if (lat and lon) else None
                z_id = zone_info.get("zone_id") if zone_info else None

                sp_ctx = self._static_context.get(z_id, {}) if z_id else {}
                norm_rec = self.normalizer.normalize_openmeteo_record(raw, zone_id=z_id, spatial_context=sp_ctx)
                if norm_rec:
                    self.cache.save_record(norm_rec)
                    records.append(norm_rec)
            logger.info("Open-Meteo polling completed. Processed %d records.", len(records))
        except Exception as e:
            logger.error("Open-Meteo scheduled poll failed: %s", e, exc_info=True)
        return records

    async def trigger_full_sync(self) -> Dict[str, Any]:
        """
        Execute a complete end-to-end synchronization across all sources:
        1. Refresh static rasters & zone boundaries
        2. Ingest OpenAQ & Open-Meteo
        3. Match nearest weather observations geographically per zone (Issue 1)
        4. Provide explicit diagnostic notes for interpolated & synthetic data (Issues 2, 4)
        5. Consolidate and populate all 200+ Lahore Zones into cache.
        """
        logger.info("Triggering full ingestion and consolidation sweep...")
        if not self._is_initialized:
            await self.initialize_static_sources()

        # Run dynamic polls concurrently
        aq_task = asyncio.create_task(self.poll_openaq_job())
        meteo_task = asyncio.create_task(self.poll_openmeteo_job())
        aq_records, meteo_records = await asyncio.gather(aq_task, meteo_task, return_exceptions=False)

        # Index dynamic records by Zone ID
        aq_by_zone = {r.zone_id: r for r in aq_records}
        meteo_by_zone = {r.zone_id: r for r in meteo_records}

        consolidated_count = 0
        now_ts = datetime.now(timezone.utc)

        # Merge and cache all known Zones
        for z_id, sp_ctx in self._static_context.items():
            aq_rec = aq_by_zone.get(z_id)
            meteo_rec = meteo_by_zone.get(z_id)
            weather_is_nearest = False
            nearest_grid_name = None

            # Issue 1: Geographic nearest-neighbor weather matching when no direct zone match
            if meteo_rec is None and meteo_records:
                z_lat = sp_ctx["centroid_lat"]
                z_lon = sp_ctx["centroid_lon"]

                best_dist = float("inf")
                best_meteo = meteo_records[0]

                for m_rec in meteo_records:
                    m_lat = (
                        m_rec.spatial_context.centroid_lat
                        if m_rec.spatial_context and m_rec.spatial_context.centroid_lat is not None
                        else settings.LAHORE_LATITUDE
                    )
                    m_lon = (
                        m_rec.spatial_context.centroid_lon
                        if m_rec.spatial_context and m_rec.spatial_context.centroid_lon is not None
                        else settings.LAHORE_LONGITUDE
                    )
                    dist_sq = (m_lat - z_lat) ** 2 + (m_lon - z_lon) ** 2
                    if dist_sq < best_dist:
                        best_dist = dist_sq
                        best_meteo = m_rec

                meteo_rec = best_meteo
                weather_is_nearest = True
                nearest_grid_name = (
                    best_meteo.spatial_context.zone_name
                    if best_meteo.spatial_context and best_meteo.spatial_context.zone_name
                    else best_meteo.zone_id
                )

            # Issue 2 & 4: Construct explicit diagnostic notes and calculate confidence score
            is_interpolated = (aq_rec is None)
            zone_notes_list = []

            if is_interpolated:
                zone_notes_list.append(
                    "no direct AQI station in this zone — using nearest-zone weather context only, AQI pending M2 spatial interpolation"
                )

            if weather_is_nearest and nearest_grid_name:
                zone_notes_list.append(
                    f"no direct weather grid point in this zone — using nearest grid point ({nearest_grid_name})"
                )

            if self._is_synthetic_raster:
                zone_notes_list.append(
                    "spatial context (elevation/slope/NDVI/population) from synthetic placeholder rasters — real WorldPop/Copernicus data not yet sourced"
                )

            # Calculate confidence score
            if aq_rec is not None:
                confidence = 1.0
            else:
                confidence = 0.85

            if weather_is_nearest:
                confidence = min(confidence, 0.85)

            # If spatial context is synthetic, factor into confidence score
            if self._is_synthetic_raster:
                confidence = min(confidence, 0.70)

            zone_notes_str = "; ".join(zone_notes_list) if zone_notes_list else None

            merged_rec = self.normalizer.merge_zone_snapshot(
                zone_id=z_id,
                aq_record=aq_rec,
                weather_record=meteo_rec,
                spatial_context=sp_ctx,
                timestamp_utc=now_ts,
                is_interpolated=is_interpolated,
                confidence=confidence,
                is_stale=False,
                notes=zone_notes_str,
            )

            if merged_rec:
                self.cache.save_record(merged_rec)
                consolidated_count += 1

        logger.info("Full sync completed. %d Zones updated in cache.", consolidated_count)
        return {
            "status": "success",
            "timestamp_utc": now_ts.isoformat(),
            "zones_synced": consolidated_count,
            "openaq_records": len(aq_records),
            "openmeteo_records": len(meteo_records),
        }

    async def fetch_historical_dataset(self, days: int = 730) -> Dict[str, Any]:
        """
        Fetch multi-year historical dataset for weather and AQI (FR-INGEST-02 / FR-INGEST-10).
        Persists historical logs under .cache/historical/.
        """
        logger.info("Fetching 2-year historical environmental datasets (days=%d)...", days)
        hist_aq = await self.openaq_client.fetch_historical_measurements(days=days)
        hist_weather = await self.openmeteo_client.fetch_historical_weather()

        hist_dir = self.cache.cache_dir / "historical"
        hist_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(hist_dir / "historical_aqi.json", "w", encoding="utf-8") as f:
            json.dump(hist_aq, f, indent=2)
        with open(hist_dir / "historical_weather.json", "w", encoding="utf-8") as f:
            json.dump(hist_weather, f, indent=2)

        logger.info("Persisted historical data to %s", hist_dir)
        return {
            "status": "success",
            "historical_aqi_records": len(hist_aq),
            "historical_weather_days": len(hist_weather.get("daily", {}).get("time", [])),
            "output_directory": str(hist_dir),
        }

    def start(self):
        """Start the background scheduler with configured polling intervals."""
        if not self._scheduler.running:
            self._scheduler.add_job(
                self.poll_openaq_job,
                trigger=IntervalTrigger(minutes=settings.OPENAQ_POLL_INTERVAL_MINUTES),
                id="openaq_poll",
                name="OpenAQ Periodic Ingestion",
                replace_existing=True,
            )
            self._scheduler.add_job(
                self.poll_openmeteo_job,
                trigger=IntervalTrigger(minutes=settings.OPENMETEO_POLL_INTERVAL_MINUTES),
                id="openmeteo_poll",
                name="Open-Meteo Periodic Ingestion",
                replace_existing=True,
            )
            self._scheduler.start()
            logger.info(
                "Scheduler started. OpenAQ interval: %dm, Open-Meteo interval: %dm",
                settings.OPENAQ_POLL_INTERVAL_MINUTES,
                settings.OPENMETEO_POLL_INTERVAL_MINUTES,
            )

    def stop(self):
        """Gracefully shutdown the background scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped.")
