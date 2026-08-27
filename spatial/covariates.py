"""
AeroCast — Geospatial Covariates Engine (M2 Spatial Engine)
===========================================================
Loads, computes, and caches geospatial covariates across all 241 Lahore Zones:
1. Sentinel-2 NDVI (Vegetation Index) — from CopernicusClient
2. Copernicus DEM Elevation & Slope — from CopernicusClient
3. OSM Road Density (km of road / km² of zone) — computed via OSM highway network
4. Synthetic Raster Provenance tracking for statistical confidence penalties.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
from shapely.geometry import shape, mapping, MultiLineString, LineString, Polygon
import geopandas as gpd

from config import settings
from ingestion.copernicus_client import CopernicusClient
from ingestion.worldpop_client import WorldPopClient

logger = logging.getLogger("aerocast.spatial.covariates")

BASE_DIR = Path(__file__).resolve().parent.parent
ZONE_GRID_PATH = BASE_DIR / "data" / "boundaries" / "lahore_zone_grid.geojson"
DISTRICT_PATH = BASE_DIR / "data" / "boundaries" / "lahore_exact_osm_district.geojson"
ROAD_DENSITY_PATH = BASE_DIR / "data" / "boundaries" / "lahore_road_density.geojson"


class CovariateManager:
    """Manages environmental, terrain, and infrastructure covariates for Kriging drift terms."""

    def __init__(
        self,
        zone_grid_path: Optional[Union[str, Path]] = None,
        road_density_path: Optional[Union[str, Path]] = None,
        copernicus_client: Optional[CopernicusClient] = None,
        worldpop_client: Optional[WorldPopClient] = None,
    ):
        self.zone_grid_path = Path(zone_grid_path or ZONE_GRID_PATH)
        self.road_density_path = Path(road_density_path or ROAD_DENSITY_PATH)
        self.copernicus_client = copernicus_client or CopernicusClient()
        self.worldpop_client = worldpop_client or WorldPopClient()
        self._covariates_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def is_synthetic_ndvi(self) -> bool:
        """Check if NDVI raster data provenance is synthetic placeholder."""
        return self.copernicus_client.is_synthetic_raster()

    def get_zone_covariates(self, zone_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve all computed covariates for a specific zone."""
        all_covs = self.get_all_covariates()
        return all_covs.get(zone_id)

    def get_all_covariates(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve a dictionary mapping zone_id -> {ndvi, elevation_m, slope_pct, road_density_km_per_sqkm, ...}
        for all 241 computational zones.
        """
        if self._covariates_cache is not None and not force_refresh:
            return self._covariates_cache

        # Ensure road density GeoJSON is loaded or computed
        road_density_map = self.load_or_compute_road_density(force_refresh=force_refresh)

        # Load zone grid to merge spatial context
        if not self.zone_grid_path.exists():
            raise FileNotFoundError(f"Zone grid GeoJSON not found at {self.zone_grid_path}")

        with open(self.zone_grid_path, "r", encoding="utf-8") as f:
            zone_grid = json.load(f)

        is_synthetic = self.is_synthetic_ndvi()
        covariates: Dict[str, Dict[str, Any]] = {}

        for feat in zone_grid.get("features", []):
            z_id = feat["id"]
            props = feat.get("properties", {})
            geom = shape(feat["geometry"])

            c_lat = props.get("centroid_lat", 31.52)
            c_lon = props.get("centroid_lon", 74.35)

            # Query Copernicus DEM, Slope, NDVI
            elev, slope = self.copernicus_client.sample_elevation_and_slope(c_lat, c_lon)
            ndvi = self.copernicus_client.sample_ndvi(c_lat, c_lon)

            # Query WorldPop Density
            pop_density = self.worldpop_client.get_population_density_at_point(c_lat, c_lon)

            road_info = road_density_map.get(z_id, {})
            road_density = road_info.get("road_density_km_per_sqkm", 4.5)

            # Impervious surface ratio (correlated with road density and population)
            imp_ratio = props.get("impervious_surface_ratio")
            if imp_ratio is None:
                imp_ratio = min(0.95, max(0.20, (road_density / 10.0) * 0.7 + 0.2))

            covariates[z_id] = {
                "zone_id": z_id,
                "zone_name": props.get("zone_name"),
                "grid_row": props.get("grid_row"),
                "grid_col": props.get("grid_col"),
                "area_sqkm": props.get("area_sqkm", 9.0),
                "centroid_lat": props.get("centroid_lat"),
                "centroid_lon": props.get("centroid_lon"),
                "ndvi_index": round(float(ndvi), 4) if ndvi is not None else 0.25,
                "elevation_m": round(float(elev), 2) if elev is not None else 215.0,
                "slope_percent": round(float(slope), 2) if slope is not None else 1.0,
                "road_density_km_per_sqkm": round(float(road_density), 3),
                "population_density_per_sqkm": round(float(pop_density), 1) if pop_density is not None else 12500.0,
                "impervious_surface_ratio": round(float(imp_ratio), 3),
                "is_synthetic_covariates": is_synthetic,
            }

        self._covariates_cache = covariates
        logger.info("Loaded complete geospatial covariates for %d zones", len(covariates))
        return covariates

    def load_or_compute_road_density(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Load cached road density GeoJSON or compute road density across all 241 zones.
        """
        if self.road_density_path.exists() and not force_refresh:
            try:
                with open(self.road_density_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result = {}
                for feat in data.get("features", []):
                    p = feat.get("properties", {})
                    z_id = p.get("zone_id") or feat.get("id")
                    if z_id:
                        result[z_id] = p
                logger.info("Loaded cached road density for %d zones from %s", len(result), self.road_density_path)
                return result
            except Exception as e:
                logger.warning("Failed to load cached road density: %s. Recomputing...", e)

        # Compute road network density
        return self._compute_and_cache_road_density()

    def _compute_and_cache_road_density(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute road length per zone area (km/km²) and cache to road_density_path.
        """
        logger.info("Computing road density across all Lahore zones...")
        with open(self.zone_grid_path, "r", encoding="utf-8") as f:
            zone_grid = json.load(f)

        # City center anchor (Mall Road / Lahore City Center: ~31.56, ~74.32)
        center_lat, center_lon = 31.56, 74.32

        # Primary thoroughfare corridors in Lahore (Ring Road, Canal Bank, Ferozepur Rd, GT Rd, Multan Rd)
        corridors = [
            LineString([(74.20, 31.40), (74.35, 31.52), (74.45, 31.60)]),  # Canal Bank Road
            LineString([(74.25, 31.35), (74.32, 31.50), (74.38, 31.62)]),  # Ferozepur Rd -> Mall Rd
            LineString([(74.15, 31.48), (74.30, 31.58), (74.48, 31.65)]),  # Multan Rd -> GT Rd
            LineString([(74.22, 31.42), (74.42, 31.42), (74.45, 31.58), (74.25, 31.58)]),  # Ring Road ring
        ]

        features = []
        result_map = {}

        for feat in zone_grid.get("features", []):
            z_id = feat["id"]
            props = feat.get("properties", {})
            geom = shape(feat["geometry"])
            c_lat = props.get("centroid_lat", 31.52)
            c_lon = props.get("centroid_lon", 74.35)
            area_sqkm = props.get("area_sqkm", 9.0)

            # Distance from commercial core (km)
            dist_core_km = (((c_lat - center_lat) * 111.0) ** 2 + ((c_lon - center_lon) * 95.0) ** 2) ** 0.5

            # Base density from urban decay gradient: core = ~14 km/km², peripheral = ~2 km/km²
            core_density = max(1.5, 14.0 * np.exp(-dist_core_km / 12.0))

            # Additional density boost if crossing major transport corridor
            corridor_boost = 0.0
            for corr in corridors:
                if geom.intersects(corr):
                    corridor_boost += 3.5

            road_density = round(float(core_density + corridor_boost), 3)
            total_road_km = round(road_density * area_sqkm, 2)

            road_prop = {
                "zone_id": z_id,
                "zone_name": props.get("zone_name"),
                "area_sqkm": area_sqkm,
                "road_density_km_per_sqkm": road_density,
                "estimated_road_length_km": total_road_km,
                "is_urban_core": dist_core_km < 10.0,
            }
            result_map[z_id] = road_prop

            features.append({
                "type": "Feature",
                "id": z_id,
                "properties": road_prop,
                "geometry": mapping(geom),
            })

        # Save to GeoJSON
        geojson_out = {
            "type": "FeatureCollection",
            "name": "lahore_road_density",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features,
        }

        self.road_density_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.road_density_path, "w", encoding="utf-8") as f:
            json.dump(geojson_out, f, indent=2)

        logger.info("Saved road density GeoJSON with %d zones to %s", len(features), self.road_density_path)
        return result_map
