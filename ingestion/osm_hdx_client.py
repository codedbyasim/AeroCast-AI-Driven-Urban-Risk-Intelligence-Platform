"""
OpenStreetMap (OSM) & HDX Ingestion Client for AeroCast (SRS v1.1 Compliant).
Loads and queries Lahore's canonical 200-Zone spatial grid (lahore_zone_grid.geojson).
Performs point-in-polygon lookups to map coordinate observations (lat, lon) to Zone IDs (ZONE-LHR-####).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import geopandas as gpd
from shapely.geometry import Point, shape
from config import settings

logger = logging.getLogger("aerocast.osm_hdx")


class OSMHDXClient:
    """Client for querying the Lahore 200-Zone spatial grid."""

    def __init__(self, zone_grid_path: Optional[Union[str, Path]] = None):
        self.zone_grid_path = Path(zone_grid_path or settings.OSM_HDX_ZONE_GRID_PATH)
        self._gdf: Optional[gpd.GeoDataFrame] = None

    def load_zone_grid(self) -> gpd.GeoDataFrame:
        """
        Load the canonical Lahore Zone grid into a GeoPandas GeoDataFrame.
        If file does not exist, automatically generates it using build_zone_grid.
        """
        if self._gdf is not None:
            return self._gdf

        if not self.zone_grid_path.exists():
            logger.info("Zone grid file not found at %s. Generating canonical zone grid...", self.zone_grid_path)
            from build_zone_grid import generate_zone_grid
            generate_zone_grid(output_geojson_path=self.zone_grid_path)

        logger.info("Loading Lahore zone grid from %s", self.zone_grid_path)
        try:
            with open(self.zone_grid_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            if "features" in raw_json:
                gdf = gpd.GeoDataFrame.from_features(raw_json["features"], crs="EPSG:4326")
            else:
                gdf = gpd.GeoDataFrame.from_features([raw_json], crs="EPSG:4326")

            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)

            self._gdf = gdf
            logger.info("Successfully loaded %d zones from grid file", len(self._gdf))
            return self._gdf
        except Exception as e:
            logger.error("Failed to load zone grid from %s: %s", self.zone_grid_path, e, exc_info=True)
            raise

    # Backwards compatibility alias
    def load_union_councils(self) -> gpd.GeoDataFrame:
        return self.load_zone_grid()

    def find_zone_by_coordinates(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Perform spatial point-in-polygon lookup to map lat/lon to a Lahore Zone.
        Falls back to nearest zone polygon centroid if point lies slightly outside clipped boundaries.
        """
        gdf = self.load_zone_grid()
        pt = Point(lon, lat)

        # 1. Direct spatial polygon containment check
        matches = gdf[gdf.geometry.contains(pt)]
        if not matches.empty:
            row = matches.iloc[0]
            return self._row_to_dict(row)

        # 2. Fallback: Nearest zone centroid
        centroids = gdf.geometry.centroid
        distances = (centroids.x - pt.x) ** 2 + (centroids.y - pt.y) ** 2
        nearest_idx = distances.idxmin()
        row = gdf.loc[nearest_idx]
        return self._row_to_dict(row)

    # Backwards compatibility alias
    def find_uc_by_coordinates(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        return self.find_zone_by_coordinates(lat, lon)

    def _row_to_dict(self, row: Any) -> Dict[str, Any]:
        """Format GeoDataFrame row into standardized zone dictionary."""
        zone_id = str(row.get("zone_id", row.get("id", "ZONE-LHR-0001")))
        c_lat = float(row.get("centroid_lat", row.geometry.centroid.y))
        c_lon = float(row.get("centroid_lon", row.geometry.centroid.x))

        return {
            "zone_id": zone_id,
            "zone_name": str(row.get("zone_name", f"Zone {zone_id}")),
            "grid_row": int(row.get("grid_row", 1)),
            "grid_col": int(row.get("grid_col", 1)),
            "area_sqkm": float(row.get("area_sqkm", 9.0)),
            "centroid_lat": c_lat,
            "centroid_lon": c_lon,
            "district": str(row.get("district", "Lahore District")),
            "impervious_surface_ratio": float(row.get("impervious_surface_ratio", 0.65)),
        }

    def get_all_zones(self) -> List[Dict[str, Any]]:
        """Return list of all Lahore Zone metadata records."""
        gdf = self.load_zone_grid()
        results = []
        for _, row in gdf.iterrows():
            results.append(self._row_to_dict(row))
        return results

    # Backwards compatibility alias
    def get_all_union_councils(self) -> List[Dict[str, Any]]:
        return self.get_all_zones()
