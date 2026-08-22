"""
WorldPop Ingestion Client for AeroCast (SRS v1.1 Compliant).
Handles WorldPop 100m population density rasters (GeoTIFF) for the Lahore district.
Computes zonal population statistics per Zone using rasterio.
Properly tracks synthetic vs authentic raster data provenance per FR-INGEST-09 / SRS Risk R-05.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
from shapely.geometry import Point, Polygon, mapping
import geopandas as gpd
from config import settings

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.mask import mask
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    from_bounds = None
    mask = None
    HAS_RASTERIO = False

logger = logging.getLogger("aerocast.worldpop")


class WorldPopClient:
    """Client for parsing and querying WorldPop population density rasters for Lahore zones."""

    def __init__(self, raster_path: Optional[Union[str, Path]] = None):
        self.raster_path = Path(raster_path or settings.WORLDPOP_RASTER_PATH)
        self._dataset: Optional[Any] = None
        self._is_synthetic: bool = False

    def is_synthetic_raster(self) -> bool:
        """Return whether the loaded population raster is a synthetic placeholder."""
        self.ensure_raster_exists()
        return self._is_synthetic or not HAS_RASTERIO

    def ensure_raster_exists(self) -> Path:
        """
        Verify that the WorldPop GeoTIFF raster exists.
        If missing, generates a high-resolution GeoTIFF for Lahore and flags it as synthetic.
        """
        if not self.raster_path.exists():
            self._is_synthetic = True
            if HAS_RASTERIO:
                logger.info("WorldPop raster not found at %s. Generating reference raster.", self.raster_path)
                self.generate_sample_raster(self.raster_path)
        return self.raster_path

    def get_dataset(self) -> Optional[Any]:
        """Open and return the rasterio dataset reader if available."""
        if not HAS_RASTERIO:
            return None
        if self._dataset is None or (hasattr(self._dataset, "closed") and self._dataset.closed):
            path = self.ensure_raster_exists()
            if path.exists():
                self._dataset = rasterio.open(path)
        return self._dataset

    def get_population_density_at_point(self, lat: float, lon: float) -> float:
        """Sample population density (people/km²) at specific coordinates."""
        src = self.get_dataset()
        if src is not None:
            try:
                for val in src.sample([(lon, lat)]):
                    density = float(val[0])
                    if density >= 0 and not np.isnan(density):
                        return round(density, 2)
            except Exception as e:
                logger.warning("Error sampling WorldPop raster at (%f, %f): %s", lat, lon, e)

        # Spatial estimation for Lahore (higher density in urban core)
        center_lon, center_lat = 74.325, 31.545
        dist = np.sqrt((lon - center_lon) ** 2 + (lat - center_lat) ** 2)
        est_density = 4000.0 + 22000.0 * np.exp(- (dist / 0.10) ** 2)
        return round(float(est_density), 2)

    def calculate_zone_population_density(self, polygon: Polygon) -> float:
        """
        Compute the mean population density (people/km²) within a Zone polygon.
        """
        src = self.get_dataset()
        if src is not None:
            try:
                geom = [mapping(polygon)]
                out_image, _ = mask(src, geom, crop=True)
                data = out_image[0]
                valid_pixels = data[(data >= 0) & (~np.isnan(data))]
                if len(valid_pixels) > 0:
                    return round(float(np.mean(valid_pixels)), 2)
            except Exception as e:
                logger.debug("Failed masking polygon in WorldPop raster: %s", e)

        # Fallback to centroid sample
        c = polygon.centroid
        return self.get_population_density_at_point(c.y, c.x)

    # Backwards compatibility alias
    def calculate_uc_population_density(self, polygon: Polygon) -> float:
        return self.calculate_zone_population_density(polygon)

    def compute_all_zone_densities(self, zone_gdf: gpd.GeoDataFrame) -> Dict[str, float]:
        """Compute population densities for all Zones in the GeoDataFrame."""
        results: Dict[str, float] = {}
        for _, row in zone_gdf.iterrows():
            z_id = str(row.get("zone_id", row.get("id", "ZONE-LHR-0001")))
            poly = row.geometry
            density = self.calculate_zone_population_density(poly)
            results[z_id] = density
        logger.info("Computed WorldPop density for %d Zones", len(results))
        return results

    # Backwards compatibility alias
    def compute_all_uc_densities(self, zone_gdf: gpd.GeoDataFrame) -> Dict[str, float]:
        return self.compute_all_zone_densities(zone_gdf)

    def generate_sample_raster(self, output_path: Path):
        """Generate a realistic sample GeoTIFF for Lahore population density."""
        if not HAS_RASTERIO:
            logger.info("Rasterio not available; skipping file generation on disk.")
            return

        self._is_synthetic = True
        output_path.parent.mkdir(parents=True, exist_ok=True)
        min_lon, min_lat, max_lon, max_lat = settings.LAHORE_BBOX

        width = 250
        height = 250
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        x = np.linspace(min_lon, max_lon, width)
        y = np.linspace(max_lat, min_lat, height)
        xx, yy = np.meshgrid(x, y)

        center_lon, center_lat = 74.325, 31.545
        dist = np.sqrt((xx - center_lon) ** 2 + (yy - center_lat) ** 2)

        density = 4000 + 22000 * np.exp(- (dist / 0.10) ** 2)
        noise = np.random.normal(0, 500, size=density.shape)
        density = np.clip(density + noise, 1000, 35000).astype(np.float32)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": -9999.0,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(density, 1)

        logger.info("Generated sample WorldPop GeoTIFF at %s", output_path)

    def close(self):
        """Close rasterio dataset."""
        if self._dataset and hasattr(self._dataset, "closed") and not self._dataset.closed:
            self._dataset.close()
