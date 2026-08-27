"""
Detailed proof of real GeoTIFF raster extraction and Kriging for a specific Lahore zone.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping

from config import settings
from spatial.interface import get_zone_interpolated


def main(target_zone_id: str = "ZONE-LHR-0075"):
    zone_path = Path(settings.OSM_HDX_ZONE_GRID_PATH)
    with open(zone_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    matching = [f for f in gj["features"] if f["properties"]["zone_id"] == target_zone_id]
    if not matching:
        print(f"Zone {target_zone_id} not found!")
        return

    zone_feat = matching[0]
    props = zone_feat["properties"]
    geom = shape(zone_feat["geometry"])
    geom_mapping = [mapping(geom)]

    print("=" * 76)
    print("          PROOF OF LIVE RASTER & SPATIAL ENGINE INTEGRATION             ")
    print("=" * 76 + "\n")
    print(f"Target Zone ID     : {props['zone_id']}")
    print(f"Grid Coordinates   : Row {props['grid_row']}, Col {props['grid_col']}")
    print(f"Zone Centroid      : {props['centroid_lat']:.5f}° N, {props['centroid_lon']:.5f}° E")
    print(f"Zone Polygon Area  : {props['area_sqkm']:.2f} km²")
    print("Location Context   : Gulberg / Mall Road / Jail Road Urban Core\n")

    # 1. WorldPop GeoTIFF
    wp_path = Path(settings.WORLDPOP_RASTER_PATH)
    with rasterio.open(wp_path) as src:
        out_img, _ = mask(src, geom_mapping, crop=True)
        wp_data = out_img[0]
        valid_wp = wp_data[(wp_data >= 0) & (~np.isnan(wp_data))]
        wp_tags = src.tags()

    print("--- [1] WorldPop 100m Population Density Raster (.tif) ---")
    print(f"  • Source File         : {wp_path.name}")
    print(f"  • Metadata Provenance : {wp_tags.get('provenance')} (Source: {wp_tags.get('source')})")
    print(f"  • Pixels inside Zone  : {len(valid_wp):,} cells (100m resolution)")
    print(f"  • Min Cell Density    : {valid_wp.min():,.1f} people/km²")
    print(f"  • Mean Zone Density   : {valid_wp.mean():,.1f} people/km²")
    print(f"  • Max Cell Density    : {valid_wp.max():,.1f} people/km²")
    estimated_pop = int(valid_wp.mean() * props["area_sqkm"])
    print(f"  • Estimated Total Pop : {estimated_pop:,} citizens living in this {props['area_sqkm']:.1f} km² zone\n")

    # 2. Copernicus DEM GeoTIFF
    dem_path = Path(settings.COPERNICUS_DEM_RASTER_PATH)
    with rasterio.open(dem_path) as src:
        out_img, _ = mask(src, geom_mapping, crop=True)
        dem_data = out_img[0]
        valid_dem = dem_data[(dem_data > -9000) & (~np.isnan(dem_data))]
        dem_tags = src.tags()

    print("--- [2] Copernicus DEM GLO-30 Elevation Raster (.tif) ---")
    print(f"  • Source File         : {dem_path.name}")
    print(f"  • Metadata Provenance : {dem_tags.get('provenance')} (Source: {dem_tags.get('source')})")
    print(f"  • Pixels inside Zone  : {len(valid_dem):,} cells (30m high-res elevation)")
    print(f"  • Elevation Min       : {valid_dem.min():.2f} m")
    print(f"  • Elevation Mean      : {valid_dem.mean():.2f} m above sea level")
    print(f"  • Elevation Max       : {valid_dem.max():.2f} m")
    print(f"  • Topography Gradient : Flat alluvial plain (relief span: {(valid_dem.max() - valid_dem.min()):.1f}m)\n")

    # 3. Sentinel-2 NDVI GeoTIFF
    ndvi_path = Path(settings.COPERNICUS_NDVI_RASTER_PATH)
    with rasterio.open(ndvi_path) as src:
        out_img, _ = mask(src, geom_mapping, crop=True)
        ndvi_data = out_img[0]
        valid_ndvi = ndvi_data[(ndvi_data >= -1.0) & (ndvi_data <= 1.0) & (~np.isnan(ndvi_data))]
        ndvi_tags = src.tags()

    print("--- [3] Copernicus Sentinel-2 L2A NDVI Raster (.tif) ---")
    print(f"  • Source File         : {ndvi_path.name}")
    print(f"  • Metadata Provenance : {ndvi_tags.get('provenance')} (Source: {ndvi_tags.get('source')})")
    print(f"  • Pixels inside Zone  : {len(valid_ndvi):,} cells (Satellite multispectral)")
    print(f"  • NDVI Mean Index     : {valid_ndvi.mean():.3f} (Commercial urban built-up vs greenery index)\n")

    # 4. Spatial Engine Output
    zone_all_vars = get_zone_interpolated(target_zone_id)
    spatial_res = zone_all_vars.get("aqi_pm25") if zone_all_vars else {}
    if not spatial_res:
        # Fallback to single variable grid
        from spatial.interface import get_interpolated_grid
        pm25_grid = get_interpolated_grid("aqi_pm25")
        spatial_res = pm25_grid.get(target_zone_id, {})

    print("--- [4] Spatial Engine Output for this Zone ---")
    print(f"  • Target Variable     : aqi_pm25")
    val = spatial_res.get('value')
    if val is not None:
        print(f"  • Interpolated / Value: {val:.2f} µg/m³")
    else:
        print(f"  • Interpolated / Value: N/A")
    print(f"  • Interpolation Method: {spatial_res.get('method')}")
    print(f"  • Kriging Variance    : {spatial_res.get('variance', 0.0):.2f}")
    conf = spatial_res.get('confidence_score')
    if conf is not None:
        print(f"  • Confidence Score    : {conf * 100:.1f}%")
    print(f"  • Diagnostic Notes    : {spatial_res.get('notes')}")
    print("=" * 76)


if __name__ == "__main__":
    zone = sys.argv[1] if len(sys.argv) > 1 else "ZONE-LHR-0075"
    main(zone)
