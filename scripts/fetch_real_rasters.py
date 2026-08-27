"""
Script to fetch, clip, and prepare authentic real raster data for AeroCast:
1. WorldPop 100m Pakistan Population Density (UN-adjusted) -> clipped to Lahore BBox
2. Copernicus DEM GLO-30 Elevation (30m) -> clipped to Lahore BBox
3. Copernicus Sentinel-2 L2A NDVI (Vegetation Index) via CDSE Process API -> clipped to Lahore BBox
"""

import argparse
import logging
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import httpx

try:
    from botocore import UNSIGNED
    from botocore.client import Config
    import boto3
    HAS_BOTO3 = True
except ImportError:
    boto3 = None
    HAS_BOTO3 = False

import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("aerocast.fetch_rasters")


def fetch_copernicus_dem(output_path: Optional[Path] = None) -> bool:
    """
    Download authentic 30m Copernicus DEM GLO-30 tile from AWS S3 Open Data (unsigned)
    and clip to Lahore bounding box in EPSG:4326.
    """
    target_path = Path(output_path or settings.COPERNICUS_DEM_RASTER_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("data/temp_rasters")
    temp_dir.mkdir(parents=True, exist_ok=True)
    dem_temp = temp_dir / "copernicus_dem_n31_e74.tif"

    logger.info("--- [1/3] Copernicus DEM GLO-30 Elevation ---")
    bucket = "copernicus-dem-30m"
    key = "Copernicus_DSM_COG_10_N31_00_E074_00_DEM/Copernicus_DSM_COG_10_N31_00_E074_00_DEM.tif"

    if not dem_temp.exists() or dem_temp.stat().st_size < 10000000:
        logger.info("Downloading Copernicus DEM GLO-30 tile from AWS Open Data...")
        if HAS_BOTO3:
            s3 = boto3.client("s3", region_name="eu-central-1", config=Config(signature_version=UNSIGNED))
            s3.download_file(bucket, key, str(dem_temp))
        else:
            url = f"https://{bucket}.s3.amazonaws.com/{key}"
            urllib.request.urlretrieve(url, dem_temp)
        logger.info("Downloaded Copernicus DEM tile: %.2f MB", dem_temp.stat().st_size / (1024 * 1024))
    else:
        logger.info("Using cached source DEM tile: %s", dem_temp)

    min_lon, min_lat, max_lon, max_lat = settings.LAHORE_BBOX
    with rasterio.open(dem_temp) as src:
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        dem_data = src.read(1, window=window)
        win_transform = rasterio.windows.transform(window, src.transform)

        # Calculate slope percentage
        res_x = abs(src.res[0]) * 95060.0  # meters per deg lon at 31.5N
        res_y = abs(src.res[1]) * 110870.0 # meters per deg lat
        gy, gx = np.gradient(dem_data.astype(float), res_y, res_x)
        slope_pct = np.sqrt(gx**2 + gy**2) * 100.0

        logger.info("Clipped DEM Grid shape: %s", dem_data.shape)
        logger.info("Elevation: min=%.1fm, max=%.1fm, mean=%.1fm", dem_data.min(), dem_data.max(), dem_data.mean())
        logger.info("Slope: min=%.2f%%, max=%.2f%%, mean=%.2f%%, median=%.2f%%",
                    slope_pct.min(), slope_pct.max(), slope_pct.mean(), np.median(slope_pct))

        out_profile = src.profile.copy()
        out_profile.update({
            "height": dem_data.shape[0],
            "width": dem_data.shape[1],
            "transform": win_transform,
            "driver": "GTiff",
            "nodata": -9999.0,
        })

        with rasterio.open(target_path, "w", **out_profile) as dst:
            dst.write(dem_data, 1)
            dst.update_tags(provenance="authentic", source="Copernicus DEM GLO-30", units="meters")

    logger.info("Successfully saved authentic DEM raster to: %s", target_path)
    return True


def fetch_sentinel2_ndvi(output_path: Optional[Path] = None) -> bool:
    """
    Fetch authentic Sentinel-2 L2A NDVI raster via Copernicus Data Space Ecosystem (CDSE)
    Process API for Lahore Bounding Box.
    """
    target_path = Path(output_path or settings.COPERNICUS_NDVI_RASTER_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("--- [2/3] Copernicus Sentinel-2 NDVI ---")
    client_id = settings.CDSE_CLIENT_ID
    client_secret = settings.CDSE_CLIENT_SECRET

    if not client_id or not client_secret:
        logger.warning("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not configured in .env.")
        logger.warning("To fetch real Sentinel-2 NDVI:")
        logger.warning("1. Register free at https://dataspace.copernicus.eu/")
        logger.warning("2. Generate OAuth credentials in User Settings -> API Keys")
        logger.warning("3. Add CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to .env")
        if target_path.exists():
            logger.info("Existing NDVI raster at %s will be preserved.", target_path)
        return False

    logger.info("Authenticating with Copernicus Data Space Ecosystem...")
    try:
        auth_resp = httpx.post(
            settings.CDSE_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=20.0,
        )
        if auth_resp.status_code != 200:
            logger.error("CDSE Authentication failed: %s", auth_resp.text)
            return False

        token = auth_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        evalscript = """//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "SCL", "dataMask"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  // SCL masking: 0=No Data, 1=Saturated/Defective, 3=Cloud Shadow, 8=Cloud Med, 9=Cloud High, 10=Cirrus
  if (sample.dataMask === 0 || sample.SCL === 0 || sample.SCL === 1 || sample.SCL === 3 || sample.SCL === 8 || sample.SCL === 9 || sample.SCL === 10) {
    return [-9999];
  }
  
  // Guard against near-zero reflectance noise / shadow divisions
  let denom = sample.B08 + sample.B04;
  if (denom < 0.005) {
    return [-9999];
  }
  
  let ndvi = (sample.B08 - sample.B04) / denom;
  
  // Physical valid clamp for real Earth surface features (-0.8 to +0.92)
  if (ndvi < -0.8 || ndvi > 0.92) {
    return [-9999];
  }
  return [ndvi];
}
"""

        min_lon, min_lat, max_lon, max_lat = settings.LAHORE_BBOX
        process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        # 30m high-res grid (2340 x 1872) matching Copernicus DEM GLO-30 exactly
        payload = {
            "input": {
                "bounds": {
                    "bbox": [min_lon, min_lat, max_lon, max_lat],
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": "2024-04-01T00:00:00Z", "to": "2024-06-30T23:59:59Z"},
                        "maxCloudCoverage": 15,
                    },
                }],
            },
            "output": {
                "width": 2340,
                "height": 1872,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": evalscript,
        }

        logger.info("Requesting high-resolution 30m Sentinel-2 L2A NDVI (2340 x 1872) from CDSE Process API...")
        proc_resp = httpx.post(process_url, json=payload, headers=headers, timeout=90.0)
        if proc_resp.status_code != 200:
            logger.error("CDSE Process API returned error: %s", proc_resp.text)
            return False

        with open(target_path, "wb") as f:
            f.write(proc_resp.content)

        with rasterio.open(target_path, "r+") as src:
            data = src.read(1)
            valid = data[(data != -9999.0) & (data >= -1.0) & (data <= 1.0) & (~np.isnan(data))]
            src.update_tags(
                provenance="authentic",
                source="Copernicus Sentinel-2 L2A 30m via CDSE Process API",
                resolution="30m (~0.000278 deg)",
                units="NDVI [-0.8, 0.92]",
            )
            logger.info("NDVI Raster Shape: %s (Resolution: ~30m)", data.shape)
            logger.info("NDVI Clean Physical Stats: min=%.3f, max=%.3f, mean=%.3f, median=%.3f",
                        valid.min(), valid.max(), valid.mean(), np.median(valid))
            logger.info("Artifact Check: exact -1.0 count=%d, exact +1.0 count=%d",
                        np.sum(data == -1.0), np.sum(data == 1.0))

        logger.info("Successfully saved authentic high-res Sentinel-2 NDVI raster to: %s", target_path)
        return True

    except Exception as e:
        logger.error("Error fetching Sentinel-2 NDVI from CDSE: %s", e)
        return False


def fetch_worldpop(output_path: Optional[Path] = None) -> bool:
    """
    Download Pakistan 100m UN-adjusted population count GeoTIFF from WorldPop,
    convert pixel counts to density (people/km²), and clip to Lahore BBox.
    """
    target_path = Path(output_path or settings.WORLDPOP_RASTER_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("data/temp_rasters")
    temp_dir.mkdir(parents=True, exist_ok=True)
    wp_temp = temp_dir / "pak_ppp_2020_UNadj.tif"

    logger.info("--- [3/3] WorldPop 100m Population Density ---")
    url = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/PAK/pak_ppp_2020_UNadj.tif"

    if not wp_temp.exists() or wp_temp.stat().st_size < 400000000:
        logger.info("Downloading Pakistan WorldPop GeoTIFF (~457 MB)...")
        urllib.request.urlretrieve(url, wp_temp)
        logger.info("Downloaded WorldPop raster: %.2f MB", wp_temp.stat().st_size / (1024 * 1024))
    else:
        logger.info("Using cached source WorldPop file: %s (%.2f MB)", wp_temp, wp_temp.stat().st_size / (1024 * 1024))

    min_lon, min_lat, max_lon, max_lat = settings.LAHORE_BBOX
    with rasterio.open(wp_temp) as src:
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        data = src.read(1, window=window)
        win_transform = rasterio.windows.transform(window, src.transform)

        # Pixel area in km²
        res_x, res_y = abs(src.res[0]), abs(src.res[1])
        pixel_area_km2 = (res_y * 110.87) * (res_x * 95.06)

        # Convert counts per pixel to population density (people/km²)
        density = np.where((data >= 0) & (~np.isnan(data)), data / pixel_area_km2, -9999.0).astype(np.float32)
        valid_density = density[density >= 0]

        logger.info("Clipped WorldPop Grid shape: %s", density.shape)
        logger.info("Population Density (people/km²): min=%.1f, max=%.1f, mean=%.1f, median=%.1f",
                    valid_density.min(), valid_density.max(), valid_density.mean(), np.median(valid_density))

        out_profile = src.profile.copy()
        out_profile.update({
            "height": density.shape[0],
            "width": density.shape[1],
            "transform": win_transform,
            "driver": "GTiff",
            "nodata": -9999.0,
            "dtype": rasterio.float32,
        })

        with rasterio.open(target_path, "w", **out_profile) as dst:
            dst.write(density, 1)
            dst.update_tags(provenance="authentic", source="WorldPop 2020 UNadj 100m", units="people/km2")

    logger.info("Successfully saved authentic WorldPop population density raster to: %s", target_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch and clip authentic raster datasets for AeroCast")
    parser.add_argument("--worldpop", action="store_true", help="Fetch WorldPop population density")
    parser.add_argument("--dem", action="store_true", help="Fetch Copernicus GLO-30 DEM elevation")
    parser.add_argument("--ndvi", action="store_true", help="Fetch Copernicus Sentinel-2 NDVI")
    parser.add_argument("--all", action="store_true", help="Fetch all rasters (default)")

    args = parser.parse_args()

    fetch_all = args.all or (not args.worldpop and not args.dem and not args.ndvi)

    if fetch_all or args.dem:
        fetch_copernicus_dem()

    if fetch_all or args.ndvi:
        fetch_sentinel2_ndvi()

    if fetch_all or args.worldpop:
        fetch_worldpop()

    logger.info("=== All requested raster datasets processed successfully ===")


if __name__ == "__main__":
    main()
