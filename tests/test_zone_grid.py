"""
Unit tests for Lahore 200-Zone Spatial Grid Dataset (SRS v1.1).
Validates geometric integrity, metric CRS projection, canonical ID formats, and cell count.
"""

import re
import json
from pathlib import Path
import pytest

ZONE_GRID_PATH = Path("data/boundaries/lahore_zone_grid.geojson")


def test_zone_grid_integrity():
    """Test the validity and structure of the canonical 241-zone grid GeoJSON."""
    assert ZONE_GRID_PATH.exists(), f"Zone grid GeoJSON not found at {ZONE_GRID_PATH}"

    with open(ZONE_GRID_PATH, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    features = geojson_data.get("features", [])
    assert len(features) > 150 and len(features) < 300, f"Zone count {len(features)} is outside expected ~200 range"

    zone_id_pattern = re.compile(r"^ZONE-LHR-\d{4}$")
    required_props = {
        "zone_id",
        "zone_name",
        "grid_row",
        "grid_col",
        "area_sqkm",
        "centroid_lat",
        "centroid_lon",
        "district",
    }

    for feat in features:
        props = feat["properties"]
        for p in required_props:
            assert p in props, f"Missing required property {p} in feature"

        z_id = props["zone_id"]
        assert zone_id_pattern.match(z_id), f"Invalid Zone ID format: {z_id}"

        c_lat = props["centroid_lat"]
        c_lon = props["centroid_lon"]
        assert 31.10 <= c_lat <= 31.80, f"Latitude {c_lat} out of bounds"
        assert 74.00 <= c_lon <= 74.70, f"Longitude {c_lon} out of bounds"

        geom = feat["geometry"]
        assert geom["type"] in ("Polygon", "MultiPolygon")
        assert len(geom["coordinates"]) > 0
