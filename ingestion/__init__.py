"""
AeroCast M1 Data Ingestion Package.
Provides ingestion clients, normalizer, caching, and scheduling for urban risk intelligence.
"""

from .schema import NormalizedRecord, Metrics, SpatialContext, DataQuality
from .openaq_client import OpenAQClient
from .openmeteo_client import OpenMeteoClient
from .osm_hdx_client import OSMHDXClient
from .worldpop_client import WorldPopClient
from .copernicus_client import CopernicusClient
from .normalizer import DataNormalizer
from .cache import LocalDataCache
from .scheduler import IngestionScheduler
from .interface import (
    get_latest_data,
    get_all_uc_data,
    trigger_refresh,
    get_ingestion_health,
)

__all__ = [
    "NormalizedRecord",
    "Metrics",
    "SpatialContext",
    "DataQuality",
    "OpenAQClient",
    "OpenMeteoClient",
    "OSMHDXClient",
    "WorldPopClient",
    "CopernicusClient",
    "DataNormalizer",
    "LocalDataCache",
    "IngestionScheduler",
    "get_latest_data",
    "get_all_uc_data",
    "trigger_refresh",
    "get_ingestion_health",
]
