"""
Public Interface Facade for AeroCast M1 Data Ingestion (SRS v1.1 Compliant).
Provides a unified, source-agnostic API for downstream spatial interpolation (M2),
predictive analytics (M3, M4), and visualization (M6).
Keyed by canonical Zone ID (ZONE-LHR-####).
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from .cache import LocalDataCache
from .scheduler import IngestionScheduler
from .schema import NormalizedRecord

logger = logging.getLogger("aerocast.interface")

# Singleton instances for process-level access
_cache = LocalDataCache()
_scheduler: Optional[IngestionScheduler] = None


def _get_scheduler() -> IngestionScheduler:
    """Retrieve or initialize the global IngestionScheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = IngestionScheduler(cache=_cache)
    return _scheduler


def get_latest_data(zone_id: str, allow_stale: bool = True) -> Optional[Dict[str, Any]]:
    """
    Downstream Facade API: Retrieve the latest normalized environmental & weather data
    for a specific Zone (e.g., 'ZONE-LHR-0001').

    Downstream consumers (Spatial Interpolation engine M2) do not need to know about
    external sources, API keys, or polling mechanics.

    :param zone_id: Canonical Zone identifier (e.g. 'ZONE-LHR-0001')
    :param allow_stale: If True, falls back to the most recent cached snapshot with
                        data_quality.stale = True if current data is unavailable.
    :return: Standardized canonical dictionary matching SRS v1.1 Section 6.1 or None.
    """
    record = _cache.get_latest_record(zone_id)

    if record is None and allow_stale:
        record = _cache.get_stale_fallback(zone_id)

    if record is not None:
        return record.to_canonical_dict()

    logger.warning("No data found for Zone: %s", zone_id)
    return None


def get_all_zone_data(allow_stale: bool = True) -> List[Dict[str, Any]]:
    """
    Downstream Facade API: Retrieve the latest normalized environmental & weather data
    across all ~200 Zones in Lahore.

    :param allow_stale: Whether to include stale fallback records.
    :return: List of standardized canonical dictionaries for all available Zones.
    """
    latest_map = _cache.get_all_latest()
    results = [record.to_canonical_dict() for record in latest_map.values()]
    logger.info("Serving latest data for %d Zones", len(results))
    return results


# Backwards compatibility alias
def get_all_uc_data(allow_stale: bool = True) -> List[Dict[str, Any]]:
    return get_all_zone_data(allow_stale=allow_stale)


async def trigger_refresh_async() -> Dict[str, Any]:
    """
    Asynchronously trigger an immediate refresh across all data sources
    and update local caches.
    """
    sched = _get_scheduler()
    return await sched.trigger_full_sync()


def trigger_refresh() -> Dict[str, Any]:
    """
    Synchronously trigger an immediate ingestion sweep across all data sources.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = asyncio.create_task(trigger_refresh_async())
            return {"status": "refresh_triggered_async"}
        else:
            return loop.run_until_complete(trigger_refresh_async())
    except RuntimeError:
        return asyncio.run(trigger_refresh_async())


def sync_historical(days: int = 730) -> Dict[str, Any]:
    """
    Fetch and persist multi-year historical datasets for ML training and backtesting.
    """
    sched = _get_scheduler()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return {"status": "historical_sync_triggered_async"}
        else:
            return loop.run_until_complete(sched.fetch_historical_dataset(days=days))
    except RuntimeError:
        return asyncio.run(sched.fetch_historical_dataset(days=days))


def get_ingestion_health() -> Dict[str, Any]:
    """
    Check health status and availability of ingestion cache and Zones.
    """
    all_data = _cache.get_all_latest()
    total_records = len(all_data)
    stale_records = sum(1 for r in all_data.values() if r.data_quality.stale)
    interpolated_records = sum(1 for r in all_data.values() if r.data_quality.interpolated)

    return {
        "status": "healthy" if total_records > 0 else "degraded",
        "total_zones_cached": total_records,
        "stale_count": stale_records,
        "interpolated_count": interpolated_records,
        "fresh_count": total_records - stale_records,
    }
