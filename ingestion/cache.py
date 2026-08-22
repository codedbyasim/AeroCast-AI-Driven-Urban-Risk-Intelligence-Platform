"""
Local Caching Layer for AeroCast Data Ingestion (SRS v1.1 Compliant).
Persists normalized records locally in JSON format and implements stale-fallback logic
to guarantee continuous downstream availability during upstream API failures.
All records are keyed by canonical Zone ID (ZONE-LHR-####).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import threading

from .schema import NormalizedRecord
from config import settings

logger = logging.getLogger("aerocast.cache")


class LocalDataCache:
    """
    File-based local caching layer for normalized environmental and urban risk data.
    Organized by Zone ID (ZONE-LHR-####).
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.records_dir = self.cache_dir / "records"
        self.latest_dir = self.cache_dir / "latest"
        self._lock = threading.RLock()
        self._init_storage()

    def _init_storage(self):
        """Create necessary cache directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

    def _get_latest_file_path(self, zone_id: str) -> Path:
        """Return the path to the latest JSON snapshot for a given Zone ID."""
        safe_id = zone_id.replace(":", "_").replace("/", "_")
        return self.latest_dir / f"{safe_id}.json"

    def save_record(self, record: NormalizedRecord) -> bool:
        """
        Save a single NormalizedRecord to the local cache.
        Updates both the historical time-series log and the latest snapshot for the Zone.
        """
        if not record or not record.zone_id:
            logger.warning("Attempted to cache empty or invalid record")
            return False

        with self._lock:
            try:
                z_id = record.zone_id
                canonical_dict = record.to_canonical_dict()

                # 1. Update latest snapshot for this Zone
                latest_path = self._get_latest_file_path(z_id)
                with open(latest_path, "w", encoding="utf-8") as f:
                    json.dump(canonical_dict, f, indent=2)

                # 2. Append to Zone historical partition
                zone_hist_dir = self.records_dir / z_id
                zone_hist_dir.mkdir(parents=True, exist_ok=True)
                ts_str = record.timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
                hist_file = zone_hist_dir / f"{ts_str}.json"
                with open(hist_file, "w", encoding="utf-8") as f:
                    json.dump(canonical_dict, f, indent=2)

                logger.debug("Successfully cached record for %s", z_id)
                return True
            except Exception as e:
                logger.error("Failed to save record for %s to cache: %s", record.zone_id, e)
                return False

    def save_records(self, records: List[NormalizedRecord]) -> int:
        """Batch save multiple normalized records."""
        count = 0
        with self._lock:
            for rec in records:
                if self.save_record(rec):
                    count += 1
        logger.info("Batch cached %d / %d records", count, len(records))
        return count

    def get_latest_record(self, zone_id: str) -> Optional[NormalizedRecord]:
        """
        Retrieve the latest cached NormalizedRecord for a specific Zone.
        """
        latest_path = self._get_latest_file_path(zone_id)
        if not latest_path.exists():
            logger.debug("No cached record found for %s", zone_id)
            return None

        with self._lock:
            try:
                with open(latest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return NormalizedRecord(**data)
            except Exception as e:
                logger.error("Error reading cache for %s: %s", zone_id, e)
                return None

    def get_all_latest(self) -> Dict[str, NormalizedRecord]:
        """
        Retrieve a mapping of zone_id -> latest NormalizedRecord for all cached Zones.
        """
        results: Dict[str, NormalizedRecord] = {}
        with self._lock:
            for path in self.latest_dir.glob("*.json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    rec = NormalizedRecord(**data)
                    results[rec.zone_id] = rec
                except Exception as e:
                    logger.warning("Error reading cached file %s: %s", path, e)
        return results

    def get_stale_fallback(
        self,
        zone_id: str,
        reason: Optional[str] = "Upstream API fetch failure; served from cache",
    ) -> Optional[NormalizedRecord]:
        """
        Retrieve the most recent cached value for a Zone and mark `data_quality.stale = True`.
        Ensures continuous service availability for downstream Spatial Engine during outages.
        """
        rec = self.get_latest_record(zone_id)
        if not rec:
            logger.warning("No stale fallback available for %s", zone_id)
            return None

        # Update data_quality to mark as stale
        rec.data_quality.stale = True
        rec.data_quality.notes = reason
        rec.data_quality.confidence_score = max(0.4, round(rec.data_quality.confidence_score * 0.75, 2))
        logger.info("Serving stale fallback data for %s (notes: %s)", zone_id, reason)
        return rec

    def clear_cache(self):
        """Clear all cached records (primarily for testing and clean syncs)."""
        import shutil
        with self._lock:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self._init_storage()
            logger.info("Cache cleared successfully")
