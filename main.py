"""
AeroCast — M1 Data Ingestion Service Entrypoint (SRS v1.1 Compliant).
Coordinates data ingestion, normalization, local caching, and periodic polling
for environmental and urban risk intelligence across Lahore's 200-Zone spatial grid.
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from config import settings
from ingestion.cache import LocalDataCache
from ingestion.scheduler import IngestionScheduler
from ingestion.interface import (
    get_latest_data,
    get_all_zone_data,
    get_ingestion_health,
    sync_historical,
)


def setup_logging():
    """Configure structured logging format for AeroCast."""
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


async def run_daemon():
    """
    Run the ingestion service continuously as a background daemon.
    Initializes static data, runs an initial ingestion sweep, and schedules periodic polls.
    """
    logger = logging.getLogger("aerocast.main")
    logger.info("==================================================")
    logger.info("  Starting AeroCast — M1 Data Ingestion Daemon")
    logger.info("  Covering 200-Zone Metric Grid (~1800 sq km)")
    logger.info("==================================================")

    cache = LocalDataCache()
    scheduler = IngestionScheduler(cache=cache)

    # 1. Initialize static data & perform initial synchronization sweep
    logger.info("Executing initial data sync across all sources...")
    sync_result = await scheduler.trigger_full_sync()
    logger.info("Initial sync completed: %s", sync_result)

    # 2. Start APScheduler background jobs
    scheduler.start()
    logger.info("Periodic polling active. Press Ctrl+C to terminate.")

    # 3. Graceful shutdown handler
    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Termination signal received. Shutting down gracefully...")
        scheduler.stop()
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutdown initiated by user.")
    finally:
        scheduler.stop()
        logger.info("Ingestion service terminated cleanly.")


async def run_single_sync():
    """Perform a one-time ingestion sweep across all data sources and display summary."""
    logger = logging.getLogger("aerocast.main")
    logger.info("Executing single ingestion sweep across 200-Zone grid...")

    cache = LocalDataCache()
    scheduler = IngestionScheduler(cache=cache)

    res = await scheduler.trigger_full_sync()
    logger.info("Ingestion complete! Summary: %s", json.dumps(res, indent=2))

    # Print sample record
    sample = cache.get_latest_record("ZONE-LHR-0001")
    if sample:
        print("\n--- Canonical Record Sample (ZONE-LHR-0001) ---")
        print(json.dumps(sample.to_canonical_dict(), indent=2))
    print("\n[+] Health status:", json.dumps(get_ingestion_health(), indent=2))


def main():
    """CLI Entrypoint for AeroCast Ingestion service."""
    setup_logging()
    logger = logging.getLogger("aerocast.cli")

    parser = argparse.ArgumentParser(
        description="AeroCast — M1 Data Ingestion Layer CLI (SRS v1.1)"
    )
    parser.add_argument(
        "--sync", "--once",
        action="store_true",
        help="Run a single full ingestion sweep across all sources and exit",
    )
    parser.add_argument(
        "--query",
        type=str,
        metavar="ZONE_ID",
        help="Query latest normalized data for a specific Zone (e.g. ZONE-LHR-0001)",
    )
    parser.add_argument(
        "--query-all",
        action="store_true",
        help="Query and list latest normalized data for all cached Zones",
    )
    parser.add_argument(
        "--sync-historical",
        action="store_true",
        help="Fetch and persist 2-year historical weather and AQI datasets for ML training",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Display cache health and summary statistics",
    )

    args = parser.parse_args()

    if args.query:
        data = get_latest_data(args.query)
        if data:
            print(json.dumps(data, indent=2))
        else:
            print(f"No data found for Zone '{args.query}'", file=sys.stderr)
            sys.exit(1)
        return

    if args.query_all:
        all_data = get_all_zone_data()
        print(f"Total Zones cached: {len(all_data)}")
        if all_data:
            print(json.dumps(all_data[:3], indent=2))
            print(f"... and {len(all_data) - 3} more records.")
        return

    if args.sync_historical:
        print("Fetching 2-year historical environmental datasets...")
        res = sync_historical(days=730)
        print(json.dumps(res, indent=2))
        return

    if args.health:
        print(json.dumps(get_ingestion_health(), indent=2))
        return

    if args.sync:
        try:
            asyncio.run(run_single_sync())
        except KeyboardInterrupt:
            logger.info("Execution interrupted.")
        return

    # Default action: run as daemon
    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        logger.info("Daemon interrupted by user.")


if __name__ == "__main__":
    main()
