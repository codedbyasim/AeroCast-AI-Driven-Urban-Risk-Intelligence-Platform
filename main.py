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
    parser.add_argument(
        "--spatial",
        nargs="?",
        const="aqi_pm25",
        metavar="VARIABLE",
        help="Run spatial Kriging interpolation for a variable across all 241 zones (default: aqi_pm25)",
    )
    parser.add_argument(
        "--spatial-health",
        action="store_true",
        help="Display M2 Spatial Interpolation Engine status and diagnostics",
    )
    parser.add_argument(
        "--forecast",
        type=int,
        choices=[24],
        nargs="?",
        const=24,
        metavar="HORIZON",
        help="Generate 24-hour advance AQI hazard forecast for all 241 zones (default: 24)",
    )
    parser.add_argument(
        "--heat-island",
        action="store_true",
        help="Compute Urban Heat Island (UHI) risk scores across all 241 zones",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Trigger end-to-end ML model training (XGBoost 24h Regressor) and persist artifacts",
    )
    parser.add_argument(
        "--ml-health",
        action="store_true",
        help="Display Module M3 ML models and artifact status",
    )
    parser.add_argument(
        "--flood",
        action="store_true",
        help="Compute Module M4 Flash Flood & Urban Waterlogging risk scores across all 241 zones",
    )
    parser.add_argument(
        "--flood-health",
        action="store_true",
        help="Display Module M4 Flash Flood Risk Engine status and diagnostics",
    )
    parser.add_argument(
        "--dispatch-alerts",
        action="store_true",
        help="Evaluate multi-hazard thresholds across all 241 zones and dispatch emergency alerts (Module M7)",
    )
    parser.add_argument(
        "--alerts-health",
        action="store_true",
        help="Display Module M7 Early Warning Alert Dispatcher status and subscriber metrics",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Execute chronological walk-forward model backtesting and extreme event evaluation (Module M8)",
    )
    parser.add_argument(
        "--backtest-drift",
        action="store_true",
        help="Display Module M8 statistical drift detector and model degradation diagnostics",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch Module M9 FastAPI REST API Server (Uvicorn)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the REST API server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the REST API server (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    if args.backtest:
        from backtesting.interface import run_backtest
        print("Executing out-of-band chronological walk-forward backtest (Module M8)...")
        results = run_backtest(export_csv=True)
        print(f"Backtest completed across {results['canonical_zones_evaluated']} canonical zones.")
        print(json.dumps(results, indent=2))
        return

    if args.backtest_drift:
        from backtesting.interface import get_drift_status
        print(json.dumps(get_drift_status(), indent=2))
        return

    if args.alerts_health:
        from alerts.interface import get_alerts_health
        print(json.dumps(get_alerts_health(), indent=2))
        return

    if args.dispatch_alerts:
        from alerts.interface import evaluate_and_dispatch_alerts
        print("Evaluating multi-hazard thresholds across all 241 Lahore zones (Module M7)...")
        result = evaluate_and_dispatch_alerts(force_reevaluate=True)
        print(f"Triggered {result['total_new_alerts_triggered']} alerts | Dispatched {result['total_dispatches_sent']} notifications.")
        print(json.dumps(result, indent=2))
        return

    if args.serve:
        import uvicorn
        print("==================================================")
        print(f"  Starting AeroCast Module M9 REST API Server")
        print(f"  Listening on http://{args.host}:{args.port}")
        print(f"  Interactive Swagger Docs: http://localhost:{args.port}/docs")
        print("==================================================")
        uvicorn.run("api.app:app", host=args.host, port=args.port, reload=False)
        return

    if args.flood_health:
        from flood.interface import get_flood_health
        print(json.dumps(get_flood_health(), indent=2))
        return

    if args.flood:
        from flood.interface import get_all_zones_flood_risk
        print("Computing Module M4 Flash Flood Risk scores across all 241 Lahore zones...")
        flood_scores = get_all_zones_flood_risk(horizon_hours=24, allow_cache=False)
        print(f"Computed {len(flood_scores)} zone flood risk scores.")
        sample_keys = list(flood_scores.keys())[:3]
        sample_output = {k: flood_scores[k] for k in sample_keys}
        print(json.dumps(sample_output, indent=2))
        print(f"... and {len(flood_scores) - 3} more zones scored.")
        return

    if args.ml_health:
        from ml.interface import get_ml_health
        print(json.dumps(get_ml_health(), indent=2))
        return

    if args.train_ml:
        from ml.interface import train_and_save_models
        print("Training Module M3 AQI forecasting model (XGBoost 24h Regressor)...")
        metrics = train_and_save_models()
        print(json.dumps(metrics, indent=2))
        return

    if args.forecast:
        from ml.interface import get_all_aqi_forecasts
        horizon = args.forecast
        print(f"Generating {horizon}-hour advance AQI forecasts across 241 Lahore zones...")
        forecasts = get_all_aqi_forecasts(horizon_hours=horizon, allow_cache=False)
        print(f"Generated {len(forecasts)} forecasts.")
        sample_keys = list(forecasts.keys())[:3]
        sample_output = {k: forecasts[k] for k in sample_keys}
        print(json.dumps(sample_output, indent=2))
        print(f"... and {len(forecasts) - 3} more zones forecasted.")
        return

    if args.heat_island:
        from ml.interface import get_all_heat_island_risk
        print("Computing Urban Heat Island (UHI) risk scores across 241 Lahore zones...")
        uhi_scores = get_all_heat_island_risk(allow_cache=False)
        print(f"Computed {len(uhi_scores)} zone UHI scores.")
        sample_keys = list(uhi_scores.keys())[:3]
        sample_output = {k: uhi_scores[k] for k in sample_keys}
        print(json.dumps(sample_output, indent=2))
        print(f"... and {len(uhi_scores) - 3} more zones scored.")
        return

    if args.spatial_health:
        from spatial.interface import get_spatial_health
        print(json.dumps(get_spatial_health(), indent=2))
        return

    if args.spatial:
        from spatial.interface import get_interpolated_grid
        var_name = args.spatial
        print(f"Running M2 Spatial Kriging interpolation for '{var_name}' across all 241 zones...")
        grid = get_interpolated_grid(var_name, allow_cache=False, force_recompute=True)
        print(f"Successfully interpolated {len(grid)} zones.")
        sample_keys = list(grid.keys())[:3]
        sample_output = {k: grid[k] for k in sample_keys}
        print(json.dumps(sample_output, indent=2))
        print(f"... and {len(grid) - 3} more zones cached.")
        return

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
