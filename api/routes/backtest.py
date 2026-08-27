"""
AeroCast Backtesting & Model Evaluation REST API Routes.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timezone

from backtesting.interface import (
    run_backtest,
    get_latest_backtest_results,
    get_drift_status,
    get_backtesting_health,
)

router = APIRouter(prefix="/api/v1/backtest", tags=["Backtesting & Model Evaluation"])


@router.get("/latest", summary="Get Latest Backtest Evaluation Report")
def get_latest_backtest() -> Dict[str, Any]:
    """
    Returns the latest out-of-band backtesting evaluation report across validation
    and future test holdouts, including extreme event recall and spatial Kriging metrics.
    """
    try:
        return get_latest_backtest_results()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch backtest report: {e}")


@router.post("/run", summary="Trigger Chronological Walk-Forward Backtest")
def trigger_backtest(
    export_csv: bool = Query(True, description="Export summary CSV to reports/ directory"),
) -> Dict[str, Any]:
    """
    Runs an out-of-band chronological walk-forward backtest evaluation across all 241 zones.
    """
    try:
        return run_backtest(export_csv=export_csv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest execution failed: {e}")


@router.get("/drift-status", summary="Get Model Statistical Drift Diagnostics")
def get_model_drift() -> Dict[str, Any]:
    """
    Evaluates whether recent error metrics indicate statistical drift or degradation.
    """
    try:
        return get_drift_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate model drift: {e}")


@router.get("/health", summary="Module M8 Backtesting Engine Health")
def get_backtest_health() -> Dict[str, Any]:
    """Return Module M8 diagnostic health status."""
    return get_backtesting_health()
