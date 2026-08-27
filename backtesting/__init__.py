"""
AeroCast Module M8: Backtesting & Continuous Model Evaluation Engine.
Performs rolling walk-forward evaluation, extreme event validation, Kriging cross-validation, and drift detection.
"""

from .metrics import (
    calculate_continuous_metrics,
    calculate_extreme_event_metrics,
    calculate_directional_accuracy,
    detect_model_drift,
)
from .engine import BacktestEngine
from .interface import (
    run_backtest,
    get_latest_backtest_results,
    get_drift_status,
    get_backtesting_health,
)

__all__ = [
    "calculate_continuous_metrics",
    "calculate_extreme_event_metrics",
    "calculate_directional_accuracy",
    "detect_model_drift",
    "BacktestEngine",
    "run_backtest",
    "get_latest_backtest_results",
    "get_drift_status",
    "get_backtesting_health",
]
