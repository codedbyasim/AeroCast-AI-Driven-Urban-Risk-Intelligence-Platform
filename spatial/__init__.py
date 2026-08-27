"""
AeroCast — Module M2: Spatial Interpolation Engine
==================================================
Package exports for geostatistical Kriging, environmental covariates,
optical sensor calibration, and public facade API.
"""

from .interface import (
    get_interpolated_grid,
    get_all_interpolated_grid,
    get_zone_interpolated,
    trigger_spatial_interpolation,
    get_spatial_health,
)
from .kriging_engine import KrigingEngine
from .covariates import CovariateManager
from .calibration import calibrate_pm25_optical

__all__ = [
    "get_interpolated_grid",
    "get_all_interpolated_grid",
    "get_zone_interpolated",
    "trigger_spatial_interpolation",
    "get_spatial_health",
    "KrigingEngine",
    "CovariateManager",
    "calibrate_pm25_optical",
]
