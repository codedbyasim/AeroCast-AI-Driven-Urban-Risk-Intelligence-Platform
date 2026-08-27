"""
AeroCast — Sensor Calibration & Quality Adjustment (M2 Spatial Engine)
======================================================================
Provides EPA-standard relative humidity corrections for low-cost optical
particle count sensors (e.g., Plantower PMS5003 / PurpleAir).

Physical Rationale:
-------------------
Low-cost nephelometric / optical particle counters measure light scattering.
In high-humidity environments (common during Lahore winter smog inversions),
hygroscopic aerosol particles absorb water vapor and swell in optical diameter.
This causes optical sensors to overestimate PM2.5 mass concentration by 20%–70%.

This module applies the US EPA empirical hygroscopic growth adjustment
to calibrate raw PM2.5 observations before feeding them into Kriging models.
"""

import logging
from typing import Optional

logger = logging.getLogger("aerocast.spatial.calibration")


def calibrate_pm25_optical(
    raw_pm25: Optional[float],
    relative_humidity_percent: Optional[float],
) -> Optional[float]:
    """
    Apply EPA empirical hygroscopic humidity adjustment to raw optical PM2.5.

    Formula:
    --------
    For RH <= 30%:
        PM2.5_corrected = PM2.5_raw (no significant hygroscopic growth)
    For 30% < RH <= 100%:
        Hygroscopic Growth Factor = 1.0 + 0.24 * (RH / 100)^2
        PM2.5_corrected = PM2.5_raw / Growth_Factor

    :param raw_pm25: Raw uncalibrated PM2.5 reading in µg/m³.
    :param relative_humidity_percent: Relative humidity in % (0 - 100).
    :return: Calibrated PM2.5 concentration in µg/m³ (rounded to 2 decimal places).
    """
    if raw_pm25 is None:
        return None

    if raw_pm25 < 0.0:
        return 0.0

    if relative_humidity_percent is None or relative_humidity_percent <= 30.0:
        return round(float(raw_pm25), 2)

    # Clamp RH between 30% and 100%
    rh = min(100.0, max(30.0, float(relative_humidity_percent)))

    # EPA / Barkjohn empirical growth factor model for optical sensors
    growth_factor = 1.0 + 0.24 * ((rh / 100.0) ** 2)
    calibrated = raw_pm25 / growth_factor

    logger.debug(
        "Calibrated PM2.5: raw=%.2f µg/m³, RH=%.1f%%, factor=%.3f -> corrected=%.2f µg/m³",
        raw_pm25,
        rh,
        growth_factor,
        calibrated,
    )
    return round(max(0.0, float(calibrated)), 2)
