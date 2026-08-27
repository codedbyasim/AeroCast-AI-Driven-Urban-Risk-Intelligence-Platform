"""
AeroCast Urban Heat Island (UHI) Risk Scoring Engine (Module M3).
Computes a hyper-local thermal risk score per Zone by combining surface temperature,
vegetation cooling index (Sentinel-2 NDVI), impervious surface ratio, and population exposure.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import numpy as np

from config import settings
from spatial.covariates import CovariateManager

logger = logging.getLogger("aerocast.ml.uhi")


class HeatIslandScorer:
    """
    Computes zonal Urban Heat Island (UHI) risk scores [0.0 - 1.0].
    
    Formula Rationale:
    ------------------
    - Temperature Weight (w_temp = 0.35): Thermal intensity normalized against Lahore summer baseline.
    - Impervious Surface Weight (w_imp = 0.25): High thermal inertia of asphalt/concrete surfaces.
    - Population Exposure Weight (w_pop = 0.20): Citizen exposure density in urban core.
    - Vegetation Cooling Weight (w_ndvi = 0.20): Evapotranspirative cooling from green canopies.
    """

    # Model Weights
    W_TEMP: float = 0.35
    W_IMPERVIOUS: float = 0.25
    W_POPULATION: float = 0.20
    W_NDVI: float = 0.20

    # Physical Reference Baselines for Lahore
    TEMP_MIN_REF_C: float = 25.0   # Baseline night/rural temp
    TEMP_MAX_REF_C: float = 45.0   # Severe heatwave peak
    POP_MAX_REF: float = 25000.0   # Max urban density per km²

    def __init__(self, covariate_manager: Optional[CovariateManager] = None):
        self.covariate_manager = covariate_manager or CovariateManager()
        self._covariates = self.covariate_manager.get_all_covariates()

    def score_zone(
        self,
        zone_id: str,
        temperature_c: Optional[float] = None,
        is_synthetic_weather: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate UHI risk score [0.0 - 1.0] and risk category for a specific zone.
        """
        cov = self._covariates.get(zone_id, {
            "impervious_surface_ratio": 0.65,
            "population_density_per_sqkm": 12500.0,
            "ndvi_index": 0.22,
            "elevation_m": 214.0,
        })

        if temperature_c is None:
            from ingestion.interface import get_latest_data
            live_data = get_latest_data(zone_id)
            if live_data and live_data.get("metrics", {}).get("temperature_c") is not None:
                temperature_c = float(live_data["metrics"]["temperature_c"])
            else:
                temperature_c = 34.5  # Lahore summer daytime baseline

        # 1. Temperature Term [0, 1]
        temp_norm = np.clip((temperature_c - self.TEMP_MIN_REF_C) / (self.TEMP_MAX_REF_C - self.TEMP_MIN_REF_C), 0.0, 1.0)

        # 2. Impervious Surface Term [0, 1]
        imp_ratio = float(cov.get("impervious_surface_ratio", 0.65))

        # 3. Population Density Exposure Term [0, 1]
        pop_density = float(cov.get("population_density_per_sqkm", 12500.0))
        pop_norm = np.clip(pop_density / self.POP_MAX_REF, 0.0, 1.0)

        # 4. Vegetation Cooling Term [0, 1] (NDVI: higher greenery reduces heat risk)
        ndvi = float(cov.get("ndvi_index", 0.22))
        ndvi_norm = np.clip((ndvi + 0.1) / 0.8, 0.0, 1.0)

        # Composite UHI Risk Formulation
        raw_score = (
            (self.W_TEMP * temp_norm)
            + (self.W_IMPERVIOUS * imp_ratio)
            + (self.W_POPULATION * pop_norm)
            - (self.W_NDVI * ndvi_norm)
        )
        # Rescale into [0.0, 1.0]
        uhi_risk_score = round(float(np.clip(raw_score, 0.0, 1.0)), 3)

        # Risk Classification
        if uhi_risk_score < 0.30:
            risk_category = "Low"
        elif uhi_risk_score < 0.55:
            risk_category = "Moderate"
        elif uhi_risk_score < 0.75:
            risk_category = "High"
        else:
            risk_category = "Severe (Urban Heat Island Hotspot)"

        # Confidence Score based on data provenance
        is_synthetic_ndvi = self.covariate_manager.is_synthetic_ndvi()
        base_conf = 0.90
        if is_synthetic_weather:
            base_conf -= 0.15
        if is_synthetic_ndvi:
            base_conf -= 0.15

        return {
            "zone_id": zone_id,
            "uhi_risk_score": uhi_risk_score,
            "heat_island_risk_score": uhi_risk_score,
            "heat_island_score": uhi_risk_score,
            "risk_category": risk_category,
            "temperature_c": round(float(temperature_c), 1),
            "ndvi_index": round(ndvi, 3),
            "impervious_surface_ratio": round(imp_ratio, 2),
            "population_density_per_sqkm": round(pop_density, 1),
            "confidence_score": round(base_conf, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def score_all_zones(self, temperature_grid: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compute UHI risk scores for all 241 Lahore zones.
        """
        results = {}
        for z_id in self._covariates.keys():
            t_val = temperature_grid.get(z_id) if temperature_grid else None
            results[z_id] = self.score_zone(z_id, temperature_c=t_val)
        return results
