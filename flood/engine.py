"""
AeroCast Module M4: Flash Flood Risk Calculation Engine.
Deterministic hydrological runoff risk scoring combining meteorological precipitation,
Copernicus terrain slope/elevation depressions, OSM imperviousness, and antecedent wetness.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from config import settings
from spatial.covariates import CovariateManager

logger = logging.getLogger("aerocast.flood.engine")


class FlashFloodScorer:
    """
    Computes deterministic flash flood and urban waterlogging risk scores across Lahore's 241 zones.
    Aligns with Punjab Disaster Management Authority (PDMA) and WASA Lahore urban flood standards.
    """

    # Model Weights (Sum to 1.00)
    WEIGHT_PRECIPITATION = 0.40      # Forecasted 24h Rainfall Intensity
    WEIGHT_IMPERVIOUSNESS = 0.25     # Urban Concrete / Pavement Fraction
    WEIGHT_SLOPE_FLATNESS = 0.15     # Topographic Flatness / Stagnation Risk
    WEIGHT_ELEVATION_SINK = 0.10     # Localized Low-Lying Depression Index
    WEIGHT_ANTECEDENT_WETNESS = 0.10 # Antecedent Soil Moisture / 7-Day Exponential Decay Rain

    def __init__(self, covariate_manager: Optional[CovariateManager] = None):
        self.covariate_mgr = covariate_manager or CovariateManager()
        self._zone_covariates = self.covariate_mgr.get_all_covariates()
        self._historical_weather_cache: Optional[Dict[str, Any]] = None
        self._init_elevation_bounds()

    def _init_elevation_bounds(self):
        """Compute elevation bounds across all 241 zones for depression normalization."""
        elevations = [
            cov.get("elevation_m", 214.0)
            for cov in self._zone_covariates.values()
            if cov.get("elevation_m") is not None
        ]
        self.min_elevation = float(min(elevations)) if elevations else 200.0
        self.max_elevation = float(max(elevations)) if elevations else 225.0
        self.elev_range = max(1.0, self.max_elevation - self.min_elevation)

    def _get_antecedent_precipitation_7d(self, zone_id: str, precip_24h_fallback: float = 0.0) -> Tuple[float, float, bool]:
        """
        Calculates the 7-day exponential-decay Antecedent Precipitation Index (API):
        API = sum_{k=1..7} (0.85)^k * P[t-k]
        
        Returns:
            (api_value, sum_7d_rainfall_mm, is_fallback)
        """
        hist_file = Path(settings.CACHE_DIR) / "historical" / "historical_weather.json"
        if self._historical_weather_cache is None and hist_file.exists():
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    self._historical_weather_cache = json.load(f)
            except Exception as e:
                logger.warning("Failed to read historical weather for API calculation: %s", e)

        if self._historical_weather_cache and "daily" in self._historical_weather_cache:
            daily = self._historical_weather_cache["daily"]
            precip_series = daily.get("precipitation_sum", [])
            if len(precip_series) >= 7:
                # Take the most recent 7 complete days prior to current
                past_7 = precip_series[-7:]
                # API = sum_{k=1..7} (0.85)^k * P[t-k] (where k=1 is yesterday, k=7 is 7 days ago)
                # past_7[-1] is yesterday (k=1), past_7[-7] is 7 days ago (k=7)
                api_val = sum((0.85 ** k) * float(past_7[-k] or 0.0) for k in range(1, 8))
                raw_sum = sum(float(p or 0.0) for p in past_7)
                return float(api_val), float(raw_sum), False

        # Fallback to same-day proxy if historical archive is missing
        proxy_api = precip_24h_fallback * 0.4
        return float(proxy_api), float(precip_24h_fallback), True

    def calculate_zone_flood_risk(
        self,
        zone_id: str,
        horizon_hours: int = 24,
        current_weather: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate deterministic flash flood risk score for a specific zone.

        :param zone_id: Unique Zone ID (e.g. 'ZONE-LHR-0075')
        :param horizon_hours: Prediction horizon in hours (default: 24)
        :param current_weather: Optional weather dict with precipitation_sum (mm), rain intensity, etc.
        :return: Comprehensive flood risk dictionary with component breakdown and advisories.
        """
        cov = self._zone_covariates.get(zone_id, {})
        if not cov:
            logger.warning("Zone %s not found in CovariateManager. Using urban default covariates.", zone_id)
            cov = {
                "impervious_surface_ratio": 0.50,
                "elevation_m": 214.0,
                "slope_percent": 1.5,
                "road_density_km_per_sqkm": 5.0,
                "population_density_per_sqkm": 8000.0,
            }

        # 1. Weather Inputs (Precipitation in mm)
        is_fallback_weather = False
        fallback_notes = []

        if current_weather is not None and "precipitation_sum" in current_weather:
            precip_24h = float(current_weather.get("precipitation_sum") or 0.0)
            precip_intensity = float(current_weather.get("precipitation_intensity_max", precip_24h / 12.0) or 0.0)
            if "antecedent_precipitation_3d" in current_weather or "antecedent_precipitation_index" in current_weather:
                api_val = float(current_weather.get("antecedent_precipitation_index", current_weather.get("antecedent_precipitation_3d", 0.0)) or 0.0)
                raw_7d = api_val
                is_fallback_api = False
            else:
                api_val, raw_7d, is_fallback_api = self._get_antecedent_precipitation_7d(zone_id, precip_24h)
        else:
            # Pull live data from M1 cache via ingestion.interface.get_latest_data()
            from ingestion.interface import get_latest_data
            live_rec = get_latest_data(zone_id)
            if live_rec and live_rec.get("metrics"):
                metrics = live_rec["metrics"]
                raw_precip = metrics.get("rainfall_mm_forecast")
                if raw_precip is not None:
                    precip_24h = float(raw_precip)
                else:
                    precip_24h = 0.0
                    is_fallback_weather = True
                    fallback_notes.append("No live rainfall_mm_forecast in M1 cache; defaulted to 0.0mm")
                
                # Check data quality notes
                dq = live_rec.get("data_quality", {})
                if dq.get("stale"):
                    is_fallback_weather = True
                    fallback_notes.append("M1 cached telemetry is stale")
            else:
                precip_24h = 0.0
                is_fallback_weather = True
                fallback_notes.append("No live M1 record found for zone; defaulted to 0.0mm")

            precip_intensity = precip_24h / 12.0 if precip_24h > 0 else 0.0
            api_val, raw_7d, is_fallback_api = self._get_antecedent_precipitation_7d(zone_id, precip_24h)

        if is_fallback_api:
            fallback_notes.append("Antecedent precipitation used same-day proxy (historical archive unavailable)")

        # 2. Terrain & Land-Cover Covariates
        impervious = float(cov.get("impervious_surface_ratio", 0.50))
        slope = float(cov.get("slope_percent", 1.5))
        elevation = float(cov.get("elevation_m", 214.0))
        road_density = float(cov.get("road_density_km_per_sqkm", 5.0))

        # 3. Component Normalization Functions
        # A. Precipitation Intensity Factor: 0mm -> 0.0, 50mm -> 0.625, >=80mm -> 1.0
        precip_score = float(np.clip(precip_24h / 80.0, 0.0, 1.0))
        if precip_intensity > 25.0:  # Cloudburst boost (> 25mm/hr)
            precip_score = min(1.0, precip_score + 0.15)

        # B. Impervious Surface Factor: High concrete/paving -> high runoff
        impervious_score = float(np.clip(impervious, 0.0, 1.0))

        # C. Slope Inversion / Flatness: Flat areas (slope < 1.0%) trap standing water
        slope_flatness_score = float(np.clip(1.0 - (slope / 4.0), 0.0, 1.0))

        # D. Topographic Depression (Basin Sink): Lower elevation relative to Lahore range
        elevation_depression_score = float(np.clip(
            1.0 - ((elevation - self.min_elevation) / self.elev_range),
            0.0, 1.0
        ))

        # E. 7-Day Antecedent Soil Saturation (API): Saturated soil reduces infiltration
        # Normalized against 50.0mm cumulative index
        soil_saturation_score = float(np.clip(api_val / 50.0, 0.0, 1.0))

        # 4. Composite Deterministic Flood Risk Score
        raw_risk = (
            self.WEIGHT_PRECIPITATION * precip_score
            + self.WEIGHT_IMPERVIOUSNESS * impervious_score
            + self.WEIGHT_SLOPE_FLATNESS * slope_flatness_score
            + self.WEIGHT_ELEVATION_SINK * elevation_depression_score
            + self.WEIGHT_ANTECEDENT_WETNESS * soil_saturation_score
        )

        flood_risk_score = round(float(np.clip(raw_risk, 0.0, 1.0)), 3)

        # 5. Risk Categorization (NDMA / PDMA Standards)
        if flood_risk_score < 0.25:
            risk_category = "Low"
            alert_level = "GREEN"
            expected_inundation = "None (< 2 cm surface water)"
            advisory = "Routine drainage capacity adequate. Normal vehicular traffic flow expected."
        elif flood_risk_score < 0.50:
            risk_category = "Moderate"
            alert_level = "YELLOW"
            expected_inundation = "Minor Puddling (2 - 8 cm on curbs)"
            advisory = "Caution for low-lying pedestrian crossings and unpaved shoulders."
        elif flood_risk_score < 0.75:
            risk_category = "High"
            alert_level = "ORANGE"
            expected_inundation = "Localized Waterlogging (8 - 25 cm street inundation)"
            advisory = "WASA dewatering teams on standby. Expect traffic delays at underpasses and key intersections."
        else:
            risk_category = "Severe"
            alert_level = "RED"
            expected_inundation = "Critical Flash Flooding (> 25 cm deep waterlogging)"
            advisory = "EMERGENCY WASA DEWATERING REQUIRED. High risk of underpass submergence and commercial basement flooding."

        return {
            "zone_id": zone_id,
            "horizon_hours": horizon_hours,
            "flood_risk_score": flood_risk_score,
            "risk_category": risk_category,
            "alert_level": alert_level,
            "expected_inundation_depth": expected_inundation,
            "actionable_advisory": advisory,
            "component_breakdown": {
                "forecasted_precipitation_24h_mm": round(precip_24h, 1),
                "antecedent_precipitation_7d_mm": round(raw_7d, 1),
                "antecedent_precipitation_index": round(api_val, 2),
                "precipitation_risk_contribution": round(self.WEIGHT_PRECIPITATION * precip_score, 3),
                "impervious_surface_contribution": round(self.WEIGHT_IMPERVIOUSNESS * impervious_score, 3),
                "slope_flatness_contribution": round(self.WEIGHT_SLOPE_FLATNESS * slope_flatness_score, 3),
                "elevation_depression_contribution": round(self.WEIGHT_ELEVATION_SINK * elevation_depression_score, 3),
                "antecedent_saturation_contribution": round(self.WEIGHT_ANTECEDENT_WETNESS * soil_saturation_score, 3),
            },
            "terrain_context": {
                "impervious_surface_ratio": round(impervious, 3),
                "slope_percent": round(slope, 2),
                "elevation_m": round(elevation, 1),
                "road_density_km_per_sqkm": round(road_density, 2),
            },
            "data_quality": {
                "is_fallback_weather": is_fallback_weather,
                "is_fallback_antecedent": is_fallback_api,
                "notes": "; ".join(fallback_notes) if fallback_notes else "Live M1 weather and historical 7-day API active",
            },
        }

    def calculate_all_zones_flood_risk(
        self,
        horizon_hours: int = 24,
        weather_by_zone: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute flash flood risk across all 241 canonical zones in Lahore."""
        results = {}
        for zone_id in self._zone_covariates.keys():
            wx = weather_by_zone.get(zone_id) if weather_by_zone else None
            results[zone_id] = self.calculate_zone_flood_risk(
                zone_id, horizon_hours=horizon_hours, current_weather=wx
            )
        return results
