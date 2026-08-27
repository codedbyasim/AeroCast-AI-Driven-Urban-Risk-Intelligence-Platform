"""
AeroCast — Geostatistical Kriging & Spatial Interpolation Engine (M2)
====================================================================
Implements Ordinary Kriging and Universal Kriging (with NDVI & Road Density drift)
across the 241-zone metric grid of Lahore District (SRS v1.1 Section 2.2 / FR-SPATIAL-01-03).
Provides mathematically grounded spatial uncertainty (kriging variance) and
automatic IDW fallback when direct control points are sparse (< 4 points).
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from ingestion.schema import NormalizedRecord, Metrics
from ingestion.cache import LocalDataCache
from .covariates import CovariateManager
from .calibration import calibrate_pm25_optical

logger = logging.getLogger("aerocast.spatial.kriging")


class KrigingEngine:
    """Geostatistical interpolation engine for 241-zone environmental risk intelligence."""

    def __init__(
        self,
        cache: Optional[LocalDataCache] = None,
        covariate_manager: Optional[CovariateManager] = None,
    ):
        self.cache = cache or LocalDataCache()
        self.covariate_manager = covariate_manager or CovariateManager()

    def interpolate_variable(
        self,
        variable: str,
        records_map: Optional[Dict[str, NormalizedRecord]] = None,
        variogram_model: str = "spherical",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform spatial interpolation for a specific environmental metric across all 241 zones.

        :param variable: Attribute name from Metrics model (e.g. 'aqi_pm25', 'temperature_c', etc.)
        :param records_map: Optional pre-loaded mapping of zone_id -> NormalizedRecord.
        :param variogram_model: Variogram model type ('spherical', 'gaussian', 'exponential', 'linear').
        :return: Mapping of zone_id -> {value, variance, confidence_score, method, is_direct_sensor, notes}
        """
        if records_map is None:
            records_map = self.cache.get_all_latest()

        covariates = self.covariate_manager.get_all_covariates()
        is_synthetic_covariates = self.covariate_manager.is_synthetic_ndvi()

        # 1. Collect control points (training points with non-null values)
        train_lons: List[float] = []
        train_lats: List[float] = []
        train_vals: List[float] = []
        train_ndvi: List[float] = []
        train_roads: List[float] = []
        direct_zone_ids = set()
        direct_zone_values: Dict[str, float] = {}

        for z_id, record in records_map.items():
            metrics = record.metrics
            val = getattr(metrics, variable, None)

            if val is not None and not np.isnan(val):
                c_lon = record.spatial_context.centroid_lon
                c_lat = record.spatial_context.centroid_lat

                if c_lon is None or c_lat is None:
                    cov = covariates.get(z_id, {})
                    c_lon = cov.get("centroid_lon")
                    c_lat = cov.get("centroid_lat")

                if c_lon is not None and c_lat is not None:
                    # Apply EPA optical sensor humidity calibration for PM2.5
                    if variable == "aqi_pm25":
                        rh = metrics.relative_humidity_percent
                        val = calibrate_pm25_optical(val, rh)

                    train_lons.append(float(c_lon))
                    train_lats.append(float(c_lat))
                    train_vals.append(float(val))

                    # Track whether this was a direct sensor reading
                    if not record.data_quality.interpolated:
                        direct_zone_ids.add(z_id)
                        direct_zone_values[z_id] = float(val)

                    cov = covariates.get(z_id, {})
                    train_ndvi.append(float(cov.get("ndvi_index", 0.25)))
                    train_roads.append(float(cov.get("road_density_km_per_sqkm", 4.5)))

        # 2. Collect all target points (all 241 zone centroids)
        target_zone_ids: List[str] = []
        target_lons: List[float] = []
        target_lats: List[float] = []
        target_ndvi: List[float] = []
        target_roads: List[float] = []

        for z_id, cov in covariates.items():
            target_zone_ids.append(z_id)
            target_lons.append(float(cov["centroid_lon"]))
            target_lats.append(float(cov["centroid_lat"]))
            target_ndvi.append(float(cov.get("ndvi_index", 0.25)))
            target_roads.append(float(cov.get("road_density_km_per_sqkm", 4.5)))

        num_train = len(train_vals)
        logger.info("Interpolating '%s' across %d zones with %d control points", variable, len(target_zone_ids), num_train)

        # 3. Handle edge cases & execute Kriging or Fallback
        results: Dict[str, Dict[str, Any]] = {}

        if num_train == 0:
            logger.warning("No control points found for '%s'. Returning baseline defaults.", variable)
            return self._generate_empty_fallback(target_zone_ids, variable, covariates)

        # Case A: < 4 points -> Execute IDW fallback
        if num_train < 4:
            logger.info("Control points (%d) < 4 for '%s'. Executing IDW fallback.", num_train, variable)
            pred_vals, pred_vars = self._execute_idw(
                train_lons, train_lats, train_vals, target_lons, target_lats
            )
            method = "idw_fallback"
            sample_var = float(np.var(train_vals)) if num_train > 1 else 1.0

        # Case B: PM2.5 with Universal Kriging (NDVI & Road Density drift)
        elif variable == "aqi_pm25":
            pred_vals, pred_vars, method = self._execute_universal_or_ordinary_kriging(
                train_lons,
                train_lats,
                train_vals,
                target_lons,
                target_lats,
                train_ndvi,
                train_roads,
                target_ndvi,
                target_roads,
                variogram_model,
            )
            sample_var = max(1.0, float(np.var(train_vals)))

        # Case C: Other variables with Ordinary Kriging
        else:
            pred_vals, pred_vars, method = self._execute_ordinary_kriging(
                train_lons, train_lats, train_vals, target_lons, target_lats, variogram_model
            )
            sample_var = max(1.0, float(np.var(train_vals)))

        # 4. Determine variance bounds across target grid for continuous confidence scaling
        var_min = float(np.min(pred_vars)) if len(pred_vars) > 0 else 0.0
        var_max = float(np.max(pred_vars)) if len(pred_vars) > 0 else 1.0

        # 5. Construct output dictionary with statistical confidence calculation (FR-SPATIAL-03)
        for i, z_id in enumerate(target_zone_ids):
            is_direct = z_id in direct_zone_ids

            if is_direct and z_id in direct_zone_values:
                # Direct ground-truth sensor anchor: preserve exact measured reading and zero spatial variance
                val = round(float(direct_zone_values[z_id]), 2)
                variance = 0.0
                conf = 0.95
                note = "Direct physical sensor observation (calibrated)"
            else:
                val = round(float(pred_vals[i]), 2)
                variance = round(float(pred_vars[i]), 4)
                conf = self._calculate_confidence_score(
                    variance=variance,
                    var_min=var_min,
                    var_max=var_max,
                    is_direct=False,
                    is_synthetic_covariates=is_synthetic_covariates,
                    method=method,
                )
                if method == "universal_kriging":
                    note = "Universal Kriging with Sentinel-2 NDVI & OSM road density spatial drift"
                elif method == "ordinary_kriging":
                    note = f"Ordinary Kriging ({variogram_model} variogram) over {num_train} control points"
                else:
                    note = f"IDW fallback interpolation ({num_train} sparse control points)"

            # Apply synthetic raster provenance penalty
            if is_synthetic_covariates:
                conf = min(0.75, conf)
                note += "; spatial covariates derived from placeholder rasters"

            results[z_id] = {
                "zone_id": z_id,
                "variable": variable,
                "value": val,
                "variance": variance,
                "confidence_score": conf,
                "spatial_confidence": conf,
                "method": "direct_sensor" if is_direct else method,
                "is_direct_sensor": is_direct,
                "control_points_used": num_train,
                "notes": note,
            }

        return results

    def interpolate_all_variables(
        self,
        records_map: Optional[Dict[str, NormalizedRecord]] = None,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Interpolate all environmental and weather variables across all 241 zones.

        :return: Nested dictionary: zone_id -> variable -> {value, variance, confidence_score, ...}
        """
        if variables is None:
            variables = [
                "aqi_pm25",
                "aqi_pm10",
                "no2_ppb",
                "temperature_c",
                "relative_humidity_percent",
                "wind_speed_kmh",
                "rainfall_mm_forecast",
                "surface_pressure_hpa",
            ]

        if records_map is None:
            records_map = self.cache.get_all_latest()

        all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for var_name in variables:
            var_grid = self.interpolate_variable(var_name, records_map=records_map)
            for z_id, res in var_grid.items():
                if z_id not in all_results:
                    all_results[z_id] = {}
                all_results[z_id][var_name] = res

        return all_results

    def _execute_universal_or_ordinary_kriging(
        self,
        x_train: List[float],
        y_train: List[float],
        z_train: List[float],
        x_target: List[float],
        y_target: List[float],
        ndvi_train: List[float],
        roads_train: List[float],
        ndvi_target: List[float],
        roads_target: List[float],
        variogram_model: str,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Attempt Universal Kriging with external drift; fallback to Ordinary Kriging if degenerate."""
        z_arr = np.array(z_train, dtype=float)
        # Robust outlier handling: clip extreme localized sensor anomalies for variogram stability
        if len(z_arr) >= 6:
            q25, q75 = np.percentile(z_arr, 25), np.percentile(z_arr, 75)
            iqr = q75 - q25
            upper_bound = q75 + 2.5 * iqr
            lower_bound = max(0.0, q25 - 2.5 * iqr)
            fit_z = np.clip(z_arr, lower_bound, upper_bound)
        else:
            fit_z = z_arr

        try:
            uk = UniversalKriging(
                x_train,
                y_train,
                fit_z,
                variogram_model=variogram_model,
                drift_terms=["specified"],
                specified_drift=[np.array(ndvi_train), np.array(roads_train)],
                verbose=False,
                enable_plotting=False,
            )
            z_pred, ss_pred = uk.execute(
                "points",
                np.array(x_target),
                np.array(y_target),
                specified_drift_arrays=[np.array(ndvi_target), np.array(roads_target)],
            )
            return z_pred, ss_pred, "universal_kriging"
        except Exception as e:
            logger.warning("Universal Kriging drift fitting failed (%s). Falling back to Ordinary Kriging.", e)
            return self._execute_ordinary_kriging(x_train, y_train, z_train, x_target, y_target, variogram_model)

    def _execute_ordinary_kriging(
        self,
        x_train: List[float],
        y_train: List[float],
        z_train: List[float],
        x_target: List[float],
        y_target: List[float],
        variogram_model: str,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Execute Ordinary Kriging using PyKrige."""
        z_arr = np.array(z_train, dtype=float)
        # Robust outlier handling for variogram stability
        if len(z_arr) >= 6:
            q25, q75 = np.percentile(z_arr, 25), np.percentile(z_arr, 75)
            iqr = q75 - q25
            upper_bound = q75 + 2.5 * iqr
            lower_bound = max(0.0, q25 - 2.5 * iqr)
            fit_z = np.clip(z_arr, lower_bound, upper_bound)
        else:
            fit_z = z_arr

        try:
            ok = OrdinaryKriging(
                x_train,
                y_train,
                fit_z,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
            )
            z_pred, ss_pred = ok.execute("points", np.array(x_target), np.array(y_target))
            return z_pred, ss_pred, "ordinary_kriging"
        except Exception as e:
            logger.warning("Ordinary Kriging failed (%s). Falling back to IDW.", e)
            z_pred, ss_pred = self._execute_idw(x_train, y_train, z_train, x_target, y_target)
            return z_pred, ss_pred, "idw_fallback"

    def _execute_idw(
        self,
        x_train: List[float],
        y_train: List[float],
        z_train: List[float],
        x_target: List[float],
        y_target: List[float],
        power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute Inverse Distance Weighting (IDW) interpolation."""
        x_tr = np.array(x_train)
        y_tr = np.array(y_train)
        z_tr = np.array(z_train)

        preds = []
        variances = []
        sample_var = float(np.var(z_tr)) if len(z_tr) > 1 else 1.0

        for xt, yt in zip(x_target, y_target):
            dist = np.sqrt((x_tr - xt) ** 2 + (y_tr - yt) ** 2)
            # If point is directly on a control point
            min_idx = np.argmin(dist)
            if dist[min_idx] < 1e-6:
                preds.append(z_tr[min_idx])
                variances.append(0.0)
            else:
                weights = 1.0 / (dist ** power)
                w_sum = np.sum(weights)
                val = np.sum(weights * z_tr) / w_sum
                # Variance increases with distance to nearest neighbor
                var = sample_var * min(2.0, dist[min_idx] / 0.05)
                preds.append(val)
                variances.append(var)

        return np.array(preds), np.array(variances)

    def _calculate_confidence_score(
        self,
        variance: float,
        var_min: float,
        var_max: float,
        is_direct: bool,
        is_synthetic_covariates: bool,
        method: str,
    ) -> float:
        """
        Calculate statistically grounded confidence score in [0.0, 1.0] (FR-SPATIAL-03).
        Ensures continuous, non-saturating score distribution across spatial variance gradient.
        """
        # 1. Continuous variance scaling relative to empirical prediction grid variance
        var_span = max(1e-4, var_max - var_min)
        rel_var = min(1.0, max(0.0, (variance - var_min) / var_span)) if var_max > var_min else 0.0

        # Base confidence: 0.88 for minimum variance down to 0.43 for maximum variance
        base_confidence = 0.88 - 0.45 * rel_var

        # Direct sensor calibration anchor
        if is_direct:
            score = 0.95
        else:
            score = base_confidence * 0.90

        # Penalize IDW fallback
        if method == "idw_fallback":
            score = min(0.55, score * 0.70)

        # Penalize synthetic placeholder covariates
        if is_synthetic_covariates:
            score = min(0.75, score)

        return round(max(0.20, min(1.0, float(score))), 2)

    def _generate_empty_fallback(
        self,
        zone_ids: List[str],
        variable: str,
        covariates: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Fallback when zero control points exist for a variable."""
        defaults = {
            "aqi_pm25": 95.0,
            "aqi_pm10": 160.0,
            "no2_ppb": 28.0,
            "temperature_c": 26.0,
            "relative_humidity_percent": 65.0,
            "wind_speed_kmh": 10.0,
            "rainfall_mm_forecast": 0.0,
            "surface_pressure_hpa": 1012.0,
        }
        fallback_val = defaults.get(variable, 0.0)
        res = {}
        for z_id in zone_ids:
            res[z_id] = {
                "zone_id": z_id,
                "variable": variable,
                "value": fallback_val,
                "variance": 1.0,
                "confidence_score": 0.30,
                "spatial_confidence": 0.30,
                "method": "zero_control_points_fallback",
                "is_direct_sensor": False,
                "control_points_used": 0,
                "notes": f"No ground truth data available for {variable} — served baseline heuristic",
            }
        return res
