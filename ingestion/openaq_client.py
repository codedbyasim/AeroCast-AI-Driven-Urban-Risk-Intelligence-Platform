"""
OpenAQ API Client for AeroCast (SRS v1.1 Compliant).
Ingests live and historical air quality measurements (PM2.5, PM10, NO2) for Lahore.
Uses httpx for async HTTP requests and tenacity for retry with exponential backoff.
Properly tags fallback synthetic data to satisfy FR-INGEST-09 / SRS Risk R-01.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from config import settings

logger = logging.getLogger("aerocast.openaq")


class OpenAQClient:
    """Async client for fetching air quality observations from OpenAQ API v3."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key or settings.OPENAQ_API_KEY
        self.base_url = (base_url or settings.OPENAQ_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.HTTP_TIMEOUT_SECONDS

        # Reference Lahore monitoring stations
        self.known_lahore_stations = [
            {"id": "LHR-AQ-01", "name": "US Consulate Lahore / Gulberg", "lat": 31.5165, "lon": 74.3496},
            {"id": "LHR-AQ-02", "name": "Town Hall / Lower Mall", "lat": 31.5645, "lon": 74.3095},
            {"id": "LHR-AQ-03", "name": "Model Town Park Central", "lat": 31.4832, "lon": 74.3218},
            {"id": "LHR-AQ-04", "name": "Liberty Market / Gulberg III", "lat": 31.5108, "lon": 74.3441},
            {"id": "LHR-AQ-05", "name": "Jail Road EPA Punjab", "lat": 31.5420, "lon": 74.3370},
            {"id": "LHR-AQ-06", "name": "DHA Phase 5 Commercial", "lat": 31.4678, "lon": 74.4021},
            {"id": "LHR-AQ-07", "name": "Johar Town G-1 Market", "lat": 31.4697, "lon": 74.2728},
            {"id": "LHR-AQ-08", "name": "Kot Lakhpat Industrial", "lat": 31.4420, "lon": 74.3380},
            {"id": "LHR-AQ-09", "name": "Walled City / Delhi Gate", "lat": 31.5830, "lon": 74.3200},
            {"id": "LHR-AQ-10", "name": "Ravi Road Shahdara", "lat": 31.6210, "lon": 74.2930},
            {"id": "LHR-AQ-11", "name": "Wagah Border Sector", "lat": 31.6020, "lon": 74.5710},
            {"id": "LHR-AQ-12", "name": "Samanabad Roundabout", "lat": 31.5340, "lon": 74.3010},
        ]

    def _get_headers(self) -> Dict[str, str]:
        """Construct request headers including optional API key."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_MIN_SECONDS, max=settings.RETRY_BACKOFF_MAX_SECONDS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _fetch_api(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute async HTTP GET request with retries and exponential backoff."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info("Requesting OpenAQ API: url=%s, params=%s", url, params)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            return response.json()

    async def fetch_latest_measurements(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_m: int = 30000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch latest air quality measurements for Lahore.
        Queries locations and sensors around Lahore coordinates.
        Returns a list of standardized raw measurement dicts.
        """
        target_lat = lat or settings.LAHORE_LATITUDE
        target_lon = lon or settings.LAHORE_LONGITUDE
        records: List[Dict[str, Any]] = []

        try:
            if not self.api_key:
                logger.info("OpenAQ API key not configured; using tagged synthetic fallback observations.")
                return self.generate_fallback_observations()

            # Query OpenAQ v3 locations API by Lahore bounding box or coordinates
            min_lon, min_lat, max_lon, max_lat = settings.LAHORE_BBOX
            params = {
                "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                "limit": 100,
            }
            data = await self._fetch_api("locations", params)
            results = data.get("results", [])

            for loc in results:
                loc_id = str(loc.get("id"))
                loc_name = loc.get("name", "Unknown Station")
                coords = loc.get("coordinates", {})
                loc_lat = coords.get("latitude", target_lat)
                loc_lon = coords.get("longitude", target_lon)
                sensors = loc.get("sensors", [])

                pm25_val, pm10_val, no2_val = None, None, None
                last_updated = datetime.now(timezone.utc).isoformat()

                for sensor in sensors:
                    param_name = sensor.get("parameter", {}).get("name", "").lower()
                    latest = sensor.get("latest", {})
                    val = latest.get("value")
                    dt = latest.get("datetime")
                    if dt:
                        last_updated = dt

                    if "pm25" in param_name or "pm2.5" in param_name:
                        pm25_val = float(val) if val is not None else None
                    elif "pm10" in param_name:
                        pm10_val = float(val) if val is not None else None
                    elif "no2" in param_name:
                        no2_val = float(val) if val is not None else None

                if pm25_val is not None or pm10_val is not None or no2_val is not None:
                    records.append({
                        "station_id": loc_id,
                        "station_name": loc_name,
                        "latitude": loc_lat,
                        "longitude": loc_lon,
                        "pm25": pm25_val,
                        "pm10": pm10_val,
                        "no2": no2_val,
                        "timestamp_utc": last_updated,
                        "source": "OpenAQ",
                        "is_fallback": False,
                        "fallback_reason": None,
                    })

            if not records:
                logger.warning("OpenAQ API returned 0 active stations. Generating tagged fallback observations.")
                records = self.generate_fallback_observations(reason="OpenAQ API returned 0 active stations")

        except Exception as e:
            logger.error("Failed to fetch live OpenAQ data: %s. Using tagged fallback dataset.", e)
            records = self.generate_fallback_observations(reason=f"OpenAQ API error: {str(e)}")

        logger.info("Retrieved %d air quality observation records for Lahore", len(records))
        return records

    async def fetch_historical_measurements(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        days: int = 730,  # ~2 years historical data
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical air quality time series (FR-INGEST-02 / FR-INGEST-10).
        Used by M3 ML models and M8 Backtesting.
        """
        target_lat = lat or settings.LAHORE_LATITUDE
        target_lon = lon or settings.LAHORE_LONGITUDE

        if self.api_key:
            try:
                # Query historical endpoints if available
                end_dt = datetime.now(timezone.utc)
                start_dt = end_dt - timedelta(days=days)
                params = {
                    "coordinates": f"{target_lat},{target_lon}",
                    "date_from": start_dt.strftime("%Y-%m-%d"),
                    "date_to": end_dt.strftime("%Y-%m-%d"),
                    "limit": 1000,
                }
                data = await self._fetch_api("measurements", params)
                results = data.get("results", [])
                if results:
                    return results
            except Exception as e:
                logger.warning("Historical OpenAQ API query failed: %s. Generating historical series.", e)

        # Generate realistic 2-year seasonal historical time series for Lahore
        return self._generate_historical_series(days=days, lat=target_lat, lon=target_lon)

    def generate_fallback_observations(
        self, reason: str = "synthetic fallback — OpenAQ API key unconfigured / API unreachable"
    ) -> List[Dict[str, Any]]:
        """
        Generate realistic air quality observations for Lahore monitoring stations.
        Strictly tags records as fallback per FR-INGEST-09 (SRS Risk R-01).
        """
        import random
        now_iso = datetime.now(timezone.utc).isoformat()
        fallback_records = []

        # Baseline values for Lahore (seasonal smog and urban pollution baseline)
        base_pm25 = 165.0
        base_pm10 = 210.0
        base_no2 = 38.0

        for station in self.known_lahore_stations:
            jitter = (hash(station["id"]) % 30) - 15
            pm25 = round(max(25.0, base_pm25 + jitter + random.uniform(-10, 15)), 2)
            pm10 = round(max(40.0, base_pm10 + (jitter * 1.3) + random.uniform(-15, 20)), 2)
            no2 = round(max(5.0, base_no2 + (jitter * 0.4) + random.uniform(-4, 6)), 2)

            fallback_records.append({
                "station_id": station["id"],
                "station_name": station["name"],
                "latitude": station["lat"],
                "longitude": station["lon"],
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
                "timestamp_utc": now_iso,
                "source": "OpenAQ-Synthetic",
                "is_fallback": True,
                "fallback_reason": reason,
            })

        return fallback_records

    def _generate_historical_series(self, days: int, lat: float, lon: float) -> List[Dict[str, Any]]:
        """Generate structured daily historical AQI records capturing winter smog spikes."""
        import random
        history = []
        end_date = datetime.now(timezone.utc)

        for d in range(days, 0, -1):
            dt = end_date - timedelta(days=d)
            month = dt.month

            # Lahore smog season: Oct-Feb (peaks in Nov/Dec/Jan)
            if month in (11, 12, 1):
                seasonal_pm25 = 280.0 + random.uniform(-40, 80)
                seasonal_pm10 = 340.0 + random.uniform(-50, 90)
            elif month in (10, 2):
                seasonal_pm25 = 180.0 + random.uniform(-30, 50)
                seasonal_pm10 = 230.0 + random.uniform(-40, 60)
            elif month in (7, 8):  # Monsoon washout
                seasonal_pm25 = 55.0 + random.uniform(-15, 25)
                seasonal_pm10 = 90.0 + random.uniform(-20, 30)
            else:  # Spring/Summer background
                seasonal_pm25 = 110.0 + random.uniform(-25, 35)
                seasonal_pm10 = 150.0 + random.uniform(-30, 45)

            history.append({
                "date": dt.strftime("%Y-%m-%d"),
                "latitude": lat,
                "longitude": lon,
                "pm25": round(seasonal_pm25, 2),
                "pm10": round(seasonal_pm10, 2),
                "no2": round(25.0 + random.uniform(-10, 15), 2),
                "source": "OpenAQ-Historical",
                "is_fallback": True,
                "fallback_reason": "synthetic historical baseline series",
            })

        return history
