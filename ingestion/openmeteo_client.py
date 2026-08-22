"""
Open-Meteo API Client for AeroCast (SRS v1.1 Compliant).
Ingests 7-day weather forecast and historical weather metrics for Lahore coordinates.
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

logger = logging.getLogger("aerocast.openmeteo")


class OpenMeteoClient:
    """Async client for fetching weather forecast and historical records from Open-Meteo."""

    def __init__(
        self,
        forecast_url: Optional[str] = None,
        historical_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.forecast_url = forecast_url or settings.OPENMETEO_FORECAST_URL
        self.historical_url = historical_url or settings.OPENMETEO_HISTORICAL_URL
        self.timeout = timeout or settings.HTTP_TIMEOUT_SECONDS

        # Key geographic grid sampling points across Lahore (Central, North, South, East, West)
        self.lahore_grid_points = [
            {"name": "Lahore Central / Mall Road", "lat": 31.5580, "lon": 74.3250},
            {"name": "Lahore North / Shahdara", "lat": 31.6200, "lon": 74.2850},
            {"name": "Lahore South / Model Town - DHA", "lat": 31.4750, "lon": 74.3650},
            {"name": "Lahore East / Shalimar - Wagah", "lat": 31.5850, "lon": 74.4500},
            {"name": "Lahore West / Thokar Niaz Baig", "lat": 31.4700, "lon": 74.2400},
            {"name": "Lahore Airbase / Cantt", "lat": 31.5204, "lon": 74.3587},
        ]

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_BACKOFF_MIN_SECONDS, max=settings.RETRY_BACKOFF_MAX_SECONDS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute async HTTP GET request with tenacity retry policy."""
        logger.info("Requesting Open-Meteo API: url=%s, params=%s", url, params)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def fetch_current_and_forecast(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        forecast_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Fetch current weather and 7-day forecast for given or default Lahore coordinates.
        """
        target_lat = lat or settings.LAHORE_LATITUDE
        target_lon = lon or settings.LAHORE_LONGITUDE

        params = {
            "latitude": target_lat,
            "longitude": target_lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m",
            "hourly": "temperature_2m,precipitation,precipitation_probability,wind_speed_10m",
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }

        try:
            raw_data = await self._get_json(self.forecast_url, params)
            current = raw_data.get("current", {})
            hourly = raw_data.get("hourly", {})

            # Compute next 24h rainfall sum from hourly precipitation
            precip_hourly = hourly.get("precipitation", [])
            rain_24h_sum = sum(precip_hourly[:24]) if precip_hourly else current.get("precipitation", 0.0)

            return {
                "latitude": raw_data.get("latitude", target_lat),
                "longitude": raw_data.get("longitude", target_lon),
                "elevation": raw_data.get("elevation", 214.0),
                "temperature_c": current.get("temperature_2m"),
                "relative_humidity_percent": current.get("relative_humidity_2m"),
                "rainfall_mm_forecast": round(float(rain_24h_sum), 2) if rain_24h_sum is not None else 0.0,
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "surface_pressure_hpa": current.get("surface_pressure"),
                "timestamp_utc": current.get("time", datetime.now(timezone.utc).isoformat()),
                "source": "Open-Meteo",
                "is_fallback": False,
                "fallback_reason": None,
                "raw_response": raw_data,
            }
        except Exception as e:
            logger.error("Error fetching Open-Meteo forecast: %s. Using tagged synthetic fallback.", e)
            return self._generate_fallback_weather(target_lat, target_lon, reason=f"Open-Meteo error: {str(e)}")

    async def fetch_grid_weather(self) -> List[Dict[str, Any]]:
        """
        Fetch weather observations across representative grid points across Lahore.
        """
        results: List[Dict[str, Any]] = []
        for pt in self.lahore_grid_points:
            try:
                weather = await self.fetch_current_and_forecast(lat=pt["lat"], lon=pt["lon"])
                weather["grid_point_name"] = pt["name"]
                results.append(weather)
            except Exception as err:
                logger.warning("Failed grid fetch for %s: %s", pt["name"], err)
                results.append(self._generate_fallback_weather(
                    pt["lat"], pt["lon"], name=pt["name"], reason=f"Grid point fetch failure: {str(err)}"
                ))

        return results

    async def fetch_historical_weather(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fetch historical daily weather for Lahore (up to 2 years historical data, FR-INGEST-02 / FR-INGEST-10).
        """
        target_lat = lat or settings.LAHORE_LATITUDE
        target_lon = lon or settings.LAHORE_LONGITUDE

        if not end_date:
            end_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now(timezone.utc) - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

        params = {
            "latitude": target_lat,
            "longitude": target_lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "timezone": "UTC",
        }

        try:
            raw_data = await self._get_json(self.historical_url, params)
            return {
                "latitude": target_lat,
                "longitude": target_lon,
                "daily": raw_data.get("daily", {}),
                "source": "Open-Meteo-Archive",
                "is_fallback": False,
                "fallback_reason": None,
            }
        except Exception as e:
            logger.error("Error fetching historical Open-Meteo weather: %s", e)
            return {
                "latitude": target_lat,
                "longitude": target_lon,
                "daily": {},
                "source": "Open-Meteo-Archive",
                "is_fallback": True,
                "fallback_reason": f"Historical weather API fetch failed: {str(e)}",
            }

    def _generate_fallback_weather(
        self, lat: float, lon: float, name: str = "Lahore Central", reason: str = "synthetic fallback — Open-Meteo API unreachable"
    ) -> Dict[str, Any]:
        """Generate realistic weather snapshot for Lahore when API is unreachable."""
        import random
        return {
            "latitude": lat,
            "longitude": lon,
            "elevation": 214.0,
            "temperature_c": round(34.5 + random.uniform(-2.5, 3.5), 1),
            "relative_humidity_percent": round(58.0 + random.uniform(-10, 15), 1),
            "rainfall_mm_forecast": round(max(0.0, random.uniform(0.0, 12.0)), 1),
            "wind_speed_kmh": round(12.0 + random.uniform(-4, 6), 1),
            "surface_pressure_hpa": 1012.5,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "Open-Meteo-Synthetic",
            "grid_point_name": name,
            "is_fallback": True,
            "fallback_reason": reason,
        }
