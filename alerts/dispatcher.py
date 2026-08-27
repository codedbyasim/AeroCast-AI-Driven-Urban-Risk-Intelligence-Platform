"""
AeroCast Module M7: Core Alert Dispatcher & Multi-Channel Engine.
Evaluates multi-hazard rules across all 241 zones, manages cooldowns, and dispatches alerts.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

from config import settings
from .models import (
    HazardAlert, AlertSubscription, DispatchResult,
    HazardType, SeverityLevel, AlertChannel
)
from .templates import format_smog_alert, format_flood_alert, format_heat_alert

logger = logging.getLogger("aerocast.alerts")


class AlertDispatcher:
    """
    Evaluates multi-hazard thresholds across all 241 zones, handles deduplication / cooldowns,
    and dispatches structured notifications to registered agencies and citizen channels.
    """

    # Cooldown duration before re-alerting the exact same hazard in the same zone
    COOLDOWN_HOURS = 6

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR) / "alerts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.cache_dir / "alert_history.json"
        self.subscriptions_file = self.cache_dir / "subscriptions.json"
        
        self._active_alerts: Dict[str, HazardAlert] = {}
        self._cooldowns: Dict[str, datetime] = {}  # "ZONE-0001:smog" -> datetime
        self._subscriptions: List[AlertSubscription] = []
        
        self._load_subscriptions()
        self._load_history()

    def _load_subscriptions(self):
        """Load registered agency and channel subscribers, or initialize default mock channels."""
        if self.subscriptions_file.exists():
            try:
                with open(self.subscriptions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._subscriptions = [AlertSubscription(**item) for item in data]
                return
            except Exception as e:
                logger.warning("Failed to load alert subscriptions: %s", e)

        # Default Stakeholder Subscriptions (Illustrative Placeholders for Demonstration)
        # Note: These URLs and numbers are placeholder examples and mock targets for simulated dispatch,
        # not live production government webhook integrations.
        self._subscriptions = [
            AlertSubscription(
                subscription_id="SUB-PDMA-01",
                agency_name="Punjab Disaster Management Authority (PDMA)",
                channel=AlertChannel.WEBHOOK,
                target="https://api.pdma.punjab.gov.pk/v1/early-warning/webhooks",
                subscribed_zones=["ALL"],
                subscribed_hazards=[HazardType.SMOG, HazardType.FLASH_FLOOD, HazardType.HEAT_ISLAND],
                min_severity=SeverityLevel.WATCH,
            ),
            AlertSubscription(
                subscription_id="SUB-WASA-01",
                agency_name="Water and Sanitation Agency (WASA Lahore)",
                channel=AlertChannel.WEBHOOK,
                target="https://wasa.punjab.gov.pk/api/drainage/alerts",
                subscribed_zones=["ALL"],
                subscribed_hazards=[HazardType.FLASH_FLOOD],
                min_severity=SeverityLevel.WATCH,
            ),
            AlertSubscription(
                subscription_id="SUB-EPA-01",
                agency_name="Environmental Protection Agency (EPA Punjab)",
                channel=AlertChannel.WEBHOOK,
                target="https://epd.punjab.gov.pk/api/v1/anti-smog/telemetry",
                subscribed_zones=["ALL"],
                subscribed_hazards=[HazardType.SMOG],
                min_severity=SeverityLevel.WATCH,
            ),
            AlertSubscription(
                subscription_id="SUB-RESCUE-01",
                agency_name="Punjab Emergency Service (Rescue 1122)",
                channel=AlertChannel.SMS,
                target="+923001122000",
                subscribed_zones=["ALL"],
                subscribed_hazards=[HazardType.SMOG, HazardType.FLASH_FLOOD],
                min_severity=SeverityLevel.WARNING,
            ),
        ]
        self._save_subscriptions()

    def _save_subscriptions(self):
        try:
            with open(self.subscriptions_file, "w", encoding="utf-8") as f:
                json.dump([s.model_dump() for s in self._subscriptions], f, indent=2)
        except Exception as e:
            logger.error("Failed to save subscriptions: %s", e)

    def _load_history(self):
        """Load past alert journal with strict zone deduplication."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("active_alerts", []):
                    alert = HazardAlert(**item)
                    # Check if not expired
                    exp = datetime.fromisoformat(alert.expires_at_utc)
                    if datetime.now(timezone.utc) < exp:
                        self._active_alerts[alert.cooldown_key] = alert
                        self._cooldowns[alert.cooldown_key] = datetime.fromisoformat(alert.timestamp_utc)
            except Exception as e:
                logger.warning("Failed to load alert history: %s", e)

    def _save_history(self, new_dispatches: Optional[List[DispatchResult]] = None):
        try:
            payload = {
                "last_updated_utc": datetime.now(timezone.utc).isoformat(),
                "total_active_alerts": len(self._active_alerts),
                "active_alerts": [a.model_dump() for a in self._active_alerts.values()],
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.error("Failed to save alert history: %s", e)

    def evaluate_all_hazards(
        self,
        force_reevaluate: bool = False
    ) -> List[HazardAlert]:
        """
        Scan all 241 zones across Smog (M3), Flash Flood (M4), and UHI (M3).
        Generates strictly deduplicated HazardAlert objects (1 per zone per hazard type).
        """
        from ml.interface import get_all_aqi_forecasts, get_all_heat_island_risk
        from flood.interface import get_all_zones_flood_risk

        forecasts = get_all_aqi_forecasts(horizon_hours=24, allow_cache=True)
        uhi_scores = get_all_heat_island_risk(allow_cache=True)
        flood_scores = get_all_zones_flood_risk(horizon_hours=24, allow_cache=True)

        new_alerts: List[HazardAlert] = []
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(hours=24)).isoformat()

        # 1. Evaluate Smog / PM2.5 Forecast Thresholds
        for zid, fc in forecasts.items():
            pred = float(fc.get("forecasted_pm25") or 0.0)
            curr = float(fc.get("current_pm25") or 65.0)
            ci = fc.get("uncertainty_interval_80") or [pred - 15, pred + 15]

            if pred >= 100.0:
                severity = SeverityLevel.EMERGENCY if pred >= 150.0 else SeverityLevel.WARNING
                cd_key = f"{zid}:smog"

                if not self._is_in_cooldown(cd_key) or force_reevaluate:
                    alert_id = f"ALT-SMG-{zid}"
                    msgs = format_smog_alert(
                        zone_id=zid,
                        zone_name=f"Zone {zid}",
                        forecast_pm25=pred,
                        current_pm25=curr,
                        ci_range=ci,
                        severity=severity,
                    )
                    alert = HazardAlert(
                        alert_id=alert_id,
                        zone_id=zid,
                        zone_name=f"Zone {zid}",
                        hazard_type=HazardType.SMOG,
                        severity=severity,
                        title=f"24h Severe Smog Spike ({pred:.1f} µg/m³)" if severity == SeverityLevel.EMERGENCY else f"24h High Pollution Alert ({pred:.1f} µg/m³)",
                        trigger_metric="forecasted_pm25_24h",
                        trigger_value=pred,
                        threshold_value=150.0 if severity == SeverityLevel.EMERGENCY else 100.0,
                        unit="µg/m³",
                        messages=msgs,
                        actionable_instructions={
                            "citizens": "Wear N95 masks, avoid outdoor morning exercise, keep windows closed.",
                            "authorities": "Deploy EPA anti-smog squads, restrict industrial emissions, enforce dust control."
                        },
                        expires_at_utc=expires,
                        cooldown_key=cd_key,
                    )
                    new_alerts.append(alert)
                    self._active_alerts[cd_key] = alert
                    self._cooldowns[cd_key] = now

        # 2. Evaluate Flash Flood Thresholds
        for zid, fl in flood_scores.items():
            f_score = float(fl.get("flood_risk_score") or 0.0)
            precip = float(fl.get("component_breakdown", {}).get("forecasted_precipitation_24h_mm") or 0.0)
            inund = fl.get("expected_inundation_depth") or "None"

            if f_score >= 0.50:
                severity = SeverityLevel.EMERGENCY if f_score >= 0.75 else SeverityLevel.WATCH
                cd_key = f"{zid}:flood"

                if not self._is_in_cooldown(cd_key) or force_reevaluate:
                    alert_id = f"ALT-FLD-{zid}"
                    msgs = format_flood_alert(
                        zone_id=zid,
                        zone_name=f"Zone {zid}",
                        flood_score=f_score,
                        precip_mm=precip,
                        inundation_desc=inund,
                        severity=severity,
                    )
                    alert = HazardAlert(
                        alert_id=alert_id,
                        zone_id=zid,
                        zone_name=f"Zone {zid}",
                        hazard_type=HazardType.FLASH_FLOOD,
                        severity=severity,
                        title=f"Critical Urban Inundation Warning ({f_score:.2f}/1.00)" if severity == SeverityLevel.EMERGENCY else f"Localized Waterlogging Watch ({f_score:.2f}/1.00)",
                        trigger_metric="flood_risk_score",
                        trigger_value=f_score,
                        threshold_value=0.75 if severity == SeverityLevel.EMERGENCY else 0.50,
                        unit="score",
                        messages=msgs,
                        actionable_instructions={
                            "citizens": "Avoid low-lying underpasses, secure electrical appliances, call WASA 1334.",
                            "authorities": "Deploy mobile dewatering suction pumps, clear choked stormwater drains."
                        },
                        expires_at_utc=expires,
                        cooldown_key=cd_key,
                    )
                    new_alerts.append(alert)
                    self._active_alerts[cd_key] = alert
                    self._cooldowns[cd_key] = now

        # 3. Evaluate Urban Heat Island Thresholds
        for zid, uhi in uhi_scores.items():
            u_score = float(uhi.get("heat_island_risk_score") or 0.0)
            if u_score >= 0.65:
                cd_key = f"{zid}:heat"
                if not self._is_in_cooldown(cd_key) or force_reevaluate:
                    alert_id = f"ALT-UHI-{zid}"
                    msgs = format_heat_alert(
                        zone_id=zid,
                        zone_name=f"Zone {zid}",
                        uhi_score=u_score,
                        temp_c=36.5,
                        severity=SeverityLevel.WATCH,
                    )
                    alert = HazardAlert(
                        alert_id=alert_id,
                        zone_id=zid,
                        zone_name=f"Zone {zid}",
                        hazard_type=HazardType.HEAT_ISLAND,
                        severity=SeverityLevel.WATCH,
                        title=f"Elevated Urban Heat Island Vulnerability ({u_score:.2f}/1.00)",
                        trigger_metric="heat_island_risk_score",
                        trigger_value=u_score,
                        threshold_value=0.65,
                        unit="score",
                        messages=msgs,
                        actionable_instructions={
                            "citizens": "Stay hydrated, avoid direct sun exposure between 11 AM - 4 PM.",
                            "authorities": "Activate mist cooling stations and municipal water sprinklers."
                        },
                        expires_at_utc=expires,
                        cooldown_key=cd_key,
                    )
                    new_alerts.append(alert)
                    self._active_alerts[cd_key] = alert
                    self._cooldowns[cd_key] = now

        self._save_history()
        return new_alerts

    def _is_in_cooldown(self, cooldown_key: str) -> bool:
        """Check if hazard alert is currently within deduplication cooldown."""
        if cooldown_key in self._cooldowns:
            last_sent = self._cooldowns[cooldown_key]
            if datetime.now(timezone.utc) - last_sent < timedelta(hours=self.COOLDOWN_HOURS):
                return True
        return False

    def dispatch_alerts(self, alerts: List[HazardAlert]) -> List[DispatchResult]:
        """Dispatch a batch of generated alerts to all matching registered subscriber channels."""
        dispatch_results: List[DispatchResult] = []

        for alert in alerts:
            for sub in self._subscriptions:
                if not sub.is_active:
                    continue

                # Check zone filter
                if "ALL" not in sub.subscribed_zones and alert.zone_id not in sub.subscribed_zones:
                    continue

                # Check hazard type filter
                if alert.hazard_type not in sub.subscribed_hazards:
                    continue

                # Check severity filter
                severity_ranks = {
                    SeverityLevel.ADVISORY: 1,
                    SeverityLevel.WATCH: 2,
                    SeverityLevel.WARNING: 3,
                    SeverityLevel.EMERGENCY: 4,
                }
                if severity_ranks.get(alert.severity, 1) < severity_ranks.get(sub.min_severity, 2):
                    continue

                # Simulate channel dispatch with latency tracking
                result = self._dispatch_to_channel(alert, sub)
                dispatch_results.append(result)

        return dispatch_results

    def _dispatch_to_channel(self, alert: HazardAlert, subscription: AlertSubscription) -> DispatchResult:
        """Simulate high-speed multi-channel dispatch delivery."""
        import time
        start = time.time()
        
        # Message selection based on subscriber channel
        msg_text = alert.messages.get("en", alert.title)

        # Mock transmission success
        latency = round((time.time() - start) * 1000 + 12.5, 2)
        
        logger.info(
            "DISPATCHED [%s] to %s (%s) for %s: %s",
            subscription.channel.value.upper(),
            subscription.agency_name,
            subscription.target,
            alert.zone_id,
            alert.title
        )

        return DispatchResult(
            dispatch_id=f"DSP-{uuid.uuid4().hex[:8].upper()}",
            alert_id=alert.alert_id,
            zone_id=alert.zone_id,
            channel=subscription.channel,
            target=subscription.target,
            status="DELIVERED",
            message_snippet=msg_text[:120] + "...",
            delivery_latency_ms=latency,
            simulated=True,
        )

    def get_active_alerts(self) -> List[HazardAlert]:
        """Return list of currently active non-expired alerts."""
        now = datetime.now(timezone.utc)
        active = []
        for a in self._active_alerts.values():
            if datetime.fromisoformat(a.expires_at_utc) > now and a.is_active:
                active.append(a)
        # Sort by severity
        severity_order = {SeverityLevel.EMERGENCY: 0, SeverityLevel.WARNING: 1, SeverityLevel.WATCH: 2, SeverityLevel.ADVISORY: 3}
        active.sort(key=lambda x: severity_order.get(x.severity, 4))
        return active

    def register_subscription(self, sub: AlertSubscription) -> AlertSubscription:
        """Register a new stakeholder webhook / SMS alert subscription."""
        self._subscriptions.append(sub)
        self._save_subscriptions()
        return sub
