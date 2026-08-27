"""
AeroCast Alert & Notification Data Models (Pydantic v2).
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class HazardType(str, Enum):
    SMOG = "smog"
    FLASH_FLOOD = "flash_flood"
    HEAT_ISLAND = "heat_island"
    MULTI_HAZARD = "multi_hazard"


class SeverityLevel(str, Enum):
    ADVISORY = "ADVISORY"    # Informational / minor risk
    WATCH = "WATCH"          # Elevated conditions (Orange)
    WARNING = "WARNING"      # High danger threshold (Red)
    EMERGENCY = "EMERGENCY"  # Critical public safety alert (Maroon/Flash)


class AlertChannel(str, Enum):
    WEBHOOK = "webhook"
    SMS = "sms"
    EMAIL = "email"
    DASHBOARD = "dashboard"


class HazardAlert(BaseModel):
    """Normalized multi-hazard emergency alert record."""
    alert_id: str
    zone_id: str
    zone_name: Optional[str] = None
    hazard_type: HazardType
    severity: SeverityLevel
    title: str
    trigger_metric: str
    trigger_value: float
    threshold_value: float
    unit: str
    messages: Dict[str, str] = Field(
        default_factory=dict,
        description="Multi-lingual text: {'en': '...', 'ur': '...', 'roman_ur': '...'}"
    )
    actionable_instructions: Dict[str, str] = Field(
        default_factory=dict,
        description="Civic & Authority directives: {'citizens': '...', 'authorities': '...'}"
    )
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at_utc: str
    is_active: bool = True
    cooldown_key: str


class AlertSubscription(BaseModel):
    """Subscriber registration model for Webhook, SMS, or Email alerts."""
    subscription_id: str
    agency_name: str
    channel: AlertChannel
    target: str = Field(..., description="Webhook URL, Phone Number (+92...), or Email")
    subscribed_zones: List[str] = Field(default_factory=lambda: ["ALL"], description="List of Zone IDs or ['ALL']")
    subscribed_hazards: List[HazardType] = Field(
        default_factory=lambda: [HazardType.SMOG, HazardType.FLASH_FLOOD, HazardType.HEAT_ISLAND]
    )
    min_severity: SeverityLevel = SeverityLevel.WATCH
    is_active: bool = True
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DispatchResult(BaseModel):
    """Delivery confirmation record for an alert dispatch action."""
    dispatch_id: str
    alert_id: str
    zone_id: str
    channel: AlertChannel
    target: str
    status: str = Field(..., description="'DELIVERED', 'FAILED', 'COOLDOWN_SUPPRESSED'")
    message_snippet: str
    delivery_latency_ms: float = 0.0
    simulated: bool = Field(True, description="Indicates whether this dispatch transmission is simulated")
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
