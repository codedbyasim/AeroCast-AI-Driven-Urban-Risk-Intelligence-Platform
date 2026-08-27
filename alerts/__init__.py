"""
AeroCast Module M7: Early Warning Alert & Notification Dispatcher.
Multi-hazard automated threshold monitoring, multi-lingual alerting, and multi-channel dispatching.
"""

from .models import HazardAlert, AlertSubscription, DispatchResult, HazardType, SeverityLevel, AlertChannel
from .dispatcher import AlertDispatcher
from .interface import (
    evaluate_and_dispatch_alerts,
    get_active_alerts,
    get_alert_history,
    register_subscription,
    get_alerts_health,
)

__all__ = [
    "HazardAlert",
    "AlertSubscription",
    "DispatchResult",
    "HazardType",
    "SeverityLevel",
    "AlertChannel",
    "AlertDispatcher",
    "evaluate_and_dispatch_alerts",
    "get_active_alerts",
    "get_alert_history",
    "register_subscription",
    "get_alerts_health",
]
