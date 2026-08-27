"""
AeroCast Module M7: Alert & Notification Dispatcher Interface.
Public facade for alert evaluation, dispatching, subscriptions, and diagnostic health checks.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .models import HazardAlert, AlertSubscription, DispatchResult
from .dispatcher import AlertDispatcher

_DISPATCHER: Optional[AlertDispatcher] = None


def _get_dispatcher() -> AlertDispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = AlertDispatcher()
    return _DISPATCHER


def evaluate_and_dispatch_alerts(force_reevaluate: bool = False) -> Dict[str, Any]:
    """
    Triggers end-to-end multi-hazard threshold evaluation across all 241 zones
    and dispatches newly triggered alerts to all active subscriber channels.
    """
    dispatcher = _get_dispatcher()
    new_alerts = dispatcher.evaluate_all_hazards(force_reevaluate=force_reevaluate)
    dispatches = dispatcher.dispatch_alerts(new_alerts)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_new_alerts_triggered": len(new_alerts),
        "total_dispatches_sent": len(dispatches),
        "alerts": [a.model_dump() for a in new_alerts],
        "dispatches": [d.model_dump() for d in dispatches],
    }


def get_active_alerts() -> List[Dict[str, Any]]:
    """Get all currently active non-expired multi-hazard alerts."""
    dispatcher = _get_dispatcher()
    return [a.model_dump() for a in dispatcher.get_active_alerts()]


def get_alert_history() -> Dict[str, Any]:
    """Get complete alert journal including active and past alerts."""
    dispatcher = _get_dispatcher()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_active": len(dispatcher.get_active_alerts()),
        "active_alerts": [a.model_dump() for a in dispatcher.get_active_alerts()],
        "registered_subscribers": len(dispatcher._subscriptions),
    }


def register_subscription(subscription_data: Dict[str, Any]) -> Dict[str, Any]:
    """Register a new stakeholder subscription."""
    dispatcher = _get_dispatcher()
    sub = AlertSubscription(**subscription_data)
    registered = dispatcher.register_subscription(sub)
    return registered.model_dump()


def get_alerts_health() -> Dict[str, Any]:
    """Return Module M7 diagnostic health status."""
    dispatcher = _get_dispatcher()
    return {
        "status": "healthy",
        "module": "M7_Alert_Dispatcher",
        "total_active_alerts": len(dispatcher.get_active_alerts()),
        "registered_subscribers_count": len(dispatcher._subscriptions),
        "cooldown_hours": dispatcher.COOLDOWN_HOURS,
        "supported_channels": ["webhook", "sms", "email", "dashboard"],
        "supported_languages": ["en", "ur", "roman_ur"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
