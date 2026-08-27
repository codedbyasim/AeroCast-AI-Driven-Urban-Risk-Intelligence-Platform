"""
AeroCast Alert & Notification REST API Routes.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from datetime import datetime, timezone

from alerts.interface import (
    evaluate_and_dispatch_alerts,
    get_active_alerts,
    get_alert_history,
    register_subscription,
    get_alerts_health,
)
from alerts.models import AlertSubscription, SeverityLevel, HazardType

router = APIRouter(prefix="/api/v1/alerts", tags=["Early Warning & Alerts"])


@router.get("/active", summary="Get Active Multi-Hazard Alerts")
def list_active_alerts() -> Dict[str, Any]:
    """
    Returns all currently active non-expired emergency early warning alerts
    across Lahore District (Smog, Flash Flood, Heat Island).
    """
    try:
        active = get_active_alerts()
        return {
            "total_active_alerts": len(active),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "alerts": active,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch active alerts: {e}")


@router.get("/history", summary="Get Alert History Journal")
def list_alert_history() -> Dict[str, Any]:
    """
    Returns historical alert journal and subscriber counts.
    """
    try:
        return get_alert_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch alert history: {e}")


@router.post("/dispatch", summary="Trigger Multi-Hazard Alert Evaluation & Simulated Dispatch")
def trigger_alert_dispatch(
    force_reevaluate: bool = Query(False, description="Bypass 6-hour deduplication cooldown if True"),
) -> Dict[str, Any]:
    """
    Synchronously triggers multi-hazard rule evaluation across all 241 zones,
    generates multi-lingual warnings, and runs simulated dispatch delivery to registered stakeholder channels.
    Note: Real webhook and SMS transmission is currently simulated (mock delivery with latency tracking).
    """
    try:
        result = evaluate_and_dispatch_alerts(force_reevaluate=force_reevaluate)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert dispatch failed: {e}")


@router.post("/subscribe", summary="Register Stakeholder Alert Subscription")
def create_subscription(
    subscription: AlertSubscription = Body(...),
) -> Dict[str, Any]:
    """
    Registers a new government agency or citizen subscriber for automated Webhook, SMS, or Email alerts.
    """
    try:
        registered = register_subscription(subscription.model_dump())
        return {
            "status": "success",
            "message": f"Successfully registered alert subscription for '{subscription.agency_name}'",
            "subscription": registered,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Subscription registration failed: {e}")


@router.get("/health", summary="Module M7 Alert Engine Health")
def get_alerts_engine_health() -> Dict[str, Any]:
    """Return Module M7 diagnostic health status."""
    return get_alerts_health()
