"""
AeroCast Module M4: Flash Flood Risk Calculation Engine.
Deterministic hydrological runoff risk scoring across Lahore's 241-zone grid.
"""

from .engine import FlashFloodScorer
from .interface import (
    get_zone_flood_risk,
    get_all_zones_flood_risk,
    get_flood_health,
)

__all__ = [
    "FlashFloodScorer",
    "get_zone_flood_risk",
    "get_all_zones_flood_risk",
    "get_flood_health",
]
