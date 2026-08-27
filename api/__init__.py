"""
AeroCast Module M9: REST API Layer.
High-performance FastAPI service orchestrating spatial, predictive, and multi-hazard intelligence across Lahore.
"""

from .app import create_app

__all__ = ["create_app"]
