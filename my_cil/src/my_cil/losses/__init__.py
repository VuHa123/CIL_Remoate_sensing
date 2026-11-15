"""Losses and distribution-distance estimators."""

from .club import CLUB
from .sinkhorn import SinkhornDistance

__all__ = ["CLUB", "SinkhornDistance"]
