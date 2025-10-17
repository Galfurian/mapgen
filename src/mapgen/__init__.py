"""Procedural fantasy map generation library."""

__version__ = "0.1.0"

from .level import Level, Position
from .map_generator import MapGenerator

__all__ = ["Level", "Position", "MapGenerator"]
