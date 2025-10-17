"""Procedural fantasy map generation library."""

__version__ = "0.1.0"

from .map import Map, Position, Settlement
from .map_generator import MapGenerator

__all__ = ["Map", "MapGenerator", "Position", "Settlement"]
