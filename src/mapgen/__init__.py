"""Procedural fantasy map generation library."""

__version__ = "0.1.0"

from .map_data import MapData, Position, Settlement, Tile
from .map_generator import MapGenerator

__all__ = [
    "MapData",
    "Position",
    "Settlement",
    "Tile",
    "MapGenerator",
]
