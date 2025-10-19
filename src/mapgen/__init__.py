"""Procedural fantasy map generation library."""

import logging

__version__ = "0.1.0"

# Set up package logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

from .map_data import MapData, Position, Settlement, Tile
from .map_generator import MapGenerator

__all__ = [
    "MapData",
    "MapGenerator",
    "Position",
    "Settlement",
    "Tile",
    "logger",
]

