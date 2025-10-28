"""Procedural fantasy map generation library."""

import logging

from . import flora
from .map_data import (
    MapData,
    Position,
    Road,
    Settlement,
    Tile,
    WaterRoute,
    get_default_tile_collections,
)
from .map_generator import generate_map
from .visualization import (
    get_ascii_layer,
    get_ascii_map,
    plot_3d_map,
    plot_map,
    plot_map_layer,
)

__version__ = "0.1.0"

# Set up package logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if none exists.
if not logger.handlers:
    console_handler = logging.StreamHandler()
    # Create formatter.
    formatter = logging.Formatter("%(name)-22s %(levelname)-8s : %(message)s")
    console_handler.setFormatter(formatter)
    # Add handler to logger.
    logger.addHandler(console_handler)

__all__ = [
    "MapData",
    "Position",
    "Road",
    "Settlement",
    "Tile",
    "WaterRoute",
    "flora",
    "generate_map",
    "get_ascii_layer",
    "get_ascii_map",
    "get_default_tile_collections",
    "logger",
    "plot_3d_map",
    "plot_map",
    "plot_map_layer",
]
