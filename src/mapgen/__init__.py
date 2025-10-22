"""Procedural fantasy map generation library."""

import logging

from .map_data import MapData, Position, Settlement, Tile, PlacementMethod
from .map_generator import generate_map, get_default_tiles
from .visualization import (
    get_ascii_map,
    plot_base_terrain,
    plot_contour_lines,
    plot_elevation_map,
    plot_map,
    plot_rainfall_map,
    plot_3d_map,
    plot_roads,
    plot_settlements,
)

__version__ = "0.1.0"

# Set up package logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if none exists.
if not logger.handlers:
    console_handler = logging.StreamHandler()
    # Create formatter.
    formatter = logging.Formatter(
        "[%(name)s] %(levelname)-6s: %(message)s"
    )
    console_handler.setFormatter(formatter)
    # Add handler to logger.
    logger.addHandler(console_handler)

__all__ = [
    "MapData",
    "generate_map",
    "PlacementMethod",
    "Position",
    "Settlement",
    "Tile",
    "get_ascii_map",
    "logger",
    "plot_base_terrain",
    "plot_contour_lines",
    "plot_elevation_map",
    "plot_map",
    "plot_rainfall_map",
    "plot_3d_map",
    "plot_roads",
    "plot_settlements",
]

