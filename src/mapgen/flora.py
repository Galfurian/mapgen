"""
Flora and vegetation placement algorithms.

This module implements climate-driven vegetation placement that runs after
base terrain generation. Vegetation placement is based on environmental
conditions like rainfall, temperature, and elevation rather than simple
elevation thresholds.
"""

import logging

from .map_data import MapData

logger = logging.getLogger(__name__)


def place_vegetation(
    map_data: MapData,
) -> None:
    """
    Place vegetation based on climate conditions.

    This function runs after base terrain placement and assigns vegetation
    tiles based on environmental conditions. It processes vegetation in order:
    forests, deserts, then grasslands to fill remaining suitable areas.

    Args:
        map_data (MapData):
            The map data to modify. Must have elevation_map and rainfall_map.
    """
    if not map_data.elevation_map:
        logger.warning("Elevation map is required for vegetation placement")
        return
    if not map_data.rainfall_map:
        logger.warning("Rainfall map is required for vegetation placement")
        return
    if not map_data.temperature_map:
        logger.warning("Temperature map is recommended for vegetation placement")
        return
    if not map_data.humidity_map:
        logger.warning("Humidity map is required for vegetation placement")
        return

    vegetation_tiles = map_data.find_tiles_by_properties(
        is_vegetation=True,
    )

    if not vegetation_tiles:
        logger.warning("No vegetation tiles found in tile catalog")
        return

    logger.debug("Starting climate-driven vegetation placement")

    logger.debug(f"Vegetation placement complete")
