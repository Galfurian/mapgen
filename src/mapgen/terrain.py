"""
Terrain generation utilities for procedural map generation.

This module provides functions for generating and processing elevation maps,
applying terrain features, and assigning base terrain tiles based on elevation
data. It handles the core terrain generation logic for creating realistic
topographical maps.
"""

import logging

import numpy as np
from scipy import ndimage

from .map_data import MapData, Tile
from .utils import generate_noise_grid

logger = logging.getLogger(__name__)


def initialize_map_data(
    map_data: MapData,
    width: int,
    height: int,
) -> None:
    """
    Initialize the map data dimensions and padding.

    Args:
        map_data (MapData):
            The map data to initialize.
        width (int):
            The width of the map.
        height (int):
            The height of the map.
    """
    # Get the base terrain tiles.
    base_tiles = map_data.find_tiles_by_properties(is_base_tile=True)
    if not base_tiles:
        logger.warning("No base terrain tiles found in tile catalog")
        return
    # Sort the base tiles by terrain priority.
    base_tiles.sort(key=lambda t: t.terrain_priority)
    # Pick the highest priority base tile.
    base_tile = base_tiles[0]
    logger.debug(
        f"Initializing map with base tile: {base_tile.name} (ID: {base_tile.id})"
    )
    # Initialize the grid with the base tile ID.
    map_data.grid = [[base_tile.id for _ in range(width)] for _ in range(height)]


def generate_elevation_map(
    map_data: MapData,
    scale: float = 50.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> None:
    """
    Generate an elevation map using Perlin noise.

    Args:
        map_data (MapData):
            The map data to store the elevation map in.
        scale (float):
            The scale of the noise.
        octaves (int):
            The number of octaves.
        persistence (float):
            The persistence value.
        lacunarity (float):
            The lacunarity value.

    """
    elevation_map = generate_noise_grid(
        width=map_data.width,
        height=map_data.height,
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        base=0,
    )

    # Normalize the elevation map to the range -1 to 1.
    min_val = np.min(elevation_map)
    max_val = np.max(elevation_map)
    if max_val > min_val:
        elevation_map = (elevation_map - min_val) / (max_val - min_val) * 2 - 1

    # Adjust the elevation map based on sea level to control land/sea ratio.
    if map_data.sea_level != 0.0:
        shifted = elevation_map - map_data.sea_level
        sea_mask = shifted < 0
        land_mask = shifted >= 0

        # Adjust sea areas to the range -1 to 0.
        if np.any(sea_mask):
            sea_min = np.min(shifted[sea_mask])
            sea_max = 0.0
            if sea_min != sea_max:
                shifted[sea_mask] = -1 + (shifted[sea_mask] - sea_min) / (
                    sea_max - sea_min
                )
            else:
                shifted[sea_mask] = -1

        # Adjust land areas to the range 0 to 1.
        if np.any(land_mask):
            land_min = 0.0
            land_max = np.max(shifted[land_mask])
            if land_min != land_max:
                shifted[land_mask] = (shifted[land_mask] - land_min) / (
                    land_max - land_min
                )
            else:
                shifted[land_mask] = 0

        elevation_map = shifted

    # Round the elevation values to 4 decimal places for precision.
    elevation_map = np.round(elevation_map, decimals=4)

    # Store the processed elevation map in the map data.
    map_data.elevation_map = elevation_map.tolist()


def smooth_elevation_map(
    map_data: MapData,
    iterations: int = 1,
    sigma: float = 0.5,
) -> None:
    """
    Smooth the elevation map using Gaussian filtering.

    This function applies Gaussian blur to the elevation map to create smoother
    terrain transitions. This should be called BEFORE assigning terrain tiles,
    as the tiles are determined by elevation values.

    Args:
        map_data (MapData):
            The map data containing the elevation map to smooth.
        iterations (int):
            Number of smoothing passes to apply.
        sigma (float):
            Standard deviation for Gaussian kernel. Higher values create more
            smoothing. Typical values: 0.3-1.0. Default 0.5 provides subtle
            smoothing that removes noise while preserving terrain features.

    Raises:
        ValueError:
            If elevation_map is not available in map_data.

    """
    if not map_data.elevation_map:
        raise ValueError("Elevation map is required for smoothing")

    elevation = np.array(map_data.elevation_map)

    # Apply Gaussian smoothing for the specified number of iterations.
    for _ in range(iterations):
        elevation = ndimage.gaussian_filter(elevation, sigma=sigma)

    map_data.elevation_map = elevation.tolist()
    logger.debug(f"Smoothed elevation map with {iterations} iterations (sigma={sigma})")


def apply_base_terrain(
    map_data: MapData,
) -> None:
    """
    Apply terrain features based on elevation map.

    This function assigns terrain tiles to each position on the map based on
    elevation values and tile suitability criteria. Only tiles with
    TERRAIN_BASED placement method are considered for assignment.

    Args:
        map_data (MapData):
            The map data to modify. Must have an elevation_map.

    Raises:
        ValueError:
            If elevation_map is not available in map_data.

    """
    if not map_data.elevation_map:
        raise ValueError("Elevation map is required for terrain feature assignment")

    # Get the base terrain tiles.
    tiles = map_data.find_tiles_by_properties(is_base_tile=True)
    if not tiles:
        logger.warning("No base terrain tiles found in tile catalog")
        return

    # Sort the base tiles by terrain priority, highest priority first.
    tiles.sort(key=lambda t: t.terrain_priority, reverse=True)

    logger.debug(f"Applying terrain features using {len(tiles)} terrain tiles")

    tiles_assigned = 0

    # Iterate over each position on the map and assign a suitable tile.
    for y in range(map_data.height):
        for x in range(map_data.width):
            tiles_assigned += _apply_suitable_tile(map_data, tiles, x, y)

    logger.debug(
        f"Assigned terrain tiles to {tiles_assigned}/{map_data.width * map_data.height} positions"
    )


def _apply_suitable_tile(
    map_data: MapData,
    terrain_tiles: list[Tile],
    x: int,
    y: int,
) -> bool:
    """
    Apply the most suitable terrain tile for the given position.

    Tiles are selected based on elevation range compatibility and terrain
    priority. The highest priority tile that matches the elevation range is
    chosen.

    Args:
        map_data (MapData):
            The map data containing elevation information.
        terrain_tiles (list[Tile]):
            Pre-filtered list of terrain-based tiles, sorted by priority
            descending.
        x (int):
            The x coordinate of the position.
        y (int):
            The y coordinate of the position.

    Returns:
        bool:
            True if a suitable tile was assigned, False otherwise.

    """
    # Get the elevation at the current position.
    elevation = map_data.get_elevation(x, y)

    # Find tiles that match the elevation range.
    suitable_tiles = [
        tile
        for tile in terrain_tiles
        if tile.elevation_min <= elevation <= tile.elevation_max
    ]

    # If no tiles match, return False.
    if not suitable_tiles:
        logger.warning(
            f"No suitable terrain tile for elevation {elevation:.3f} at ({x}, {y})"
        )
        return False

    # Select the highest priority suitable tile.
    chosen_tile = suitable_tiles[0]
    # Assign the chosen tile to the position.
    map_data.set_terrain(x, y, chosen_tile)
    return True
