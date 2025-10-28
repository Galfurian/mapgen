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

from .map_data import MapData, PlacementMethod, Tile
from .utils import generate_noise_grid

logger = logging.getLogger(__name__)


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


def apply_terrain_features(
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

    # Filter and sort terrain-based tiles by priority.
    terrain_tiles = [
        tile
        for tile in map_data.tiles
        if tile.placement_method == PlacementMethod.TERRAIN_BASED
    ]

    if not terrain_tiles:
        logger.warning("No terrain-based tiles found in tile catalog")
        return

    terrain_tiles.sort(key=lambda t: t.terrain_priority, reverse=True)

    logger.debug(f"Applying terrain features using {len(terrain_tiles)} terrain tiles")

    tiles_assigned = 0
    # Iterate over each position on the map and assign a suitable tile.
    for y in range(map_data.height):
        for x in range(map_data.width):
            if _apply_suitable_tile(map_data, terrain_tiles, x, y):
                tiles_assigned += 1
            else:
                logger.warning(
                    f"No suitable tile found for elevation {map_data.get_elevation(x, y):.3f} at ({x}, {y})"
                )

    logger.debug(
        f"Assigned terrain tiles to {tiles_assigned}/{map_data.width * map_data.height} positions"
    )


def apply_base_terrain(
    map_data: MapData,
    base_terrain_tiles: list[Tile],
) -> None:
    """
    Apply base terrain tiles based on elevation map.

    This function assigns only base terrain tiles (sea, coast, plains,
    mountains) to each position on the map based on elevation values. It does
    NOT place vegetation - that is handled separately by the flora module.

    Base terrain forms the foundation of the map and represents the underlying
    geological/topographical features before vegetation and other features are
    added.

    Args:
        map_data (MapData):
            The map data to modify. Must have an elevation_map.
        base_terrain_tiles (list[Tile]):
            List of base terrain tiles to use for placement. These should only
            include elevation-driven tiles like sea, coast, plains, and
            mountains.

    Raises:
        ValueError:
            If elevation_map is not available in map_data or if no suitable
            tiles are provided.

    """
    if not map_data.elevation_map:
        raise ValueError("Elevation map is required for base terrain assignment")

    if not base_terrain_tiles:
        raise ValueError("No base terrain tiles provided")

    # Sort terrain tiles by priority.
    terrain_tiles = sorted(
        base_terrain_tiles, key=lambda t: t.terrain_priority, reverse=True
    )

    logger.debug(f"Applying base terrain using {len(terrain_tiles)} tiles")

    tiles_assigned = 0
    # Iterate over each position on the map and assign a suitable tile.
    for y in range(map_data.height):
        for x in range(map_data.width):
            if _apply_suitable_tile(map_data, terrain_tiles, x, y):
                tiles_assigned += 1
            else:
                logger.warning(
                    f"No suitable base terrain tile found for elevation {map_data.get_elevation(x, y):.3f} at ({x}, {y})"
                )

    logger.debug(
        f"Assigned base terrain to {tiles_assigned}/{map_data.width * map_data.height} positions"
    )


def classify_water_bodies(
    map_data: MapData,
    max_lake_size: int | None = None,
) -> None:
    """
    Classify water bodies as either salt water (seas/oceans) or fresh water (lakes).

    This function identifies connected water regions and classifies them based on:
    1. Connection to map edges
    2. Size threshold for edge-connected bodies

    Classification rules:
    - Isolated water bodies (not touching edges): always become lakes
    - Edge-connected water bodies:
      - ≤ max_lake_size: become lakes (fresh water)
      - > max_lake_size: remain as seas (salt water)

    Args:
        map_data (MapData):
            The map data containing terrain information.
        max_lake_size (int | None):
            Maximum size for edge-connected water bodies to be classified as lakes.
            Larger edge-connected bodies remain as salt water seas.
            If None, automatically calculated as 0.5% of map area (minimum 50).

    """
    if not map_data.elevation_map:
        logger.warning("No elevation map available for water body classification")
        return

    height, width = len(map_data.elevation_map), len(map_data.elevation_map[0])

    # Calculate max_lake_size if not provided
    if max_lake_size is None:
        # Use 1% of total map area as threshold, minimum 100
        max_lake_size = max(100, int(width * height * 0.01))

    # Create a binary mask of water tiles
    water_mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            tile = map_data.get_terrain(x, y)
            water_mask[y, x] = tile.is_water

    # Find connected components (water bodies)
    labeled_mask, num_features = ndimage.label(water_mask)

    logger.debug(f"Found {num_features} water bodies")

    # Analyze each water body
    for label in range(1, num_features + 1):
        # Get all positions in this water body
        body_positions = np.where(labeled_mask == label)
        body_size = len(body_positions[0])

        # Check if this body touches any map edge
        touches_edge = (
            np.min(body_positions[0]) == 0  # top edge
            or np.max(body_positions[0]) == height - 1  # bottom edge
            or np.min(body_positions[1]) == 0  # left edge
            or np.max(body_positions[1]) == width - 1  # right edge
        )

        # Classify water bodies:
        # - Isolated water bodies: always become lakes
        # - Edge-connected water bodies:
        #   - ≤ max_lake_size: lakes (fresh water)
        #   - > max_lake_size: seas (salt water)
        if not touches_edge:
            # Any isolated water body becomes a lake
            is_lake = True
        else:
            # Edge-connected water bodies: smaller ones become lakes
            is_lake = body_size <= max_lake_size

        if is_lake:
            # Convert to fresh water (lake)
            lake_tile = None
            for tile in map_data.tiles:
                if tile.name == "lake":
                    lake_tile = tile
                    break

            if lake_tile is None:
                logger.warning(
                    "Lake tile not found in tile collection, skipping lake conversion"
                )
            else:
                for pos_idx in range(len(body_positions[0])):
                    y, x = body_positions[0][pos_idx], body_positions[1][pos_idx]
                    map_data.set_terrain(x, y, lake_tile)

            logger.debug(
                f"Classified water body {label} as lake (size: {body_size} tiles)"
            )
        else:
            classification = "sea" if body_size > max_lake_size else "lake"
            logger.debug(
                f"Water body {label} remains as {classification} (size: {body_size} tiles)"
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
        return False

    # Select the highest priority suitable tile.
    chosen_tile = suitable_tiles[0]
    # Assign the chosen tile to the position.
    map_data.set_terrain(x, y, chosen_tile)
    return True
