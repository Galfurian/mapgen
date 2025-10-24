"""
Flora and vegetation placement algorithms.

This module implements climate-driven vegetation placement that runs after
base terrain generation. Vegetation placement is based on environmental
conditions like rainfall, temperature, and elevation rather than simple
elevation thresholds.
"""

import logging
import random

import numpy as np

from .map_data import MapData, Tile

logger = logging.getLogger(__name__)


def place_vegetation(
    map_data: MapData,
    vegetation_tiles: list[Tile],
    forest_coverage: float = 0.15,
    desert_coverage: float = 0.10,
    rainfall_threshold_forest: float = 0.6,
    rainfall_threshold_desert: float = 0.3,
    elevation_min_forest: float = 0.05,
    elevation_max_forest: float = 0.6,
) -> None:
    """
    Place vegetation based on climate conditions.

    This function runs after base terrain placement and assigns vegetation
    tiles based on environmental conditions. It processes vegetation in order:
    forests, deserts, then grasslands to fill remaining suitable areas.

    Args:
        map_data (MapData):
            The map data to modify. Must have elevation_map and rainfall_map.
        vegetation_tiles (list[Tile]):
            List of vegetation tiles available for placement.
        forest_coverage (float):
            Target coverage ratio for forests (0.0 to 1.0).
        desert_coverage (float):
            Target coverage ratio for deserts (0.0 to 1.0).
        rainfall_threshold_forest (float):
            Minimum rainfall for forest placement (0.0 to 1.0).
        rainfall_threshold_desert (float):
            Maximum rainfall for desert placement (0.0 to 1.0).
        elevation_min_forest (float):
            Minimum elevation for forests (-1.0 to 1.0).
        elevation_max_forest (float):
            Maximum elevation for forests (-1.0 to 1.0).

    Raises:
        ValueError:
            If required maps (elevation, rainfall) are not available.

    """
    if not map_data.elevation_map:
        raise ValueError("Elevation map is required for vegetation placement")
    if not map_data.rainfall_map:
        raise ValueError("Rainfall map is required for vegetation placement")

    logger.debug("Starting climate-driven vegetation placement")

    # Find vegetation tiles by name
    forest_tile = _find_tile_by_name(vegetation_tiles, "forest")
    desert_tile = _find_tile_by_name(vegetation_tiles, "desert")
    grassland_tile = _find_tile_by_name(vegetation_tiles, "grassland")

    tiles_placed = 0

    # Phase 1: Place forests in high-rainfall areas
    if forest_tile:
        logger.debug("Placing forests based on rainfall and elevation...")
        forest_count = _place_forests(
            map_data,
            forest_tile,
            coverage=forest_coverage,
            rainfall_threshold=rainfall_threshold_forest,
            elevation_min=elevation_min_forest,
            elevation_max=elevation_max_forest,
        )
        tiles_placed += forest_count
        logger.debug(f"Placed {forest_count} forest tiles")

    # Phase 2: Place deserts in low-rainfall areas
    if desert_tile:
        logger.debug("Placing deserts in arid regions...")
        desert_count = _place_deserts(
            map_data,
            desert_tile,
            coverage=desert_coverage,
            rainfall_threshold=rainfall_threshold_desert,
        )
        tiles_placed += desert_count
        logger.debug(f"Placed {desert_count} desert tiles")

    # Phase 3: Place grasslands in moderate conditions
    if grassland_tile:
        logger.debug("Placing grasslands in moderate regions...")
        grassland_count = _place_grasslands(
            map_data,
            grassland_tile,
        )
        tiles_placed += grassland_count
        logger.debug(f"Placed {grassland_count} grassland tiles")

    logger.debug(f"Vegetation placement complete: {tiles_placed} total tiles placed")


def _place_forests(
    map_data: MapData,
    forest_tile: Tile,
    coverage: float = 0.15,
    rainfall_threshold: float = 0.6,
    elevation_min: float = 0.05,
    elevation_max: float = 0.6,
) -> int:
    """
    Place forests using seed-based growth algorithm.

    Forests are placed in areas with high rainfall and suitable elevation.
    The algorithm uses seed points in optimal locations and grows forests
    outward using a spreading mechanism.

    Args:
        map_data (MapData):
            The map data to modify.
        forest_tile (Tile):
            The forest tile to place.
        coverage (float):
            Target forest coverage ratio (0.0 to 1.0).
        rainfall_threshold (float):
            Minimum rainfall for forest placement.
        elevation_min (float):
            Minimum elevation for forests.
        elevation_max (float):
            Maximum elevation for forests.

    Returns:
        int:
            Number of forest tiles placed.

    """
    width = map_data.width
    height = map_data.height
    target_tiles = int(width * height * coverage)
    tiles_placed = 0

    # Create suitability map for forests
    suitability = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            current_tile = map_data.get_terrain(x, y)
            
            # Only place forests on base terrain that can host vegetation
            if not current_tile.can_host_vegetation:
                continue

            elevation = map_data.get_elevation(x, y)
            rainfall = map_data.get_rainfall(x, y)

            # Calculate suitability based on rainfall and elevation
            # Forests prefer moderate-to-high elevation with high rainfall
            if elevation_min <= elevation <= elevation_max:
                if rainfall >= rainfall_threshold:
                    # Strongly prefer elevations between 0.15 and 0.45
                    # Use a bell curve centered at 0.3
                    elevation_factor = max(0.0, 1.0 - ((elevation - 0.3) / 0.25) ** 2)
                    suitability[y, x] = rainfall * elevation_factor

    # Find seed points (highest suitability)
    num_seeds = max(1, target_tiles // 50)
    seed_positions = []

    for _ in range(num_seeds):
        if suitability.max() <= 0:
            break
        y_max, x_max = np.unravel_index(suitability.argmax(), suitability.shape)
        seed_positions.append((x_max, y_max))
        # Zero out nearby area to spread seeds
        y_start = max(0, y_max - 10)
        y_end = min(height, y_max + 10)
        x_start = max(0, x_max - 10)
        x_end = min(width, x_max + 10)
        suitability[y_start:y_end, x_start:x_end] = 0

    # Grow forests from seeds using BFS-like spreading
    active_positions = list(seed_positions)

    while tiles_placed < target_tiles and active_positions:
        x, y = active_positions.pop(random.randint(0, len(active_positions) - 1))

        current_tile = map_data.get_terrain(x, y)
        
        # Only place forests on base terrain that can host vegetation
        if not current_tile.can_host_vegetation:
            continue

        elevation = map_data.get_elevation(x, y)
        rainfall = map_data.get_rainfall(x, y)

        # Check if this position is suitable for forest
        if elevation_min <= elevation <= elevation_max and rainfall >= rainfall_threshold:
            map_data.set_terrain(x, y, forest_tile)
            tiles_placed += 1

            # Add neighbors to active positions for spreading
            for neighbor in map_data.get_neighbors(x, y, walkable_only=False):
                neighbor_tile = map_data.get_terrain(neighbor.x, neighbor.y)
                
                # Only spread to tiles that can host vegetation
                if (
                    neighbor_tile.can_host_vegetation
                    and (neighbor.x, neighbor.y) not in active_positions
                ):
                    # Probabilistic spreading based on rainfall
                    neighbor_rainfall = map_data.get_rainfall(neighbor.x, neighbor.y)
                    if random.random() < neighbor_rainfall:
                        active_positions.append((neighbor.x, neighbor.y))

    return tiles_placed


def _place_deserts(
    map_data: MapData,
    desert_tile: Tile,
    coverage: float = 0.10,
    rainfall_threshold: float = 0.3,
) -> int:
    """
    Place deserts in arid regions.

    Deserts are placed in areas with low rainfall on base terrain that
    supports vegetation (identified by the can_host_vegetation flag).

    Args:
        map_data (MapData):
            The map data to modify.
        desert_tile (Tile):
            The desert tile to place.
        coverage (float):
            Target desert coverage ratio (0.0 to 1.0).
        rainfall_threshold (float):
            Maximum rainfall for desert placement.

    Returns:
        int:
            Number of desert tiles placed.

    """
    width = map_data.width
    height = map_data.height
    target_tiles = int(width * height * coverage)
    tiles_placed = 0

    # Collect suitable positions for deserts
    candidates = []
    for y in range(height):
        for x in range(width):
            current_tile = map_data.get_terrain(x, y)
            
            # Only place deserts on base terrain that can host vegetation
            if not current_tile.can_host_vegetation:
                continue

            rainfall = map_data.get_rainfall(x, y)
            elevation = map_data.get_elevation(x, y)

            # Deserts prefer low rainfall and low-to-medium elevation
            if rainfall < rainfall_threshold and elevation > 0.0:
                suitability = (rainfall_threshold - rainfall) * (1.0 - elevation)
                candidates.append((x, y, suitability))

    # Sort by suitability and place deserts on the driest areas
    candidates.sort(key=lambda c: c[2], reverse=True)

    for x, y, _ in candidates[:target_tiles]:
        map_data.set_terrain(x, y, desert_tile)
        tiles_placed += 1

    return tiles_placed


def _place_grasslands(
    map_data: MapData,
    grassland_tile: Tile,
) -> int:
    """
    Place grasslands in moderate conditions.

    Grasslands are placed on base terrain that supports vegetation
    (identified by the can_host_vegetation flag) where climate conditions
    are moderate. They serve as a transition biome between forests and deserts.

    Args:
        map_data (MapData):
            The map data to modify.
        grassland_tile (Tile):
            The grassland tile to place.

    Returns:
        int:
            Number of grassland tiles placed.

    """
    tiles_placed = 0

    for y in range(map_data.height):
        for x in range(map_data.width):
            current_tile = map_data.get_terrain(x, y)

            # Place grasslands only on base terrain that can host vegetation
            if not current_tile.can_host_vegetation:
                continue
                
            elevation = map_data.get_elevation(x, y)
            rainfall = map_data.get_rainfall(x, y)

            # Grasslands prefer moderate conditions
            if 0.0 <= elevation <= 0.4 and 0.3 <= rainfall <= 0.7:
                map_data.set_terrain(x, y, grassland_tile)
                tiles_placed += 1

    return tiles_placed


def _find_tile_by_name(tiles: list[Tile], name: str) -> Tile | None:
    """
    Find a tile by name in a list of tiles.

    Args:
        tiles (list[Tile]):
            List of tiles to search.
        name (str):
            Name of the tile to find.

    Returns:
        Tile | None:
            The tile if found, None otherwise.

    """
    for tile in tiles:
        if tile.name == name:
            return tile
    return None
