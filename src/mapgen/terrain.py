import logging
import random

import noise
import numpy as np

from .map_data import MapData, PlacementMethod
from .utils import generate_noise_grid

logger = logging.getLogger(__name__)


def generate_elevation_map(
    map_data: MapData,
    width: int,
    height: int,
    scale: float = 50.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    sea_level: float = 0.0,
) -> None:
    """
    Generate an elevation map using Perlin noise.

    Args:
        map_data (MapData):
            The map data to store the elevation map in.
        width (int):
            The width of the elevation map.
        height (int):
            The height of the elevation map.
        scale (float):
            The scale of the noise.
        octaves (int):
            The number of octaves.
        persistence (float):
            The persistence value.
        lacunarity (float):
            The lacunarity value.
        sea_level (float):
            The sea level elevation (controls land/sea ratio).

    """
    noise_map = generate_noise_grid(
        width=width,
        height=height,
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        base=0,
    )

    # Normalize to full -1 to 1 range
    min_val = np.min(noise_map)
    max_val = np.max(noise_map)
    if max_val > min_val:
        noise_map = (noise_map - min_val) / (max_val - min_val) * 2 - 1

    # Apply sea level adjustment to control land/sea ratio
    if sea_level != 0.0:
        shifted = noise_map - sea_level
        sea_mask = shifted < 0
        land_mask = shifted >= 0

        if np.any(sea_mask):
            sea_min = np.min(shifted[sea_mask])
            sea_max = 0.0
            if sea_min != sea_max:
                shifted[sea_mask] = -1 + (shifted[sea_mask] - sea_min) / (
                    sea_max - sea_min
                )
            else:
                shifted[sea_mask] = -1

        if np.any(land_mask):
            land_min = 0.0
            land_max = np.max(shifted[land_mask])
            if land_min != land_max:
                shifted[land_mask] = (shifted[land_mask] - land_min) / (
                    land_max - land_min
                )
            else:
                shifted[land_mask] = 0

        noise_map = shifted

    # Round to 4 decimal places
    noise_map = np.round(noise_map, decimals=4)

    map_data.elevation_map = noise_map.tolist()


def apply_terrain_features(
    map_data: MapData,
) -> None:
    """
    Apply terrain features based on noise map.

    Args:
        map_data (MapData):
            The map grid to modify.

    """
    sorted_tiles = sorted(
        map_data.tiles,
        key=lambda t: t.terrain_priority,
        reverse=True,
    )

    def apply_suitable_tile(x: int, y: int) -> bool:
        """
        Apply the most suitable tile for the given position.
        """
        elevation = map_data.get_elevation(x, y)

        # Filter tiles based on elevation
        suitable_tiles = [
            tile
            for tile in sorted_tiles
            if tile.elevation_min <= elevation <= tile.elevation_max
            and tile.placement_method.name
            == "TERRAIN_BASED"  # Only assign terrain-based tiles during terrain features
        ]

        if not suitable_tiles:
            return False

        # Choose the tile with the highest priority
        chosen_tile = max(suitable_tiles, key=lambda t: t.terrain_priority)

        map_data.set_terrain(x, y, chosen_tile)
        return True

    for y in range(map_data.height):
        for x in range(map_data.width):
            if not apply_suitable_tile(x, y):
                logger.warning(
                    f"No suitable tile found for elevation {map_data.get_elevation(x, y)} at ({x}, {y})"
                )


def smooth_terrain(
    map_data: MapData,
    iterations: int = 5,
) -> None:
    """
    Smooth the terrain using cellular automata rules.

    Args:
        map_data (MapData):
            The map grid to smooth.
        iterations (int):
            The number of smoothing iterations.

    """
    for _ in range(iterations):
        new_grid_indices = [row[:] for row in map_data.grid]
        for y in range(1, map_data.height - 1):
            for x in range(1, map_data.width - 1):
                new_tile_index = _get_smoothed_tile_index(map_data, x, y)
                if new_tile_index is not None:
                    new_grid_indices[y][x] = new_tile_index
        map_data.grid = new_grid_indices


def _get_smoothed_tile_index(
    map_data: MapData,
    x: int,
    y: int,
) -> int | None:
    """
    Get the smoothed tile index for a given position.

    Args:
        map_data (MapData):
            The map data.
        x (int):
            The x coordinate.
        y (int):
            The y coordinate.

    Returns:
        int | None:
            The new tile index or None if no change.

    """
    current_tile = map_data.get_terrain(x, y)

    # Skip algorithm-based tiles (rivers, lakes should stay as placed by algorithms)
    if current_tile.placement_method == PlacementMethod.ALGORITHM_BASED:
        return None

    # Skip obstacles (non-walkable tiles)
    if not current_tile.walkable:
        return None

    # Get neighbor properties
    neighbors = map_data.get_neighbor_tiles(
        x,
        y,
        walkable_only=False,
        include_diagonals=True,
    )

    # Count different neighbor types
    nw_count = sum(1 for n in neighbors if not n.walkable)
    mod_cost_count = sum(1 for n in neighbors if n.movement_cost > 1.0)
    neg_elev_count = sum(1 for n in neighbors if n.elevation_influence < 0.0)
    pos_elev_count = sum(1 for n in neighbors if n.elevation_influence > 0.5)

    # Apply smoothing rules
    if nw_count > 4:
        candidates = [t for t in map_data.tiles if not t.walkable]
        if candidates:
            tile = max(candidates, key=lambda t: t.smoothing_priority)
            return map_data.tiles.index(tile)
    elif neg_elev_count > 4:
        candidates = [t for t in map_data.tiles if t.elevation_influence < 0]
        if candidates:
            tile = max(candidates, key=lambda t: t.smoothing_priority)
            return map_data.tiles.index(tile)
    elif pos_elev_count > 3:
        candidates = [t for t in map_data.tiles if t.elevation_influence > 0.5]
        if candidates:
            tile = max(candidates, key=lambda t: t.smoothing_priority)
            return map_data.tiles.index(tile)
    elif mod_cost_count > 4:
        candidates = [t for t in map_data.tiles if 1.0 < t.movement_cost < 2.0]
        if candidates:
            tile = max(candidates, key=lambda t: t.smoothing_priority)
            return map_data.tiles.index(tile)

    return None
