"""Terrain generation module for procedural maps."""

import logging
import random

import noise
import numpy as np

from .map_data import MapData

logger = logging.getLogger(__name__)


def generate_noise_map(
    map_data: MapData,
    width: int,
    height: int,
    scale: float = 50.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> None:
    """
    Generate a Perlin noise map.

    Args:
        width (int):
            The width of the noise map.
        height (int):
            The height of the noise map.
        scale (float):
            The scale of the noise.
        octaves (int):
            The number of octaves.
        persistence (float):
            The persistence value.
        lacunarity (float):
            The lacunarity value.

    """
    offset_x = random.uniform(0, 10000)
    offset_y = random.uniform(0, 10000)

    noise_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            noise_map[y, x] = noise.pnoise2(
                (x / scale) + offset_x,
                (y / scale) + offset_y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0,
            )

    # Normalize to full -1 to 1 range
    min_val = np.min(noise_map)
    max_val = np.max(noise_map)
    if max_val > min_val:
        noise_map = (noise_map - min_val) / (max_val - min_val) * 2 - 1

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
        for tile in sorted_tiles:
            if tile.elevation_min <= elevation <= tile.elevation_max:
                map_data.set_terrain(x, y, tile)
                return True
        return False

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
