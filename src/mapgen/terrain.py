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


def generate_rainfall_map(
    map_data: MapData,
    width: int,
    height: int,
    temperature_scale: float = 100.0,
    humidity_scale: float = 80.0,
    elevation_scale: float = 150.0,
) -> None:
    """
    Generate a rainfall map based on climate and elevation factors.

    Rainfall is higher in areas with:
    - High humidity
    - Moderate temperatures (not too hot/cold)
    - Orographic effects (elevation gradients)

    Args:
        map_data (MapData):
            The map data to update.
        width (int):
            The width of the rainfall map.
        height (int):
            The height of the rainfall map.
        temperature_scale (float):
            Scale for temperature-based rainfall patterns.
        humidity_scale (float):
            Scale for humidity-based rainfall patterns.
        elevation_scale (float):
            Scale for elevation-based orographic effects.

    """
    # Check if we have flowing water tiles for river generation
    flowing_water_tiles = map_data.find_tiles_by_properties(is_flowing_water=True)
    if not flowing_water_tiles:
        logger.warning(
            "No flowing water tiles found in tile catalog. "
            "Skipping rainfall generation as rivers cannot be generated without flowing water tiles."
        )
        return

    # Generate base climate patterns
    temp_offset_x = random.uniform(0, 10000)
    temp_offset_y = random.uniform(0, 10000)
    humid_offset_x = random.uniform(0, 10000)
    humid_offset_y = random.uniform(0, 10000)
    elev_offset_x = random.uniform(0, 10000)
    elev_offset_y = random.uniform(0, 10000)

    rainfall_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            # Temperature factor - moderate temperatures get more rain
            temp_noise = noise.pnoise2(
                (x / temperature_scale) + temp_offset_x,
                (y / temperature_scale) + temp_offset_y,
                octaves=4,
                persistence=0.6,
                lacunarity=2.0,
                repeatx=width,
                repeaty=height,
                base=1,
            )
            # Convert to temperature-like range and calculate rainfall potential
            temperature = temp_noise * 1.5  # -1.5 to 1.5 range
            temp_factor = 1.0 - abs(temperature) * 0.5  # Peak at moderate temps

            # Humidity factor - direct correlation
            humidity_noise = noise.pnoise2(
                (x / humidity_scale) + humid_offset_x,
                (y / humidity_scale) + humid_offset_y,
                octaves=5,
                persistence=0.5,
                lacunarity=2.2,
                repeatx=width,
                repeaty=height,
                base=2,
            )
            humidity_factor = (humidity_noise + 1.0) / 2.0  # 0 to 1 range

            # Orographic factor - elevation gradients create rain shadows
            elev_noise = noise.pnoise2(
                (x / elevation_scale) + elev_offset_x,
                (y / elevation_scale) + elev_offset_y,
                octaves=3,
                persistence=0.7,
                lacunarity=1.8,
                repeatx=width,
                repeaty=height,
                base=3,
            )

            # Calculate elevation gradient (simple approximation)
            elevation = map_data.get_elevation(x, y) if map_data.elevation_map else 0.0
            # Higher elevations and steeper gradients get more rain
            orographic_factor = max(0.3, elevation * 0.7 + abs(elev_noise) * 0.3)

            # Combine factors
            rainfall = (
                temp_factor * 0.3 +
                humidity_factor * 0.4 +
                orographic_factor * 0.3
            )

            rainfall_map[y, x] = rainfall

    # Normalize to 0-1 range
    min_val = np.min(rainfall_map)
    max_val = np.max(rainfall_map)
    if max_val > min_val:
        rainfall_map = (rainfall_map - min_val) / (max_val - min_val)

    # Add some fine-scale variation
    variation_offset_x = random.uniform(0, 10000)
    variation_offset_y = random.uniform(0, 10000)

    for y in range(height):
        for x in range(width):
            variation = noise.pnoise2(
                (x / 20.0) + variation_offset_x,
                (y / 20.0) + variation_offset_y,
                octaves=2,
                persistence=0.3,
                lacunarity=2.5,
                repeatx=width,
                repeaty=height,
                base=4,
            )
            rainfall_map[y, x] += variation * 0.1
            rainfall_map[y, x] = np.clip(rainfall_map[y, x], 0.0, 1.0)

    # Round to 4 decimal places
    rainfall_map = np.round(rainfall_map, decimals=4)

    map_data.rainfall_map = rainfall_map.tolist()


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
        rainfall = map_data.get_rainfall(x, y)

        # Filter tiles based on elevation first
        suitable_tiles = [
            tile for tile in sorted_tiles
            if tile.elevation_min <= elevation <= tile.elevation_max
            and not tile.is_flowing_water  # Don't assign flowing water (rivers) during terrain features
        ]

        if not suitable_tiles:
            return False

        # If we have multiple suitable tiles, use rainfall to choose
        if len(suitable_tiles) > 1:
            # Use rainfall to influence tile selection
            # High rainfall favors tiles that thrive in wet conditions (forests)
            # Low rainfall favors tiles that thrive in dry conditions (plains)
            wet_preferred_tiles = [
                tile for tile in suitable_tiles
                if "wood" in tile.resources or "game" in tile.resources
            ]
            dry_preferred_tiles = [
                tile for tile in suitable_tiles
                if "grain" in tile.resources or "herbs" in tile.resources
            ]

            if wet_preferred_tiles and dry_preferred_tiles:
                # We have both wet and dry preferring tiles
                if rainfall > 0.6:
                    # High rainfall - prefer wet tiles (forests)
                    chosen_tile = max(wet_preferred_tiles, key=lambda t: t.terrain_priority)
                elif rainfall < 0.3:
                    # Low rainfall - prefer dry tiles (plains)
                    chosen_tile = max(dry_preferred_tiles, key=lambda t: t.terrain_priority)
                else:
                    # Moderate rainfall - use highest priority
                    chosen_tile = max(suitable_tiles, key=lambda t: t.terrain_priority)
            else:
                # For other cases, use highest priority
                chosen_tile = max(suitable_tiles, key=lambda t: t.terrain_priority)
        else:
            chosen_tile = suitable_tiles[0]

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

    # Skip flowing water tiles (rivers should stay as rivers)
    if current_tile.is_flowing_water:
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
