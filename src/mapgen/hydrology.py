"""
Hydrology utilities for the map generator.

This module provides functions for generating rainfall maps based on climate
factors and computing water accumulation (runoff) using elevation data.
"""

import logging
from collections import deque

import numpy as np
from scipy import ndimage

from .map_data import BodyOfWater, MapData, Position
from .utils import generate_noise_grid, normalize_array


logger = logging.getLogger(__name__)


def generate_rainfall_map(
    map_data: MapData,
    temp_weight: float = 0.3,
    humidity_weight: float = 0.4,
    orographic_weight: float = 0.3,
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
        temp_weight (float):
            Weight for temperature factor in rainfall calculation.
        humidity_weight (float):
            Weight for humidity factor in rainfall calculation.
        orographic_weight (float):
            Weight for orographic factor in rainfall calculation.

    """
    # Generate noise grids for different climate factors. Temperature noise:
    # Creates large-scale latitudinal climate zones (poles vs equator)
    # - scale: 100.0 (continental-scale patterns, similar to terrain)
    # - octaves: 4 (moderate detail for climate zones)
    # - persistence: 0.6 (medium detail retention across octaves)
    # - lacunarity: 2.0 (standard frequency scaling)
    # - base: 1 (unique seed for temperature patterns)
    temperature_noise: np.ndarray = generate_noise_grid(
        width=map_data.width,
        height=map_data.height,
        scale=100,
        octaves=4,
        persistence=0.6,
        lacunarity=2.0,
        base=1,
    )

    # Humidity noise: Creates regional moisture patterns (coastal vs inland)
    # - scale: 80.0 (slightly smaller than temperature for more variation)
    # - octaves: 5 (higher detail since humidity varies more locally)
    # - persistence: 0.5 (lower persistence for more variation between levels)
    # - lacunarity: 2.2 (slightly higher for different visual patterns)
    # - base: 2 (different seed from temperature)
    humidity_noise: np.ndarray = generate_noise_grid(
        width=map_data.width,
        height=map_data.height,
        scale=80,
        octaves=5,
        persistence=0.5,
        lacunarity=2.2,
        base=2,
    )

    # Orographic noise: Creates broad mountain/valley rainfall modification
    # - scale: 150.0 (larger scale for broader elevation effects)
    # - octaves: 3 (fewer octaves for smoother, consistent patterns)
    # - persistence: 0.7 (higher persistence maintains elevation influence)
    # - lacunarity: 1.8 (lower value for different frequency relationships)
    # - base: 3 (different seed from other climate factors)
    elevation_noise: np.ndarray = generate_noise_grid(
        width=map_data.width,
        height=map_data.height,
        scale=150.0,
        octaves=3,
        persistence=0.7,
        lacunarity=1.8,
        base=3,
    )

    # Fine-scale variation: Adds subtle local weather randomness
    # - scale: 20.0 (small scale for local weather variation)
    # - octaves: 2 (minimal octaves for subtle effects)
    # - persistence: 0.3 (low persistence so it doesn't dominate main patterns)
    # - lacunarity: 2.5 (higher value for different small-scale patterns)
    # - base: 4 (different seed from other factors)
    variation_noise: np.ndarray = generate_noise_grid(
        width=map_data.width,
        height=map_data.height,
        scale=20.0,
        octaves=2,
        persistence=0.3,
        lacunarity=2.5,
        base=4,
    )

    # Compute the temperature factor.
    temperature_factor = 1.0 - np.abs(temperature_noise * 1.5) * 0.5
    map_data.temperature_map = normalize_array(temperature_noise).tolist()

    # Compute the humidity factor.
    humidity_factor = (humidity_noise + 1.0) / 2.0
    map_data.humidity_map = normalize_array(humidity_noise).tolist()

    # Compute the orographic factor based on elevation.
    orographic_factor = np.maximum(
        0.3, np.array(map_data.elevation_map) * 0.7 + np.abs(elevation_noise) * 0.3
    )
    map_data.orographic_map = normalize_array(orographic_factor).tolist()

    # Combine factors with weights
    rainfall_map = (
        temperature_factor * temp_weight
        + humidity_factor * humidity_weight
        + orographic_factor * orographic_weight
    )

    # Normalize to 0-1 range
    min_val = np.min(rainfall_map)
    max_val = np.max(rainfall_map)
    if max_val > min_val:
        rainfall_map = (rainfall_map - min_val) / (max_val - min_val)

    # Add fine-scale variation
    rainfall_map = np.clip(rainfall_map + variation_noise, 0.0, 1.0)

    # Round to 4 decimal places
    rainfall_map = np.round(rainfall_map, decimals=4)

    map_data.rainfall_map = rainfall_map.tolist()


def generate_accumulation_map(map_data: MapData) -> None:
    """
    Generate water accumulation (runoff) map for the given map data.

    This function computes water accumulation based on elevation and rainfall
    data, then stores the result in map_data.accumulation_map.

    Args:
        map_data (MapData):
            The map data containing elevation and rainfall information.

    Raises:
        ValueError:
            If elevation_map is not available in map_data.

    """
    if not map_data.elevation_map:
        raise ValueError("Elevation map is required for accumulation computation")

    elevation = np.array(map_data.elevation_map)
    rainfall = (
        np.array(map_data.rainfall_map)
        if map_data.rainfall_map
        else np.ones_like(elevation) * 0.5
    )

    height, width = elevation.shape

    # Compute flow directions: each cell flows to the lowest neighbor
    flow_to = np.full((height, width, 2), -1, dtype=int)
    in_degree = np.zeros((height, width), dtype=int)
    for y in range(height):
        for x in range(width):
            min_elev = elevation[y, x]
            min_pos = (x, y)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if elevation[ny, nx] < min_elev:
                    min_elev = elevation[ny, nx]
                    min_pos = (nx, ny)
            flow_to[y, x] = min_pos

    # Build in-degree for topological sort (cells with no inflow first)
    for y in range(height):
        for x in range(width):
            tx, ty = flow_to[y, x]
            if (tx, ty) != (x, y):
                in_degree[ty, tx] += 1

    # Initialize queue with cells that have no incoming flow
    queue = deque(
        (x, y) for y in range(height) for x in range(width) if in_degree[y, x] == 0
    )

    # Initialize accumulation with local rainfall
    accumulation = rainfall.copy()

    # Process cells in topological order to accumulate flow
    while queue:
        x, y = queue.popleft()
        tx, ty = flow_to[y, x]
        if (tx, ty) != (x, y):
            accumulation[ty, tx] += accumulation[y, x]
            in_degree[ty, tx] -= 1
            if in_degree[ty, tx] == 0:
                queue.append((tx, ty))

    # Round to 4 decimal places and store in map_data
    accumulation = np.round(accumulation, decimals=4)
    map_data.accumulation_map = accumulation.tolist()


def identify_bodies_of_water(map_data: MapData) -> None:
    """
    Identify and label bodies of water on the map.

    This function analyzes the terrain map to find contiguous water regions
    and labels them as bodies of water in map_data.bodies_of_water.

    Args:
        map_data (MapData):
            The map data containing terrain information.

    """
    if not map_data.elevation_map:
        logger.warning("Elevation map is required for body of water identification")
        return

    # Get the map dimensions.
    height, width = map_data.height, map_data.width

    # Create a binary mask of water tiles
    water_mask: np.ndarray = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            tile = map_data.get_terrain(x, y)
            water_mask[y, x] = tile.is_water

    # Label connected components in the water mask.
    labeled_mask, num_features = ndimage.label(water_mask)

    # Create BodyOfWater instances for each labeled water body
    for label in range(1, num_features + 1):
        # Get all positions in this water body.
        body_positions = np.where(labeled_mask == label)
        # Create a BodyOfWater instance and add it to map_data.
        map_data.bodies_of_water.append(
            BodyOfWater(
                is_salt_water=True,
                tiles=[
                    Position(x=int(x), y=int(y))
                    for y, x in zip(body_positions[0], body_positions[1])
                ],
            )
        )


def classify_bodies_of_water(
    map_data: MapData,
    lake_size_threshold: int = 1000,
) -> None:
    """
    Classify bodies of water as seas or lakes based on their size.

    Bodies of water with a number of tiles greater than lake_size_threshold
    are classified as seas; otherwise, they are classified as lakes.

    Args:
        map_data (MapData):
            The map data containing bodies of water.
        lake_size_threshold (int):
            The size threshold to distinguish lakes from seas.

    """
    # Find salt-water tiles in the tiles collection.
    salt_water_tiles = map_data.find_tiles_by_properties(
        is_water=True,
        is_salt_water=True,
        is_flowing_water=False,
    )
    fresh_water_tiles = map_data.find_tiles_by_properties(
        is_water=True,
        is_salt_water=False,
        is_flowing_water=False,
    )
    if not salt_water_tiles:
        logger.info("No salt-water tiles found; skipping body of water classification")
        return
    if not fresh_water_tiles:
        logger.info("No fresh-water tiles found; skipping body of water classification")
        return
    # Sort tiles by terrain priority (higher priority first).
    salt_water_tiles = sorted(
        salt_water_tiles,
        key=lambda t: t.terrain_priority,
        reverse=True,
    )
    fresh_water_tiles = sorted(
        fresh_water_tiles,
        key=lambda t: t.terrain_priority,
        reverse=True,
    )

    for body in map_data.bodies_of_water:
        logger.debug(
            f"Classifying body of water with {len(body.tiles)} tiles, as {'salt water' if body.is_salt_water else 'fresh water'}."
        )
        if len(body.tiles) > lake_size_threshold:
            for tile in body.tiles:
                map_data.set_terrain(tile.x, tile.y, salt_water_tiles[0])
        else:
            for tile in body.tiles:
                map_data.set_terrain(tile.x, tile.y, fresh_water_tiles[0])
