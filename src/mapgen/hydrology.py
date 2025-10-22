"""
Hydrology utilities for mapgen: rainfall generation and accumulation (runoff) computation.
"""

import logging
import random

import noise
import numpy as np

from .map_data import MapData, Position
from .utils import generate_noise_grid

logger = logging.getLogger(__name__)


def generate_rainfall_map(
    map_data: MapData,
    width: int,
    height: int,
    temperature_scale: float = 100.0,
    humidity_scale: float = 80.0,
    elevation_scale: float = 150.0,
    temp_weight: float = 0.3,
    humidity_weight: float = 0.4,
    orographic_weight: float = 0.3,
    variation_strength: float = 0.1,
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
        temp_weight (float):
            Weight for temperature factor in rainfall calculation.
        humidity_weight (float):
            Weight for humidity factor in rainfall calculation.
        orographic_weight (float):
            Weight for orographic factor in rainfall calculation.
        variation_strength (float):
            Strength of fine-scale variation added to the map.

    """
    # Check if we have flowing water tiles for river generation
    flowing_water_tiles = map_data.find_tiles_by_properties(is_flowing_water=True)
    if not flowing_water_tiles:
        logger.warning(
            "No flowing water tiles found in tile catalog. "
            "Skipping rainfall generation as rivers cannot be generated without flowing water tiles."
        )
        return

    # Generate noise grids for different climate factors
    # Temperature noise: Creates large-scale latitudinal climate zones (poles vs equator)
    # - scale: 100.0 (continental-scale patterns, similar to terrain)
    # - octaves: 4 (moderate detail for climate zones)
    # - persistence: 0.6 (medium detail retention across octaves)
    # - lacunarity: 2.0 (standard frequency scaling)
    # - base: 1 (unique seed for temperature patterns)
    temp_noise = generate_noise_grid(
        width=width,
        height=height,
        scale=temperature_scale,
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
    humid_noise = generate_noise_grid(
        width=width,
        height=height,
        scale=humidity_scale,
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
    elev_noise = generate_noise_grid(
        width=width,
        height=height,
        scale=elevation_scale,
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
    variation_noise = generate_noise_grid(
        width=width,
        height=height,
        scale=20.0,
        octaves=2,
        persistence=0.3,
        lacunarity=2.5,
        base=4,
    )

    # Compute factors using vectorized operations
    temp_factor = _compute_temperature_factor(temp_noise)
    humidity_factor = _compute_humidity_factor(humid_noise)
    orographic_factor = _compute_orographic_factor(elev_noise, map_data, width, height)

    # Combine factors with weights
    rainfall_map = (
        temp_factor * temp_weight
        + humidity_factor * humidity_weight
        + orographic_factor * orographic_weight
    )

    # Normalize to 0-1 range
    min_val = np.min(rainfall_map)
    max_val = np.max(rainfall_map)
    if max_val > min_val:
        rainfall_map = (rainfall_map - min_val) / (max_val - min_val)

    # Add fine-scale variation
    rainfall_map += variation_noise * variation_strength
    rainfall_map = np.clip(rainfall_map, 0.0, 1.0)

    # Round to 4 decimal places
    rainfall_map = np.round(rainfall_map, decimals=4)

    map_data.rainfall_map = rainfall_map.tolist()


def _compute_temperature_factor(temp_noise: np.ndarray) -> np.ndarray:
    """Compute temperature-based rainfall factor."""
    temperature = temp_noise * 1.5
    return 1.0 - np.abs(temperature) * 0.5


def _compute_humidity_factor(humid_noise: np.ndarray) -> np.ndarray:
    """Compute humidity-based rainfall factor."""
    return (humid_noise + 1.0) / 2.0


def _compute_orographic_factor(
    elev_noise: np.ndarray,
    map_data: MapData,
    width: int,
    height: int,
) -> np.ndarray:
    """Compute orographic rainfall factor based on elevation."""
    elevation = np.array(
        [
            [
                map_data.get_elevation(x, y) if map_data.elevation_map else 0.0
                for x in range(width)
            ]
            for y in range(height)
        ]
    )
    return np.maximum(0.3, elevation * 0.7 + np.abs(elev_noise) * 0.3)


def compute_accumulation(elevation: np.ndarray, rainfall: np.ndarray) -> np.ndarray:
    """
    Compute water accumulation (runoff) for each tile given elevation and rainfall.

    Args:
        elevation (np.ndarray): 2D array of elevation values.
        rainfall (np.ndarray): 2D array of rainfall values.

    Returns:
        np.ndarray: 2D array of water accumulation values.
    """
    height, width = elevation.shape
    flow_to = np.full((height, width, 2), -1, dtype=int)
    in_degree = np.zeros((height, width), dtype=int)
    for y in range(height):
        for x in range(width):
            min_elev = elevation[y, x]
            min_pos = (x, y)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if elevation[ny, nx] < min_elev:
                        min_elev = elevation[ny, nx]
                        min_pos = (nx, ny)
            flow_to[y, x] = min_pos
    for y in range(height):
        for x in range(width):
            tx, ty = flow_to[y, x]
            if (tx, ty) != (x, y):
                in_degree[ty, tx] += 1
    from collections import deque

    queue = deque()
    for y in range(height):
        for x in range(width):
            if in_degree[y, x] == 0:
                queue.append((x, y))
    accumulation = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            accumulation[y, x] = rainfall[y, x]
    while queue:
        x, y = queue.popleft()
        tx, ty = flow_to[y, x]
        if (tx, ty) != (x, y):
            accumulation[ty, tx] += accumulation[y, x]
            in_degree[ty, tx] -= 1
            if in_degree[ty, tx] == 0:
                queue.append((tx, ty))
    return accumulation
