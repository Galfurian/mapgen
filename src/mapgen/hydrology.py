"""
Hydrology utilities for mapgen: rainfall generation, accumulation (runoff) computation and lake/basin detection.
"""

import logging
import random

import noise
import numpy as np

from .map_data import Lake, MapData, Position

logger = logging.getLogger(__name__)


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


def detect_lakes(
    elevation: np.ndarray,
    accumulation: np.ndarray,
    min_accumulation: float = 10.0,
    min_lake_size: int = 5,
    max_elevation: float = -0.1,
) -> list[Lake]:
    """
    Detect lakes in clear depressions: low elevation areas surrounded by higher ground.

    Args:
        elevation (np.ndarray): 2D elevation array.
        accumulation (np.ndarray): 2D accumulation array.
        min_accumulation (float): Minimum accumulation to consider a lake tile.
        min_lake_size (int): Minimum number of tiles for a valid lake.
        max_elevation (float): Maximum elevation for a tile to be considered part of a lake.

    Returns:
        list[Lake]: List of detected lakes.
    """
    height, width = elevation.shape
    visited = np.zeros_like(elevation, dtype=bool)
    lakes = []

    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue

            # Check if this could be a lake seed
            if (elevation[y, x] <= max_elevation and
                accumulation[y, x] >= min_accumulation):

                # Check if this position is in a depression (surrounded by higher elevation)
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighbors.append(elevation[ny, nx])

                if neighbors and min(neighbors) > elevation[y, x]:
                    # This is a depression, flood fill to find the full lake
                    region = []
                    stack = [(x, y)]
                    visited[y, x] = True
                    region_elevations = [elevation[y, x]]

                    while stack:
                        cx, cy = stack.pop()
                        region.append(Position(cx, cy))

                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < width and 0 <= ny < height and
                                not visited[ny, nx] and
                                elevation[ny, nx] <= max_elevation and
                                accumulation[ny, nx] >= min_accumulation):
                                visited[ny, nx] = True
                                stack.append((nx, ny))
                                region_elevations.append(elevation[ny, nx])

                    if len(region) >= min_lake_size:
                        total_acc = float(np.sum([accumulation[p.y, p.x] for p in region]))
                        mean_elev = float(np.mean(region_elevations))
                        center_yx = np.array([(p.y, p.x) for p in region]).mean(axis=0)
                        center = (float(center_yx[1]), float(center_yx[0]))

                        lake = Lake(
                            tiles=region,
                            center=center,
                            total_accumulation=total_acc,
                            mean_elevation=mean_elev,
                            size=len(region),
                        )
                        lakes.append(lake)

    return lakes


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
