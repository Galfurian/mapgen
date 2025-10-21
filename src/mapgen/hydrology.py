"""
Hydrology utilities for mapgen: accumulation (runoff) computation and lake/basin detection.
"""

import numpy as np
from .map_data import Lake, Position


def detect_lakes(
    elevation: np.ndarray,
    accumulation: np.ndarray,
    min_accumulation: float = 5.0,
    min_lake_size: int = 3,
    max_elevation: float = 0.1,
) -> list[Lake]:
    """
    Detect lakes/basins as contiguous low-elevation, high-accumulation regions.

    Args:
        elevation (np.ndarray): 2D elevation array.
        accumulation (np.ndarray): 2D accumulation array.
        min_accumulation (float): Minimum accumulation to consider a lake tile.
        min_lake_size (int): Minimum number of tiles for a valid lake.
        max_elevation (float): Maximum elevation for a tile to be considered part of a lake.

    Returns:
        list[Lake]: List of detected lakes.
    """
    mask = (accumulation >= min_accumulation) & (elevation <= max_elevation)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    lakes = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            # Flood fill to find contiguous region
            region = []
            stack = [(x, y)]
            visited[y, x] = True
            while stack:
                cx, cy = stack.pop()
                region.append(Position(cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
            if len(region) < min_lake_size:
                continue
            region_arr = np.array([(p.y, p.x) for p in region])
            total_acc = float(np.sum([accumulation[p.y, p.x] for p in region]))
            mean_elev = float(np.mean([elevation[p.y, p.x] for p in region]))
            center_yx = region_arr.mean(axis=0)
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
