"""
Hydrology utilities for mapgen: accumulation (runoff) computation and lake/basin detection.
"""

import numpy as np
from .map_data import Lake, Position


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
