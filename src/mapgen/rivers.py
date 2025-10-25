"""
River generation utilities for the map generator.

This module implements a simple rainfall-based flow simulation for generating
rivers. The approach involves finding sources at high elevation with high
rainfall, tracing downhill paths, and placing river tiles along the longest
paths.
"""

import logging

import numpy as np

from .map_data import MapData

logger = logging.getLogger(__name__)


def generate_rivers(
    map_data: MapData,
    min_source_elevation: float = 0.6,
    min_source_rainfall: float = 0.5,
    min_river_length: int = 10,
    sea_level: float = 0.0,
) -> None:
    """
    Generate rivers by tracing water flow from mountain sources.

    SIMPLE APPROACH:
        1. Find sources: cells with high elevation AND high rainfall
        2. Trace downhill from each source to the sea
        3. Keep only the longest 10% of rivers
        4. Place rivers continuously along those paths

    That's it. No complex erosion, no confusing thresholds.

    Args:
        map_data (MapData):
            The map data.
        min_source_elevation (float):
            Elevation percentile for river sources (0.6 = top 40% of
            elevations). Lower values = more rivers on smaller hills too.
        min_source_rainfall (float):
            Minimum rainfall percentile for sources (0.5 = top 50%).
        min_river_length (int):
            Minimum path length to place a river.
        sea_level (float):
            Sea level.

    """
    if not map_data.elevation_map or not map_data.rainfall_map:
        logger.warning("Cannot generate rivers without elevation and rainfall data")
        return

    # Convert to numpy
    elevation = np.array(map_data.elevation_map, dtype=np.float32)
    rainfall = np.array(map_data.rainfall_map, dtype=np.float32)

    # Identify potential river sources.
    sources_mask = _find_river_sources(
        elevation,
        rainfall,
        min_source_elevation,
        min_source_rainfall,
        sea_level,
    )

    num_sources = np.sum(sources_mask)
    if num_sources == 0:
        logger.warning("No river sources found")
        return

    logger.debug(f"Found {num_sources} potential river sources")

    # Trace downhill paths from sources.
    all_paths = _trace_all_river_paths(
        elevation,
        sources_mask,
        sea_level,
        min_river_length,
    )
    logger.debug(f"Traced {len(all_paths)} river paths (min length {min_river_length})")

    # Filter to keep only major rivers.
    all_paths = _select_major_rivers(all_paths)
    logger.debug(f"Keeping only {len(all_paths)} longest rivers (top 10%)")

    # Merge tributary rivers.
    all_paths = _merge_river_paths(all_paths, elevation)
    logger.debug(f"After merging: {len(all_paths)} independent river systems")

    # Place river tiles on the map.
    rivers_placed = _place_river_tiles(map_data, all_paths, sea_level)
    logger.debug(f"Placed {rivers_placed} river tiles")


def _find_river_sources(
    elevation: np.ndarray,
    rainfall: np.ndarray,
    min_source_elevation: float,
    min_source_rainfall: float,
    sea_level: float,
) -> np.ndarray:
    """
    Find potential river sources based on elevation and rainfall.

    Uses PERCENTILE approach so mountains of all heights can have rivers! A
    source must be in the top X% of elevations AND have good rainfall.

    Args:
        elevation (np.ndarray):
            Elevation map.
        rainfall (np.ndarray):
            Rainfall map.
        min_source_elevation (float):
            Elevation percentile threshold (0.6 = top 40% of elevations).
        min_source_rainfall (float):
            Rainfall percentile threshold (0.5 = top 50%).
        sea_level (float):
            Sea level.

    Returns:
        np.ndarray:
            Boolean mask of potential source locations.

    """
    # Determine rainfall threshold.
    rain_threshold = np.percentile(rainfall[rainfall > 0], 100 * min_source_rainfall)

    # Identify land areas above sea level.
    land_mask = elevation > sea_level
    if np.any(land_mask):
        # Calculate elevation percentile threshold.
        elevation_percentile = np.percentile(
            elevation[land_mask], 100 * min_source_elevation
        )
    else:
        elevation_percentile = sea_level

    # Create mask for cells meeting both criteria.
    sources_mask = (
        (elevation >= elevation_percentile)
        & (rainfall >= rain_threshold)
        & (elevation > sea_level)
    )

    return sources_mask


def _trace_all_river_paths(
    elevation: np.ndarray,
    sources_mask: np.ndarray,
    sea_level: float,
    min_river_length: int,
) -> list:
    """
    Trace downhill paths from all river sources.

    Args:
        elevation (np.ndarray):
            Elevation map.
        sources_mask (np.ndarray):
            Boolean mask of source locations.
        sea_level (float):
            Sea level.
        min_river_length (int):
            Minimum path length to keep.

    Returns:
        list:
            List of paths (each path is a list of (y, x) coordinates).

    """
    # Get coordinates of all sources.
    source_coords = np.argwhere(sources_mask)
    # Initialize list for all paths.
    all_paths = []

    # Trace path from each source.
    for sy, sx in source_coords:
        path = _trace_downhill_path(elevation, sy, sx, sea_level)
        if len(path) >= min_river_length:
            all_paths.append(path)

    return all_paths


def _select_major_rivers(paths: list, top_fraction: float = 0.1) -> list:
    """
    Select only the longest rivers (major rivers).

    Args:
        paths (list):
            List of all traced paths.
        top_fraction (float):
            Fraction of rivers to keep (0.1 = top 10%).

    Returns:
        list:
            Filtered list of longest paths.

    """
    if not paths:
        return []

    # Sort paths by length descending.
    paths.sort(key=len, reverse=True)

    # Calculate number of rivers to keep.
    num_to_keep = max(1, int(len(paths) * top_fraction))
    return paths[:num_to_keep]


def _merge_river_paths(paths: list, elevation: np.ndarray | None = None) -> list:
    """
    Merge rivers when they meet each other.

    When a river comes close to another river, it becomes a tributary. We
    truncate the shorter river at the meeting point.

    This prevents rivers from being too close to each other.

    Args:
        paths (list):
            List of paths (sorted by length, longest first).
        elevation (np.ndarray):
            Optional elevation map to scale minimum length by source elevation.

    Returns:
        list:
            List of merged paths.

    """
    if len(paths) <= 1:
        return paths

    # Build a set of all cells occupied by rivers (for quick lookup)
    merged_paths = []
    occupied_cells: set[tuple[int, int]] = set()

    # Process each path from longest to shortest.
    for path in paths:
        # Find where this path comes close to existing rivers.
        intersection_index = None
        for i, (y, x) in enumerate(path):
            # Check if near existing river (cell or adjacent).
            if _is_near_occupied(y, x, occupied_cells):
                intersection_index = i
                break

        # If intersection found, truncate path.
        if intersection_index is not None:
            truncated_path = path[:intersection_index]
            start_distance = max(5, len(truncated_path) // 5)
            visible_tiles = len(truncated_path) - start_distance

            min_visible = 5
            if elevation is not None and truncated_path:
                source_y, source_x = truncated_path[0]
                source_elev = elevation[source_y, source_x]
                min_visible = max(5, min(15, int(5 + (source_elev + 1) * 5)))

            if visible_tiles >= min_visible:
                merged_paths.append(truncated_path)
                occupied_cells.update(truncated_path)
        else:
            # No intersection, keep entire path.
            merged_paths.append(path)
            occupied_cells.update(path)

    return merged_paths


def _is_near_occupied(y: int, x: int, occupied_cells: set[tuple[int, int]]) -> bool:
    """Check if a cell or its neighbors are occupied."""
    # Check the cell itself.
    if (y, x) in occupied_cells:
        return True
    # Check adjacent cells (including diagonals).
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if (y + dy, x + dx) in occupied_cells:
                return True
    return False


def _trace_downhill_path(
    elevation: np.ndarray,
    start_y: int,
    start_x: int,
    sea_level: float,
) -> list:
    """
    Trace a path downhill from a starting point.

    Simple: follow the steepest descent until we reach sea or a sink.

    Args:
        elevation (np.ndarray):
            Elevation map.
        start_y (int):
            Starting Y coordinate.
        start_x (int):
            Starting X coordinate.
        sea_level (float):
            Sea level.

    Returns:
        list:
            List of (y, x) coordinates in the path.

    """
    height, width = elevation.shape
    # Initialize path with start position.
    path = [(start_y, start_x)]
    # Track visited positions to avoid loops.
    visited = {(start_y, start_x)}

    y, x = start_y, start_x

    # Loop until we stop flowing.
    while True:
        # Stop at sea level.
        if elevation[y, x] <= sea_level:
            break

        # Initialize lowest elevation.
        lowest_elev = elevation[y, x]
        # Initialize lowest position.
        lowest_pos = None

        # Check all 8 neighbors.
        for dy, dx in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if (ny, nx) not in visited and elevation[ny, nx] < lowest_elev:
                    lowest_elev = elevation[ny, nx]
                    lowest_pos = (ny, nx)

        # Stop if no lower neighbor.
        if lowest_pos is None:
            break

        # Move to lowest neighbor.
        y, x = lowest_pos
        path.append((y, x))
        visited.add((y, x))

        # Prevent infinite loops.
        if len(path) > 1000:
            break

    return path


def _place_river_tiles(
    map_data: MapData,
    paths: list,
    sea_level: float,
) -> int:
    """
    Place river tiles along traced paths.

    Rivers appear continuously from partway down the mountain to the sea.

    Args:
        map_data (MapData):
            Map data.
        paths (list):
            List of paths, where each path is a list of (y, x) coordinates.
        sea_level (float):
            Sea level.

    Returns:
        int:
            Number of river tiles placed.

    """
    # Find river tile in catalog.
    river_tiles = map_data.find_tiles_by_properties(
        is_water=True,
        is_salt_water=False,
        is_flowing_water=True,
    )

    if not river_tiles:
        logger.warning("No river tiles found in tile catalog")
        return 0

    # Use the first river tile.
    river_tile = river_tiles[0]

    # Initialize counter.
    rivers_placed = 0
    # Process each river path.
    for path in paths:
        # Calculate start distance for visible rivers.
        start_distance = max(5, len(path) // 5)  # Skip first 20% or at least 5 cells

        # Place tiles along the path.
        for i, (y, x) in enumerate(path):
            # Skip source area.
            if i < start_distance:
                continue

            # Get elevation at position.
            elevation = map_data.get_elevation(x, y)
            # Skip sea areas.
            if elevation <= sea_level:
                continue

            # Get current terrain tile.
            current_tile = map_data.get_terrain(x, y)

            # Skip salt water.
            if current_tile.is_water and current_tile.is_salt_water:
                continue

            # Place river tile.
            map_data.set_terrain(x, y, river_tile)
            # Increment counter.
            rivers_placed += 1

    return rivers_placed
