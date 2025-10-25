"""
River generation using simple rainfall-based flow simulation.

Simple approach:
1. Find river sources: high elevation + high rainfall
2. Flow downhill from each source
3. Place river tiles along the paths
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
            Elevation percentile for river sources (0.6 = top 40% of elevations).
            Lower values = more rivers on smaller hills too.
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

    # Find potential river sources (high elevation + high rainfall)
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

    # Trace paths from each source
    all_paths = _trace_all_river_paths(
        elevation,
        sources_mask,
        sea_level,
        min_river_length,
    )
    logger.debug(f"Traced {len(all_paths)} river paths (min length {min_river_length})")

    # Keep only the longest 10% of rivers (major rivers only!)
    all_paths = _select_major_rivers(all_paths)
    logger.debug(f"Keeping only {len(all_paths)} longest rivers (top 10%)")

    # Merge rivers when they meet (tributary behavior)
    all_paths = _merge_river_paths(all_paths, elevation)
    logger.debug(f"After merging: {len(all_paths)} independent river systems")

    # Place river tiles along paths
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

    Uses PERCENTILE approach so mountains of all heights can have rivers!
    A source must be in the top X% of elevations AND have good rainfall.

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
    # Calculate rainfall threshold
    rain_threshold = np.percentile(rainfall[rainfall > 0], 100 * min_source_rainfall)

    # Calculate elevation threshold using PERCENTILE (not absolute value!)
    # This ensures all mountain ranges get rivers, regardless of absolute height
    land_mask = elevation > sea_level
    if np.any(land_mask):
        elevation_percentile = np.percentile(
            elevation[land_mask], 100 * min_source_elevation
        )
    else:
        elevation_percentile = sea_level

    # Find cells that meet both criteria
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
    source_coords = np.argwhere(sources_mask)
    all_paths = []

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

    # Sort by length (longest first)
    paths.sort(key=len, reverse=True)

    # Keep top fraction (at least 1 river)
    num_to_keep = max(1, int(len(paths) * top_fraction))
    return paths[:num_to_keep]


def _merge_river_paths(paths: list, elevation: np.ndarray | None = None) -> list:
    """
    Merge rivers when they meet each other.

    When a river comes close to another river, it becomes a tributary.
    We truncate the shorter river at the meeting point.

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
    # Process from longest to shortest, so major rivers take priority
    merged_paths = []
    occupied_cells = set()

    for path in paths:
        # Find where this path comes close to existing rivers
        intersection_index = None
        for i, (y, x) in enumerate(path):
            # Check if this cell or any adjacent cell is already occupied
            is_near_river = False

            # Check the cell itself
            if (y, x) in occupied_cells:
                is_near_river = True
            else:
                # Check adjacent cells (including diagonals)
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if (y + dy, x + dx) in occupied_cells:
                            is_near_river = True
                            break
                    if is_near_river:
                        break

            if is_near_river:
                intersection_index = i
                break

        # If river meets another river, truncate at meeting point
        if intersection_index is not None:
            # Keep path up to (but not including) the intersection
            truncated_path = path[:intersection_index]

            # Calculate how many tiles would actually be visible
            start_distance = max(5, len(truncated_path) // 5)
            visible_tiles = len(truncated_path) - start_distance

            # Minimum visible tiles: scale with source elevation if available
            # Higher mountains = expect longer rivers (5-15 tiles minimum)
            min_visible = 5
            if elevation is not None and len(truncated_path) > 0:
                source_y, source_x = truncated_path[0]
                source_elev = elevation[source_y, source_x]
                # Scale from 5 (low elevation) to 15 (high elevation)
                # Assuming elevation range is roughly -1 to 1
                min_visible = int(5 + (source_elev + 1) * 5)
                min_visible = max(5, min(15, min_visible))

            if visible_tiles >= min_visible:
                merged_paths.append(truncated_path)
                # Add truncated path cells to occupied set
                for cell in truncated_path:
                    occupied_cells.add(cell)
        else:
            # No intersection - keep entire path
            merged_paths.append(path)
            for cell in path:
                occupied_cells.add(cell)

    return merged_paths


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
    path = [(start_y, start_x)]
    visited = {(start_y, start_x)}

    y, x = start_y, start_x

    while True:
        # Stop at sea level
        if elevation[y, x] <= sea_level:
            break

        # Find lowest neighbor
        lowest_elev = elevation[y, x]
        lowest_pos = None

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

        # No lower neighbor - we're at a sink
        if lowest_pos is None:
            break

        # Move to lowest neighbor
        y, x = lowest_pos
        path.append((y, x))
        visited.add((y, x))

        # Prevent infinite loops
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
    # Find river tile
    river_tiles = map_data.find_tiles_by_properties(
        is_water=True,
        is_salt_water=False,
        is_flowing_water=True,
    )

    if not river_tiles:
        logger.warning("No river tiles found in tile catalog")
        return 0

    river_tile = river_tiles[0]

    # Place rivers continuously along all paths
    rivers_placed = 0
    for path in paths:
        # DON'T place at source (mountain tops should not have visible rivers!)
        # Place rivers starting from partway down the mountain
        start_distance = max(5, len(path) // 5)  # Skip first 20% or at least 5 cells

        for i, (y, x) in enumerate(path):
            # Skip the source area (mountain tops)
            if i < start_distance:
                continue

            # Don't place rivers in the sea
            elevation = map_data.get_elevation(x, y)
            if elevation <= sea_level:
                continue

            current_tile = map_data.get_terrain(x, y)

            # Don't replace salt water
            if current_tile.is_water and current_tile.is_salt_water:
                continue

            # Place river
            map_data.set_terrain(x, y, river_tile)
            rivers_placed += 1

    return rivers_placed
