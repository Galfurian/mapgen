"""River generation module."""

import logging
import random

from .map_data import MapData, Position

logger = logging.getLogger(__name__)


def generate_rivers(
    map_data: MapData,
    min_river_length: int = 10,
    max_rivers: int = 5,
    rainfall_threshold: float = 0.6,
    elevation_threshold: float = 0.3,
) -> None:
    """
    Generate rivers from high elevation, high rainfall areas.

    Rivers flow downhill following paths of least resistance.

    Args:
        map_data (MapData):
            The map data containing elevation and rainfall maps.
        min_river_length (int):
            Minimum length for a river to be considered valid.
        max_rivers (int):
            Maximum number of rivers to generate.
        rainfall_threshold (float):
            Minimum rainfall value for river sources.
        elevation_threshold (float):
            Minimum elevation value for river sources.

    """
    if not map_data.rainfall_map or not map_data.elevation_map:
        logger.warning("Cannot generate rivers without rainfall and elevation data")
        return

    # Find potential river sources
    sources = _find_river_sources(map_data, rainfall_threshold, elevation_threshold)

    if not sources:
        logger.debug("No suitable river sources found")
        return

    # Limit number of rivers
    sources = sources[:max_rivers]

    logger.debug(f"Generating up to {len(sources)} rivers")

    rivers_generated = 0
    for source in sources:
        river_path = _generate_river_path(map_data, source, min_river_length)
        if river_path and len(river_path) >= min_river_length:
            _apply_river_to_map(map_data, river_path)
            rivers_generated += 1
            logger.debug(f"Generated river {rivers_generated} with {len(river_path)} tiles")
        else:
            logger.debug(f"River from {source} failed: path_length={len(river_path) if river_path else 0}")

    logger.debug(f"Generated {rivers_generated} rivers")


def _find_river_sources(
    map_data: MapData,
    rainfall_threshold: float,
    elevation_threshold: float,
) -> list[Position]:
    """
    Find river sources: high elevation, high rainfall areas not on water.

    Args:
        map_data (MapData):
            The map data.
        rainfall_threshold (float):
            Minimum rainfall for sources.
        elevation_threshold (float):
            Minimum elevation for sources.

    Returns:
        list[Position]:
            List of potential river source positions.

    """
    sources = []

    for y in range(map_data.height):
        for x in range(map_data.width):
            rainfall = map_data.get_rainfall(x, y)
            elevation = map_data.get_elevation(x, y)
            current_tile = map_data.get_terrain(x, y)

            if (rainfall >= rainfall_threshold and
                elevation >= elevation_threshold and
                not current_tile.is_water):
                sources.append(Position(x, y))

    # Sort by rainfall and elevation (highest first)
    sources.sort(key=lambda pos: (map_data.get_rainfall(pos.x, pos.y), map_data.get_elevation(pos.x, pos.y)), reverse=True)

    logger.debug(f"Found {len(sources)} potential river sources")
    return sources


def _generate_river_path(
    map_data: MapData,
    start: Position,
    min_length: int,
) -> list[Position] | None:
    """
    Generate a river path from source following downhill gradient with erosion.

    Args:
        map_data (MapData):
            The map data.
        start (Position):
            Starting position of the river.
        min_length (int):
            Minimum path length.

    Returns:
        list[Position] | None:
            The river path, or None if no valid path found.

    """
    path = [start]
    current = start
    visited = set([start])

    max_iterations = min(min_length * 2, 30)  # Hard limit of 30 tiles max per river
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Find the lowest elevation neighbor (steepest downhill)
        neighbors = map_data.get_neighbors(current.x, current.y, walkable_only=False)
        candidates = []
        for neighbor in neighbors:
            if neighbor not in visited:
                elevation = map_data.get_elevation(neighbor.x, neighbor.y)
                candidates.append((elevation, neighbor))

        if not candidates:
            # No unvisited neighbors, stop
            break

        # Sort by elevation (lowest first)
        candidates.sort(key=lambda x: x[0])

        # Check if we're still flowing downhill
        current_elevation = map_data.get_elevation(current.x, current.y)
        lowest_elevation, next_pos = candidates[0]

        # If we can't go downhill, try "erosion" - allow slight uphill if it's the only option
        # But only if the difference is small (erosion effect) and we're not too far along
        elevation_diff = lowest_elevation - current_elevation
        if elevation_diff > 0.01:  # Much more restrictive erosion
            break

        # Check if next position is salt water (ocean) - rivers stop at sea
        next_tile = map_data.get_terrain(next_pos.x, next_pos.y)
        if next_tile.is_water and next_tile.is_salt_water:
            # River has reached the sea, stop
            break

        # Add to path and continue
        path.append(next_pos)
        visited.add(next_pos)
        current = next_pos

    # Only return path if it's long enough
    if len(path) >= min_length:
        return path

    return None


def _apply_river_to_map(
    map_data: MapData,
    river_path: list[Position],
) -> None:
    """
    Apply river tiles to the map along the given path.

    Args:
        map_data (MapData):
            The map data to modify.
        river_path (list[Position]):
            The path of the river.

    """
    # Find river tiles (flowing fresh water)
    river_tiles = map_data.find_tiles_by_properties(
        is_water=True, is_salt_water=False, is_flowing_water=True
    )

    if not river_tiles:
        logger.warning("No river tiles found in tile catalog")
        return

    # Use the first matching river tile
    river_tile = river_tiles[0]

    # Apply river tiles along the path
    for position in river_path:
        # Replace any non-salt-water tile with river
        current_tile = map_data.get_terrain(position.x, position.y)
        if not (current_tile.is_water and current_tile.is_salt_water):
            map_data.set_terrain(position.x, position.y, river_tile)
