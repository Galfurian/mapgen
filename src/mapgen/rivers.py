"""River generation module."""

import logging
import random

from .map_data import MapData, Position

logger = logging.getLogger(__name__)


def generate_rivers(
    map_data: MapData,
    min_river_length: int = 10,
    max_rivers: int = 5,
    rainfall_threshold: float = 0.7,
) -> None:
    """
    Generate river network based on rainfall and elevation.

    Rivers start in high rainfall areas and flow downhill following
    elevation gradients.

    Args:
        map_data (MapData):
            The map data containing elevation and rainfall maps.
        min_river_length (int):
            Minimum length for a river to be considered valid.
        max_rivers (int):
            Maximum number of rivers to generate.
        rainfall_threshold (float):
            Minimum rainfall value (0-1) for river sources.

    """
    if not map_data.rainfall_map or not map_data.elevation_map:
        logger.warning("Cannot generate rivers without rainfall and elevation data")
        return

    # Find potential river sources (high rainfall areas)
    sources = _find_river_sources(map_data, rainfall_threshold)

    if not sources:
        logger.debug("No suitable river sources found")
        return

    # Limit number of rivers
    sources = sources[:max_rivers]

    logger.debug(
        f"Generating up to {len(sources)} rivers from {len(_find_river_sources(map_data, 0.0))} potential sources"
    )

    rivers_generated = 0
    for source in sources:
        river_path = _generate_river_path(map_data, source, min_river_length)
        if river_path and len(river_path) >= min_river_length:
            _apply_river_to_map(map_data, river_path)
            rivers_generated += 1
            logger.debug(
                f"Generated river {rivers_generated} with {len(river_path)} tiles"
            )
        else:
            logger.debug(
                f"River from {source} failed: path_length={len(river_path) if river_path else 0}, min_length={min_river_length}"
            )

    logger.debug(f"Generated {rivers_generated} rivers")


def _find_river_sources(
    map_data: MapData,
    rainfall_threshold: float,
) -> list[Position]:
    """
    Find potential river sources in high rainfall areas.

    Args:
        map_data (MapData):
            The map data.
        rainfall_threshold (float):
            Minimum rainfall value for sources.

    Returns:
        list[Position]:
            List of potential river source positions.

    """
    sources = []
    elevations = []
    rainfall_values = []

    # Sample every tile for more candidates
    sample_step = 1

    for y in range(0, map_data.height, sample_step):
        for x in range(0, map_data.width, sample_step):
            rainfall = map_data.get_rainfall(x, y)
            if rainfall >= rainfall_threshold:
                # Optionally: relax local max constraint for more sources
                # neighbors = map_data.get_neighbors(x, y)
                # is_local_max = True
                # for neighbor in neighbors:
                #     if map_data.get_rainfall(neighbor.x, neighbor.y) > rainfall:
                #         is_local_max = False
                #         break
                # if is_local_max:
                sources.append(Position(x, y))
                elevations.append(map_data.get_elevation(x, y))
                rainfall_values.append(rainfall)

    logger.debug(
        f"Found {len(sources)} candidate river sources (rainfall >= {rainfall_threshold})"
    )
    if sources:
        logger.debug(
            f"Rainfall range of sources: min={min(rainfall_values):.3f}, max={max(rainfall_values):.3f}, mean={sum(rainfall_values)/len(rainfall_values):.3f}"
        )
        logger.debug(
            f"Elevation range of sources: min={min(elevations):.3f}, max={max(elevations):.3f}, mean={sum(elevations)/len(elevations):.3f}"
        )

    # Sort by rainfall (highest first) and add some randomness
    sources.sort(key=lambda pos: map_data.get_rainfall(pos.x, pos.y), reverse=True)
    random.shuffle(sources[: len(sources) // 2])
    return sources


def _generate_river_path(
    map_data: MapData,
    start: Position,
    min_length: int,
) -> list[Position] | None:
    """
    Generate a river path from source to endpoint using downhill flow.

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

    max_iterations = min_length * 3  # Prevent infinite loops
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Find the lowest elevation neighbor
        neighbors = map_data.get_neighbors(current.x, current.y, walkable_only=False)
        if not neighbors:
            break

        # Filter out visited positions and find lowest elevation
        candidates = []
        for neighbor in neighbors:
            if neighbor not in visited:
                elevation = map_data.get_elevation(neighbor.x, neighbor.y)
                candidates.append((elevation, neighbor))

        if not candidates:
            break

        # Sort by elevation (lowest first)
        candidates.sort(key=lambda x: x[0])

        # Choose the lowest elevation neighbor
        lowest_elevation, next_pos = candidates[0]

        # Check if we're still flowing downhill
        current_elevation = map_data.get_elevation(current.x, current.y)
        if lowest_elevation >= current_elevation:
            # We've reached a local minimum, stop here
            break

        # Check if we've reached water (ocean/lake)
        current_tile = map_data.get_terrain(current.x, current.y)
        if current_tile.is_water:
            # River has reached water, stop here
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
    # Find the river tile
    river_tile = None
    for tile in map_data.tiles:
        if tile.name == "river":
            river_tile = tile
            break

    if not river_tile:
        logger.warning("No river tile found in tile catalog")
        return

    # Apply river tiles along the path
    for position in river_path:
        # Only replace non-water tiles with river
        current_tile = map_data.get_terrain(position.x, position.y)
        if not current_tile.is_water:
            map_data.set_terrain(position.x, position.y, river_tile)
