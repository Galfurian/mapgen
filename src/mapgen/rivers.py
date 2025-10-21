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
    Find potential river sources: prioritize high-elevation lakes, then springs (high rainfall, high elevation, local minima).

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

    # First, add lake centers as sources, prioritizing high-elevation lakes
    lake_sources = []
    for lake in map_data.lakes:
        # Use lake center as source, but only if not coastal and has some elevation
        if lake.mean_elevation > 0.3:  # Lower threshold for lakes
            # Check if lake is not adjacent to sea (coastal)
            center_x, center_y = lake.center
            is_coastal = False
            for pos in lake.tiles:
                neighbors = map_data.get_neighbors(pos.x, pos.y, walkable_only=False)
                for neighbor in neighbors:
                    neighbor_tile = map_data.get_terrain(neighbor.x, neighbor.y)
                    if neighbor_tile.is_water and neighbor_tile.is_salt_water:
                        is_coastal = True
                        break
                if is_coastal:
                    break
            
            if not is_coastal:
                lake_sources.append((lake.center, lake.mean_elevation, lake.total_accumulation))
    
    # Sort lakes by elevation (highest first), then by accumulation
    lake_sources.sort(key=lambda x: (x[1], x[2]), reverse=True)
    for center, _, _ in lake_sources:
        sources.append(Position(int(center[0]), int(center[1])))

    # Then, find spring sources: local minima with high rainfall and elevation
    spring_sources = []
    elevations = []
    rainfall_values = []

    sample_step = 1
    for y in range(0, map_data.height, sample_step):
        for x in range(0, map_data.width, sample_step):
            rainfall = map_data.get_rainfall(x, y)
            elevation = map_data.get_elevation(x, y)
            current_tile = map_data.get_terrain(x, y)
            
            # Higher elevation threshold and ensure not on water
            if (rainfall >= rainfall_threshold and 
                elevation > 0.5 and  # Lower elevation threshold
                not current_tile.is_water):  # Not on existing water
                
                # Check if local minimum
                neighbors = map_data.get_neighbors(x, y, walkable_only=False)
                is_local_min = True
                for neighbor in neighbors:
                    if map_data.get_elevation(neighbor.x, neighbor.y) < elevation:
                        is_local_min = False
                        break
                if is_local_min:
                    spring_sources.append(Position(x, y))
                    elevations.append(elevation)
                    rainfall_values.append(rainfall)

    # Sort springs by rainfall and elevation
    spring_sources.sort(key=lambda pos: (map_data.get_rainfall(pos.x, pos.y), map_data.get_elevation(pos.x, pos.y)), reverse=True)
    sources.extend(spring_sources)

    logger.debug(
        f"Found {len(sources)} river sources: {len(lake_sources)} from lakes, {len(spring_sources)} from springs"
    )
    if sources:
        elevations_all = [map_data.get_elevation(p.x, p.y) for p in sources]
        rainfall_all = [map_data.get_rainfall(p.x, p.y) for p in sources]
        logger.debug(
            f"Source rainfall: min={min(rainfall_all):.3f}, max={max(rainfall_all):.3f}"
        )
        logger.debug(
            f"Source elevation: min={min(elevations_all):.3f}, max={max(elevations_all):.3f}"
        )

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

        # Check if next position is salt water (ocean) - rivers stop at sea
        next_tile = map_data.get_terrain(next_pos.x, next_pos.y)
        if next_tile.is_water and next_tile.is_salt_water:
            # River has reached the sea, stop here (don't add sea tiles to path)
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
        # Only replace non-water tiles with river
        current_tile = map_data.get_terrain(position.x, position.y)
        if not current_tile.is_water:
            map_data.set_terrain(position.x, position.y, river_tile)
