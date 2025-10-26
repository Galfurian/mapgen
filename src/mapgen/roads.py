"""
Road generation module for procedural map generation.

This module handles the generation of road networks connecting settlements.
It uses A* pathfinding to create efficient routes between settlements,
ensuring realistic transportation infrastructure on the map.
"""

import heapq
import logging
import random

from .map_data import MapData, Position, Road, Settlement
from .utils import a_star_search, compute_terrain_control_point, quadratic_bezier_points

logger = logging.getLogger(__name__)


def generate_roads(
    map_data: MapData,
) -> None:
    """
    Generate road network connecting settlements.

    Args:
        map_data (MapData):
            The map grid.

    """
    if not map_data.settlements:
        logger.warning("No settlements to connect; skipping road generation.")
        return

    # Shuffle settlements to randomize connection order.
    shuffled_settlements = random.sample(
        map_data.settlements, len(map_data.settlements)
    )

    # Connect each settlement to the nearest unconnected one.
    for settlement in shuffled_settlements:
        # Find nearest settlement worth connecting.
        result = _find_nearest_settlement_worth_connecting(
            settlement,
            map_data.settlements,
            map_data,
        )
        if not result:
            logger.debug(f"No unconnected settlements found for {settlement.name}")
            continue

        # Unpack result.
        nearest, path = result

        # Curve the path for more natural road appearance
        path = _curve_road_path(path, map_data)

        # Check if road already exists
        if _road_exists(map_data, settlement.name, nearest.name):
            logger.debug(
                f"Road between {settlement.name} and {nearest.name} already exists"
            )
            continue
        # Add the road to the map data.
        map_data.roads.append(
            Road(
                start_settlement=settlement.name,
                end_settlement=nearest.name,
                path=path,
            )
        )

        logger.debug(
            f"Connected {settlement.name} to {nearest.name} with road of length {len(path)}"
        )

    num_road_tiles = sum([len(road.path) for road in map_data.roads])

    logger.debug(f"Traced {len(map_data.roads)} roads")
    logger.info(f"Placed {num_road_tiles} road tiles")


def _road_placement_validation(map_data: MapData, pos: Position) -> bool:
    """
    This function checks if a road can be placed at the given position.

    Args:
        map_data (MapData):
            The map data.
        pos (Position):
            The position to validate.

    Returns:
        bool:
            True if a road can be placed, False otherwise.
    """
    # First, check if position is within map bounds.
    if not map_data.is_valid_position(pos.x, pos.y):
        return False
    # Then, check terrain type.
    terrain = map_data.get_terrain(pos.x, pos.y)
    if not terrain.is_walkable:
        return False
    if terrain.is_salt_water:
        return False
    if terrain.is_water and not terrain.is_flowing_water:
        return False
    return True


def _curve_road_path(
    path: list[Position],
    map_data: MapData,
) -> list[Position]:
    """
    Aggressively simplify the path by trying to interpolate directly to the
    farthest valid end point, falling back to closer points if invalid. Uses
    Bezier curves for smooth bends.

    Args:
        path (list[Position]):
            The path to simplify and curve.
        map_data (MapData):
            The map data containing elevation and terrain.

    Returns:
        list[Position]:
            The simplified, curved path.

    """
    if len(path) <= 2:
        return path

    result = [path[0]]
    current_idx = 0

    while current_idx < len(path) - 1:
        start = path[current_idx]
        found = False

        # Try farthest end first
        for end_idx in range(len(path) - 1, current_idx, -1):
            end = path[end_idx]

            # Compute control point
            control = compute_terrain_control_point(start, end, map_data)

            # Generate Bezier points
            bezier_points = quadratic_bezier_points(start, control, end, num_points=20)

            # Check validity.
            if all(_road_placement_validation(map_data, pos) for pos in bezier_points):
                # Valid: add the curve points (skip start).
                result.extend(bezier_points[1:])
                current_idx = end_idx
                found = True
                break

        # No valid jump: add next point
        if not found:
            current_idx += 1
            if current_idx < len(path):
                result.append(path[current_idx])

    return result


def _find_nearest_settlement_worth_connecting(
    settlement: Settlement,
    settlements: list[Settlement],
    map_data: MapData,
) -> tuple[Settlement, list[Position]] | None:
    """
    Find the nearest settlement that can be connected with a reasonable path.

    Args:
        settlement (Settlement):
            The settlement to connect from.
        settlements (list[Settlement]):
            All settlements.
        map_data (MapData):
            The map data containing existing roads.

    Returns:
        tuple[Settlement, list[Position]] | None:
            The nearest settlement and its path, or None.

    """
    # Get candidates: settlements sorted by direct distance
    candidates = sorted(
        [s for s in settlements if s.name != settlement.name],
        key=lambda s: settlement.distance_to(s),
    )
    # Only consider top 5 nearest.
    candidates = candidates[:5]

    best = None
    best_path = None
    best_path_length = float("inf")
    for other in candidates:
        # Check if already connected via roads
        path_dist = _shortest_path_distance(map_data, settlement.name, other.name)
        if path_dist is not None and path_dist <= settlement.distance_to(other) * 0.6:
            continue
        # Compute actual path
        path = a_star_search(
            map_data,
            settlement.position,
            other.position,
            _road_placement_validation,
        )
        if path and len(path) < best_path_length:
            best = other
            best_path = path
            best_path_length = len(path)
    if best is not None and best_path is not None:
        return (best, best_path)
    return None


def _shortest_path_distance(
    map_data: MapData,
    start_name: str,
    end_name: str,
) -> float | None:
    """
    Compute the shortest path distance between two settlements via existing
    roads.

    Args:
        map_data (MapData):
            The map data.
        start_name (str):
            Name of starting settlement.
        end_name (str):
            Name of ending settlement.

    Returns:
        float | None:
            The shortest distance, or None if no path exists.

    """
    # Create a dictionary of settlements by name.
    settlements = {s.name: s for s in map_data.settlements}
    # Initialize distances to infinity.
    dist = {name: float("inf") for name in settlements}
    dist[start_name] = 0.0
    # Priority queue for Dijkstra.
    pq = [(0.0, start_name)]

    # Dijkstra's algorithm loop.
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for road in map_data.roads:
            v = None
            if road.start_settlement == u:
                v = road.end_settlement
            elif road.end_settlement == u:
                v = road.start_settlement
            if v is None:
                continue
            alt = d + settlements[u].distance_to(settlements[v])
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(pq, (alt, v))

    if dist[end_name] == float("inf"):
        return None
    return dist[end_name]


def _road_exists(map_data: MapData, start_name: str, end_name: str) -> bool:
    """
    Check if a road exists between two settlements.

    Args:
        map_data (MapData):
            The map data.
        start_name (str):
            Name of the start settlement.
        end_name (str):
            Name of the end settlement.

    Returns:
        bool:
            True if a road exists, False otherwise.

    """
    return any(
        (road.start_settlement == start_name and road.end_settlement == end_name)
        or (road.start_settlement == end_name and road.end_settlement == start_name)
        for road in map_data.roads
    )
