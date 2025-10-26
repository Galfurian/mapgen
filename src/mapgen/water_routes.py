"""
Water routes generation module for procedural map generation.

This module handles the generation of water routes connecting harbors. It uses
A* pathfinding to create efficient routes between harbors, allowing travel
through water bodies.
"""

import heapq
import logging

from .map_data import MapData, Position, Settlement, WaterRoute
from .utils import a_star_search, compute_terrain_control_point, quadratic_bezier_points
from .utils import compute_terrain_control_point, quadratic_bezier_points

logger = logging.getLogger(__name__)


def generate_water_routes(map_data: MapData) -> None:
    """
    Generate water routes connecting harbors.

    Args:
        map_data (MapData):
            The map grid.

    """
    harbors = [s for s in map_data.settlements if s.is_harbor]
    if len(harbors) < 2:
        logger.debug("Not enough harbors to connect")
        return

    # Connect each harbor to the nearest other harbor
    for harbor in harbors:
        # Find nearest harbor worth connecting via water
        result = _find_nearest_harbor_worth_connecting(
            harbor,
            harbors,
            map_data,
        )
        if not result:
            logger.debug(f"No suitable harbor found for {harbor.name}")
            continue

        # Unpack result
        nearest, path = result

        if _water_route_exists(map_data, harbor.name, nearest.name):
            logger.debug(
                f"Water route between {harbor.name} and {nearest.name} already exists"
            )
            continue

        # Curve the path for more natural water route appearance
        path = _curve_water_route_path(path, map_data)

        # Add the route (no need to check exists since we already did in
        # worth_connecting).
        map_data.water_routes.append(
            WaterRoute(
                start_harbor=harbor.name,
                end_harbor=nearest.name,
                path=path,
            )
        )
        logger.debug(
            f"Connected {harbor.name} to {nearest.name} with water route of length {len(path)}"
        )

    num_route_tiles = sum([len(route.path) for route in map_data.water_routes])
    logger.debug(f"Traced {len(map_data.water_routes)} water routes")
    logger.info(f"Placed {num_route_tiles} water route tiles")


def _find_nearest_harbor_worth_connecting(
    harbor: Settlement,
    harbors: list[Settlement],
    map_data: MapData,
) -> tuple[Settlement, list[Position]] | None:
    """
    Find the nearest harbor that can be connected with a reasonable water path.

    Args:
        harbor (Settlement):
            The harbor to connect from.
        harbors (list[Settlement]):
            All harbors.
        map_data (MapData):
            The map data containing existing water routes.

    Returns:
        tuple[Settlement, list[Position]] | None:
            The nearest harbor and path, or None.

    """
    # Get candidates: harbors sorted by direct distance
    candidates = sorted(
        [h for h in harbors if h.name != harbor.name],
        key=lambda h: harbor.distance_to(h),
    )
    # Only consider top 5 nearest.
    candidates = candidates[:5]

    best = None
    best_path = None
    best_path_length = float("inf")
    for other in candidates:
        # Check if already connected via water routes (direct or indirect)
        route_dist = _shortest_water_route_distance(map_data, harbor.name, other.name)
        if route_dist is not None:
            continue
        # Find water tiles
        start_water = _find_nearest_water_tile(map_data, harbor.position)
        goal_water = _find_nearest_water_tile(map_data, other.position)
        if not start_water or not goal_water:
            continue
        # Compute actual water path
        path = a_star_search(
            map_data, start_water, goal_water, _water_route_placement_validation
        )
        if path and len(path) < best_path_length:
            best = other
            best_path = path
            best_path_length = len(path)
    if best is None or best_path is None:
        return None
    return (best, best_path)


def _shortest_water_route_distance(
    map_data: MapData,
    start_name: str,
    end_name: str,
) -> float | None:
    """
    Compute the shortest path distance between two harbors via existing water
    routes.

    Args:
        map_data (MapData):
            The map data.
        start_name (str):
            Name of starting harbor.
        end_name (str):
            Name of ending harbor.

    Returns:
        float | None:
            The shortest distance, or None if no path exists.

    """
    # Create a dictionary of harbors by name.
    harbors = {s.name: s for s in map_data.settlements if s.is_harbor}
    # Initialize distances to infinity.
    dist = {name: float("inf") for name in harbors}
    dist[start_name] = 0.0
    # Priority queue for Dijkstra.
    pq = [(0.0, start_name)]

    # Dijkstra's algorithm loop.
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for route in map_data.water_routes:
            v = None
            if route.start_harbor == u:
                v = route.end_harbor
            elif route.end_harbor == u:
                v = route.start_harbor
            if v is None:
                continue
            alt = d + harbors[u].distance_to(harbors[v])
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(pq, (alt, v))

    if dist[end_name] == float("inf"):
        return None
    return dist[end_name]


def _find_nearest_water_tile(map_data: MapData, position: Position) -> Position | None:
    """
    Find the nearest water tile to a given position.

    Args:
        map_data (MapData):
            The map data.
        position (Position):
            The position to search from.

    Returns:
        Position | None:
            The nearest water tile position, or None if not found.

    """
    min_dist = float("inf")
    nearest = None
    for y in range(map_data.height):
        for x in range(map_data.width):
            tile = map_data.get_terrain(x, y)
            if tile.is_water:
                dist = ((x - position.x) ** 2 + (y - position.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = Position(x, y)
    return nearest


def _water_route_exists(map_data: MapData, start_name: str, end_name: str) -> bool:
    """
    Check if a water route exists between two harbors.

    Args:
        map_data (MapData):
            The map data.
        start_name (str):
            Name of the start harbor.
        end_name (str):
            Name of the end harbor.

    Returns:
        bool:
            True if a route exists, False otherwise.

    """
    return any(
        (route.start_harbor == start_name and route.end_harbor == end_name)
        or (route.start_harbor == end_name and route.end_harbor == start_name)
        for route in map_data.water_routes
    )


def _curve_water_route_path(
    path: list[Position],
    map_data: MapData,
) -> list[Position]:
    """
    Aggressively simplify the water path by trying to interpolate directly to
    the farthest valid end point, falling back to closer points if invalid. Uses
    Bezier curves with inverted gradients to follow deeper water channels.

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

            # Compute control point with inverted gradients (follow deeper
            # water)
            control = compute_terrain_control_point(
                start,
                end,
                map_data,
                control_factor=1.5,
                invert_gradients=True,
            )

            # Generate Bezier points
            bezier_points = quadratic_bezier_points(
                start,
                control,
                end,
                num_points=15,
            )

            # Check validity (must be in salt water)
            if all(
                _water_route_placement_validation(map_data, pos)
                for pos in bezier_points
            ):
                # Valid: add the curve points (skip start)
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


def _water_route_placement_validation(map_data: MapData, pos: Position) -> bool:
    """
    Validate if a water route can be placed at the given position.

    Args:
        map_data (MapData):
            The map data.
        pos (Position):
            The position to validate.

    Returns:
        bool:
            True if valid for water route placement, False otherwise.
    """
    # First, check if position is within map bounds.
    if not map_data.is_valid_position(pos.x, pos.y):
        return False
    # Then, check terrain type.
    return map_data.get_terrain(pos.x, pos.y).is_salt_water
