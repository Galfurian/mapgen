"""
Water routes generation module for procedural map generation.

This module handles the generation of water routes connecting harbors.
It uses A* pathfinding to create efficient routes between harbors,
allowing travel through water bodies.
"""

import heapq
import logging

from .map_data import MapData, Position, Settlement, WaterRoute

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
        nearest = _find_nearest_harbor_worth_connecting(harbor, harbors, map_data)
        if not nearest:
            logger.debug(f"No suitable harbor found for {harbor.name}")
            continue
        # Find nearest water tiles for start and goal
        start_water = _find_nearest_water_tile(map_data, harbor.position)
        goal_water = _find_nearest_water_tile(map_data, nearest.position)
        if not start_water or not goal_water:
            logger.debug(f"No water tiles found near {harbor.name} or {nearest.name}")
            continue
        # Find the path
        path = _a_star_water(map_data, start_water, goal_water)
        if path is None:
            logger.debug(f"No water path found between {harbor.name} and {nearest.name}")
            continue
        # Add the route (no need to check exists since we already did in worth_connecting)
        map_data.water_routes.append(
            WaterRoute(
                start_harbor=harbor.name,
                end_harbor=nearest.name,
                path=path,
            )
        )
        logger.debug(f"Connected {harbor.name} to {nearest.name} with water route of length {len(path)}")

    num_route_tiles = sum([len(route.path) for route in map_data.water_routes])
    logger.debug(f"Traced {len(map_data.water_routes)} water routes")
    logger.info(f"Placed {num_route_tiles} water route tiles")


def _find_nearest_harbor_worth_connecting(
    harbor: Settlement,
    harbors: list[Settlement],
    map_data: MapData,
) -> Settlement | None:
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
        Settlement | None:
            The nearest harbor that can be connected, or None.

    """
    # Get candidates: harbors sorted by direct distance
    candidates = sorted(
        [h for h in harbors if h.name != harbor.name],
        key=lambda h: harbor.distance_to(h),
    )[
        :5
    ]  # Top 5 nearest

    best = None
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
        path = _a_star_water(map_data, start_water, goal_water)
        if path and len(path) < best_path_length:
            best = other
            best_path_length = len(path)
    return best


def _shortest_water_route_distance(
    map_data: MapData,
    start_name: str,
    end_name: str,
) -> float | None:
    """
    Compute the shortest path distance between two harbors via existing
    water routes.

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


def _a_star_water(
    map_data: MapData, start: Position, goal: Position
) -> list[Position] | None:
    """
    Perform A* search for water routes, allowing water tiles.

    Args:
        map_data (MapData):
            The map grid.
        start (Position):
            The start position.
        goal (Position):
            The goal position.

    Returns:
        list[Position] | None:
            The path if found, None otherwise.

    """

    def heuristic(a: Position, b: Position) -> float:
        return a.manhattan_distance_to(b)

    open_set: list[tuple[Position, float, float]] = []
    closed_set = set()
    came_from: dict[Position, Position] = {}

    start_node = (start, 0.0, heuristic(start, goal))
    open_set.append(start_node)

    while open_set:
        current = min(open_set, key=lambda x: x[2])
        current_pos, current_cost, _ = current

        if current_pos == goal:
            return _reconstruct_path(current_pos, came_from)

        open_set.remove(current)
        closed_set.add(current_pos)

        for neighbor in map_data.get_neighbors(
            current_pos.x, current_pos.y, walkable_only=False
        ):
            if neighbor in closed_set:
                continue

            tile = map_data.get_terrain(neighbor.x, neighbor.y)
            # Only allow water tiles for water routes
            if not tile.is_water:
                continue
            tentative_cost = current_cost + 1.0  # Simple cost

            existing_node = next((n for n in open_set if n[0] == neighbor), None)
            if existing_node:
                if tentative_cost < existing_node[1]:
                    idx = open_set.index(existing_node)
                    open_set[idx] = (
                        neighbor,
                        tentative_cost,
                        tentative_cost + heuristic(neighbor, goal),
                    )
                    came_from[neighbor] = current_pos
            else:
                open_set.append(
                    (
                        neighbor,
                        tentative_cost,
                        tentative_cost + heuristic(neighbor, goal),
                    )
                )
                came_from[neighbor] = current_pos

    return None


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


def _reconstruct_path(
    current: Position,
    came_from: dict[Position, Position],
) -> list[Position]:
    """
    Reconstruct the path from A* search.

    Args:
        current (Position):
            The current position.
        came_from (dict[Position, Position]):
            The came_from dictionary.

    Returns:
        list[Position]:
            The reconstructed path.

    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
