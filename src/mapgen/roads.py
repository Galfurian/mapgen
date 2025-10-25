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
        nearest = _find_nearest_settlement_worth_connecting(
            settlement,
            map_data.settlements,
            map_data,
        )
        if not nearest:
            logger.debug(f"No unconnected settlements found for {settlement.name}")
            continue
        # Find the path between settlements.
        path = _a_star_search(
            map_data,
            settlement.position,
            nearest.position,
        )
        if path is None:
            logger.debug(f"No path found between {settlement.name} and {nearest.name}")
            continue
        # Add the road to the map data.
        map_data.roads.append(
            Road(
                start_settlement=settlement.name,
                end_settlement=nearest.name,
                path=path,
            )
        )

    num_road_tiles = sum([len(road.path) for road in map_data.roads])

    logger.debug(f"Traced {len(map_data.roads)} roads")
    logger.info(f"Placed {num_road_tiles} road tiles")


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


def _a_star_search(
    map_data: MapData,
    start: Position,
    goal: Position,
) -> list[Position] | None:
    """
    Perform A* search.

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

    # Priority queue of (position, cost_so_far, total_estimated_cost)
    open_set: list[tuple[Position, float, float]] = []
    # Set of positions already evaluated
    closed_set = set()
    # Dictionary mapping position to its predecessor in the path
    came_from: dict[Position, Position] = {}

    start_node = (start, 0.0, heuristic(start, goal))
    open_set.append(start_node)

    # Main A* loop.
    while open_set:
        # Find the node with the lowest total estimated cost.
        current = min(open_set, key=lambda x: x[2])
        current_pos, current_cost, _current_heuristic = current

        if current_pos == goal:
            return _reconstruct_path(current_pos, came_from)

        open_set.remove(current)
        closed_set.add(current_pos)

        # Explore neighbors.
        for neighbor in map_data.get_neighbors(current_pos.x, current_pos.y):
            if neighbor in closed_set:
                continue

            tile = map_data.get_terrain(neighbor.x, neighbor.y)
            tentative_cost = current_cost + tile.pathfinding_cost

            # Check if neighbor is already in open set.
            existing_node = next((n for n in open_set if n[0] == neighbor), None)
            if existing_node:
                # If this path is better, update the node.
                if tentative_cost < existing_node[1]:
                    idx = open_set.index(existing_node)
                    open_set[idx] = (
                        neighbor,
                        tentative_cost,
                        tentative_cost + heuristic(neighbor, goal),
                    )
                    came_from[neighbor] = current_pos
            else:
                # Add new node to open set.
                open_set.append(
                    (
                        neighbor,
                        tentative_cost,
                        tentative_cost + heuristic(neighbor, goal),
                    )
                )
                came_from[neighbor] = current_pos

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


def _find_nearest_settlement_worth_connecting(
    settlement: Settlement,
    settlements: list[Settlement],
    map_data: MapData,
) -> Settlement | None:
    """
    Find the nearest settlement where direct connection is better than existing
    paths.

    Args:
        settlement (Settlement):
            The settlement to connect from.
        settlements (list[Settlement]):
            All settlements.
        map_data (MapData):
            The map data containing existing roads.

    Returns:
        Settlement | None:
            The nearest settlement worth connecting to, or None.

    """
    # Initialize variables for nearest settlement.
    nearest = None
    min_dist = float("inf")
    # Check each other settlement.
    for other in settlements:
        if other.name == settlement.name:
            continue
        # Calculate direct distance.
        direct_dist = settlement.distance_to(other)
        # Get shortest path distance via roads.
        path_dist = _shortest_path_distance(map_data, settlement.name, other.name)
        # If direct is better or no path exists.
        if path_dist is None or path_dist > direct_dist:
            # Update if closer.
            if direct_dist < min_dist:
                min_dist = direct_dist
                nearest = other
    return nearest
