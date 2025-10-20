"""Road generation module."""

import random

import numpy as np
from sklearn.neighbors import KDTree as SKLearnKDTree

from . import logger
from .map_data import MapData, Position, Road, Settlement


def generate_roads(
    map_data: MapData,
) -> None:
    """
    Generate road network connecting settlements.

    Args:
        map_data (MapData):
            The map grid.
        noise_map (np.ndarray):
            The elevation map.

    """
    if not map_data.settlements:
        logger.info("No settlements to connect; skipping road generation.")
        return

    # Shuffle settlements to randomize connection order
    shuffled_settlements = random.sample(
        map_data.settlements, len(map_data.settlements)
    )

    for settlement in shuffled_settlements:
        # Find nearest settlement not already connected
        nearest = _find_nearest_settlement_not_connected(
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

        logger.info(f"Connected {settlement.name} to {nearest.name} with a road.")


def _reconstruct_path(
    current: Position,
    came_from: dict[Position, Position],
) -> list[Position]:
    """Reconstruct the path from A* search.

    Args:
        current (Position): The current position.
        came_from (Dict[Position, Position]): The came_from dictionary.

    Returns:
        List[Position]: The reconstructed path.

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
    """Perform A* search.

    Args:
        map_data (MapData): The map grid.
        start (Position): The start position.
        goal (Position): The goal position.

    Returns:
        list[Position] | None: The path if found, None otherwise.

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
        current_pos, current_cost, _current_heuristic = current

        if current_pos == goal:
            return _reconstruct_path(current_pos, came_from)

        open_set.remove(current)
        closed_set.add(current_pos)

        for neighbor in map_data.get_neighbors(current_pos.x, current_pos.y):
            if neighbor in closed_set:
                continue

            tile = map_data.get_terrain(neighbor.x, neighbor.y)
            tentative_cost = current_cost + tile.pathfinding_cost

            if neighbor in [n[0] for n in open_set]:
                if tentative_cost < next(n[1] for n in open_set if n[0] == neighbor):
                    for i, node in enumerate(open_set):
                        if node[0] == neighbor:
                            open_set[i] = (
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


def _settlements_are_connected(
    map_data: MapData,
    s1: str,
    s2: str,
) -> bool:
    """
    Check if two settlements are already connected by a road.

    Args:
        map_data (MapData):
            The map data.
        s1 (str):
            Name of first settlement.
        s2 (str):
            Name of second settlement.

    Returns:
        bool: True if connected, False otherwise.
    """
    for road in map_data.roads:
        if (road.start_settlement == s1 and road.end_settlement == s2) or (
            road.start_settlement == s2 and road.end_settlement == s1
        ):
            return True
    return False


def _find_nearest_settlement_not_connected(
    settlement: Settlement,
    settlements: list[Settlement],
    map_data: MapData,
) -> Settlement | None:
    """
    Find the nearest settlement not already connected to the given settlement.

    Args:
        settlement: The settlement to connect from.
        settlements: All settlements.
        map_data: The map data containing existing roads.

    Returns:
        The nearest unconnected settlement, or None.
    """
    nearest = None
    min_dist = float("inf")
    for other in settlements:
        if other.name == settlement.name:
            continue
        if _settlements_are_connected(map_data, settlement.name, other.name):
            continue
        dist = settlement.distance_to(other)
        if dist < min_dist:
            min_dist = dist
            nearest = other
    return nearest
