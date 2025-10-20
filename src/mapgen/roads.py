"""Road generation module."""

import random

import numpy as np
from sklearn.neighbors import KDTree as SKLearnKDTree

from .map_data import MapData, Position, Road, RoadType
from . import logger


def reconstruct_path(
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


def a_star_search(
    map_data: MapData,
    start: Position,
    goal: Position,
    high_points: list[Position],
    high_point_penalty: int = 5,
) -> list[Position] | None:
    """Perform A* search with high point avoidance.

    Args:
        map_data (MapData): The map grid.
        start (Position): The start position.
        goal (Position): The goal position.
        high_points (List[Position]): List of high points to avoid.
        high_point_penalty (int): Penalty for high points.

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
        current_pos, current_cost, current_heuristic = current

        if current_pos == goal:
            return reconstruct_path(current_pos, came_from)

        open_set.remove(current)
        closed_set.add(current_pos)

        for neighbor in map_data.get_neighbors(current_pos.x, current_pos.y):
            if neighbor in closed_set:
                continue

            tile = map_data.get_terrain(neighbor.x, neighbor.y)
            base_cost = tile.pathfinding_cost
            high_point_cost = high_point_penalty if neighbor in high_points else 0
            tentative_cost = current_cost + base_cost + high_point_cost

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


def generate_roads(
    map_data: MapData,
    noise_map: np.ndarray,
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

    roads = map_data.roads
    settlements = map_data.settlements

    # 1. Connect settlements using a simple nearest neighbor approach
    # (instead of full MST to avoid NetworkX)
    connected = set()
    for settlement in settlements:
        if settlement.name in connected:
            continue
        # Find nearest unconnected settlement
        nearest = None
        min_dist = float('inf')
        for other in settlements:
            if other.name != settlement.name and other.name not in connected:
                dist = settlement.distance_to(other)
                if dist < min_dist:
                    min_dist = dist
                    nearest = other
        if nearest:
            path = a_star_search(
                map_data,
                settlement.position,
                nearest.position,
                high_points=[],
            )
            if path is not None:
                # Determine road type
                has_water_tiles = any(
                    not map_data.get_terrain(pos.x, pos.y).can_build_road for pos in path
                )
                road_type = RoadType.WATER if has_water_tiles else RoadType.LAND
                roads.append(Road(
                    start_settlement=settlement.name,
                    end_settlement=nearest.name,
                    type=road_type,
                    path=path,
                ))
                connected.add(settlement.name)
                connected.add(nearest.name)

    # 2. Identify High Points (simplified, without matplotlib contours)
    high_points = []
    # For simplicity, consider points above a threshold as high points
    threshold = np.percentile(noise_map, 80)
    for y in range(noise_map.shape[0]):
        for x in range(noise_map.shape[1]):
            if noise_map[y, x] > threshold:
                high_points.append(Position(x=x, y=y))

    # 3. Add Additional Connections (Avoiding High Points)
    settlement_positions = np.array([(s.position.x, s.position.y) for s in settlements])
    kdtree = SKLearnKDTree(settlement_positions)

    existing_connections = {(r.start_settlement, r.end_settlement) for r in roads}
    existing_connections.update({(r.end_settlement, r.start_settlement) for r in roads})

    for i, settlement1 in enumerate(settlements):
        neighbor_indices = kdtree.query_radius(
            [(settlement1.position.x, settlement1.position.y)], r=40
        )[0]
        for j in neighbor_indices:
            settlement2 = settlements[j]
            if i != j and (settlement1.name, settlement2.name) not in existing_connections:
                distance = settlement1.distance_to(settlement2)

                connection_probability = (
                    settlement1.connectivity * settlement2.connectivity
                ) / (distance * 5)

                if random.random() < connection_probability:
                    path = a_star_search(
                        map_data,
                        settlement1.position,
                        settlement2.position,
                        high_points,
                    )
                    if path is not None:
                        # Determine road type
                        has_water_tiles = any(
                            not map_data.get_terrain(pos.x, pos.y).can_build_road
                            for pos in path
                        )
                        road_type = RoadType.WATER if has_water_tiles else RoadType.LAND
                        roads.append(Road(
                            start_settlement=settlement1.name,
                            end_settlement=settlement2.name,
                            type=road_type,
                            path=path,
                        ))
                        existing_connections.add((settlement1.name, settlement2.name))
                        existing_connections.add((settlement2.name, settlement1.name))
