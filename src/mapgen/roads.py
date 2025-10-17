"""Road generation module."""

import random
from typing import Any

import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree as SKLearnKDTree


def reconstruct_path(
    current: tuple[int, int], came_from: dict
) -> list[tuple[int, int]]:
    """Reconstruct path from A* search.

    Args:
        current (Tuple[int, int]): The current position.
        came_from (dict): The came_from dictionary from A* search.

    Returns:
        List[Tuple[int, int]]: The reconstructed path.

    """
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]


def get_neighbors(
    level: list[list[str]], pos: tuple[int, int]
) -> list[tuple[int, int]]:
    """Get valid neighboring positions.

    Args:
        level (List[List[str]]): The level grid.
        pos (Tuple[int, int]): The current position.

    Returns:
        List[Tuple[int, int]]: List of valid neighboring positions.

    """
    x, y = pos
    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    valid_neighbors = []
    for neighbor in neighbors:
        nx_pos, ny_pos = neighbor
        if (
            0 <= nx_pos < len(level[0])
            and 0 <= ny_pos < len(level)
            and level[ny_pos][nx_pos] not in ("#", "M")
        ):
            valid_neighbors.append(neighbor)
    return valid_neighbors


def a_star_search(
    level: list[list[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    elevation_map: np.ndarray,
    high_points: list[tuple[int, int]],
    high_point_penalty: int = 5,
) -> list[tuple[int, int]] | None:
    """Perform A* search with high point avoidance.

    Args:
        level (List[List[str]]): The level grid.
        start (Tuple[int, int]): The start position.
        goal (Tuple[int, int]): The goal position.
        elevation_map (np.ndarray): The elevation map.
        high_points (List[Tuple[int, int]]): List of high points to avoid.
        high_point_penalty (int): Penalty for high points.

    Returns:
        Optional[List[Tuple[int, int]]]: The path if found, None otherwise.

    """

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    closed_set = set()
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    start_node = (start, 0, heuristic(start, goal))
    open_set.append(start_node)

    while open_set:
        current = min(open_set, key=lambda x: x[2])
        current_pos, current_cost, current_heuristic = current

        if current_pos == goal:
            return reconstruct_path(current_pos, came_from)

        open_set.remove(current)
        closed_set.add(current_pos)

        for neighbor in get_neighbors(level, current_pos):
            if neighbor in closed_set:
                continue

            base_cost = 2 if level[neighbor[1]][neighbor[0]] == "W" else 1
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


def find_enclosed_points(
    contour_data: Any, level_value: float, elevation_map: np.ndarray
) -> list[tuple[int, int]]:
    """Find points enclosed within a contour line.

    Args:
        contour_data: The contour data from matplotlib.
        level_value (float): The elevation level value.
        elevation_map (np.ndarray): The elevation map.

    Returns:
        List[Tuple[int, int]]: List of enclosed points.

    """
    paths = contour_data.allsegs[0]
    enclosed_points = []
    for path in paths:
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if (elevation_map[y1, x1] > level_value) != (
                elevation_map[y2, x2] > level_value
            ):
                enclosed_points.append((x1, y1))
    return enclosed_points


def generate_roads(
    settlements: list[dict],
    level: list[list[str]],
    elevation_map: np.ndarray,
) -> nx.Graph:
    """Generate road network connecting settlements.

    Args:
        settlements (List[Dict]): List of settlements.
        level (List[List[str]]): The level grid.
        elevation_map (np.ndarray): The elevation map.

    Returns:
        nx.Graph: The road network graph.

    """
    graph: nx.Graph = nx.Graph()
    for settlement in settlements:
        graph.add_node(settlement["name"], pos=(settlement["x"], settlement["y"]))

    if not settlements:
        return graph

    # 1. Connect settlements using Minimum Spanning Tree (MST)
    positions = np.array([(s["x"], s["y"]) for s in settlements])
    distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2
    )
    mst = nx.minimum_spanning_tree(nx.Graph(distances))

    for i, j in mst.edges():
        settlement1 = settlements[i]
        settlement2 = settlements[j]
        path = a_star_search(
            level,
            (settlement1["x"], settlement1["y"]),
            (settlement2["x"], settlement2["y"]),
            elevation_map,
            high_points=[],
        )
        if path is not None:
            road_type = "land"
            for px, py in path:
                if level[py][px] == "W":
                    road_type = "water"
                    break
            graph.add_edge(
                settlement1["name"], settlement2["name"], type=road_type, path=path
            )

    # 2. Identify High Points
    high_points = []
    for level_value in np.linspace(elevation_map.min(), elevation_map.max(), num=10):
        import matplotlib.pyplot as plt

        contour_data = plt.contourf(
            elevation_map, levels=[level_value - 0.01, level_value + 0.01]
        )
        enclosed_points = find_enclosed_points(contour_data, level_value, elevation_map)
        high_points.extend(enclosed_points)
        plt.clf()
        plt.close()

    # 3. Add Additional Connections (Avoiding High Points)
    settlement_positions = np.array([(s["x"], s["y"]) for s in settlements])
    kdtree = SKLearnKDTree(settlement_positions)

    for i, settlement1 in enumerate(settlements):
        neighbor_indices = kdtree.query_radius(
            [(settlement1["x"], settlement1["y"])], r=40
        )[0]
        for j in neighbor_indices:
            settlement2 = settlements[j]
            if i != j and not graph.has_edge(settlement1["name"], settlement2["name"]):
                distance = (
                    (settlement1["x"] - settlement2["x"]) ** 2
                    + (settlement1["y"] - settlement2["y"]) ** 2
                ) ** 0.5

                connection_probability = (
                    settlement1["connectivity"] * settlement2["connectivity"]
                ) / (distance * 5)

                if random.random() < connection_probability:
                    path = a_star_search(
                        level,
                        (settlement1["x"], settlement1["y"]),
                        (settlement2["x"], settlement2["y"]),
                        elevation_map,
                        high_points,
                    )
                    if path is not None:
                        road_type = "land"
                        for px, py in path:
                            if level[py][px] == "W":
                                road_type = "water"
                                break
                        graph.add_edge(
                            settlement1["name"],
                            settlement2["name"],
                            type=road_type,
                            path=path,
                        )

    return graph
