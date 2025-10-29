"""
General utility functions for procedural map generation.

This module provides helper functions for generating procedural maps, including
noise generation utilities and geometric calculations.
"""

import random
from typing import Callable

import noise
import numpy as np

from .map_data import MapData, Position


def generate_noise_grid(
    width: int,
    height: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    base: int,
) -> np.ndarray:
    """
    Generate a 2D Perlin noise grid.

    Args:
        width (int):
            Width of the grid.
        height (int):
            Height of the grid.
        scale (float):
            Scale factor for noise coordinates.
        octaves (int):
            Number of noise octaves.
        persistence (float):
            Persistence value for octaves.
        lacunarity (float):
            Lacunarity value for octaves.
        base (int):
            Base seed for noise generation.

    Returns:
        np.ndarray:
            2D array of noise values.

    """
    # Generate random offsets to ensure unique noise patterns for each
    # generation.
    offset_x = random.uniform(0, 10000)
    offset_y = random.uniform(0, 10000)

    # Create an empty grid to store noise values.
    grid = np.zeros((height, width))

    # Iterate over each cell in the grid and compute Perlin noise.
    for y in range(height):
        for x in range(width):
            grid[y, x] = noise.pnoise2(
                (x / scale) + offset_x,
                (y / scale) + offset_y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=base,
            )
    return grid


def quadratic_bezier_points(
    p0: Position, p1: Position, p2: Position, num_points: int
) -> list[Position]:
    """
    Generate points along a quadratic Bezier curve.

    Args:
        p0 (Position): Start point.
        p1 (Position): Control point.
        p2 (Position): End point.
        num_points (int): Number of points to generate.

    Returns:
        list[Position]: Points on the curve.
    """
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = (1 - t) ** 2 * p0.x + 2 * (1 - t) * t * p1.x + t**2 * p2.x
        y = (1 - t) ** 2 * p0.y + 2 * (1 - t) * t * p1.y + t**2 * p2.y
        points.append(Position(round(x), round(y)))
    return points


def calculate_bezier_points_count(
    start: Position,
    end: Position,
    min_points: int = 8,
    max_points: int = 30,
    points_per_unit: float = 0.5,
) -> int:
    """
    Calculate the appropriate number of Bezier curve points based on distance.

    This ensures smoother curves for longer distances while avoiding excessive
    computation for short curves.

    Args:
        start (Position): Start position.
        end (Position): End position.
        min_points (int): Minimum number of points (default: 8).
        max_points (int): Maximum number of points (default: 30).
        points_per_unit (float): Points to add per unit distance (default: 0.5).

    Returns:
        int: Number of points to use for the Bezier curve.
    """
    distance = start.distance_to(end)
    num_points = int(min_points + distance * points_per_unit)
    return max(min_points, min(max_points, num_points))


def curve_path(
    path: list[Position],
    map_data: MapData,
    position_validator: Callable[[MapData, Position], bool],
    control_factor: float = 2.0,
    invert_gradients: bool = False,
) -> list[Position]:
    """
    Aggressively simplify the path by trying to interpolate directly to the
    farthest valid end point, falling back to closer points if invalid. Uses
    Bezier curves for smooth bends. Tries both forward and backward directions
    and chooses the one with fewer points.

    Args:
        path (list[Position]):
            The path to simplify and curve.
        map_data (MapData):
            The map data containing elevation and terrain.
        position_validator (Callable[[MapData, Position], bool]):
            Function to validate if a position is valid for path placement.
        control_factor (float):
            Multiplier for gradient influence in control point calculation.
        invert_gradients (bool):
            If True, invert gradient direction for control point calculation.

    Returns:
        list[Position]:
            The simplified, curved path.
    """
    if len(path) <= 2:
        return path

    # Try curving in forward direction
    forward_path = curve_path_direction(
        path,
        map_data,
        position_validator,
        control_factor,
        invert_gradients,
        reverse=False,
    )

    # Try curving in backward direction
    backward_path = curve_path_direction(
        path,
        map_data,
        position_validator,
        control_factor,
        invert_gradients,
        reverse=True,
    )

    # Return the one with fewer points
    return forward_path if len(forward_path) <= len(backward_path) else backward_path


def curve_path_direction(
    path: list[Position],
    map_data: MapData,
    position_validator: Callable[[MapData, Position], bool],
    control_factor: float = 2.0,
    invert_gradients: bool = False,
    reverse: bool = False,
) -> list[Position]:
    """
    Aggressively simplify the path in one direction by trying to interpolate
    directly to the farthest valid end point, falling back to closer points if invalid.

    Args:
        path (list[Position]):
            The path to simplify and curve.
        map_data (MapData):
            The map data containing elevation and terrain.
        position_validator (Callable[[MapData, Position], bool]):
            Function to validate if a position is valid for path placement.
        control_factor (float):
            Multiplier for gradient influence in control point calculation.
        invert_gradients (bool):
            If True, invert gradient direction for control point calculation.
        reverse (bool):
            Whether to process the path in reverse order.

    Returns:
        list[Position]:
            The simplified, curved path.
    """
    if len(path) <= 2:
        return path

    # Work on a copy and potentially reverse it
    work_path = path[::-1] if reverse else path[:]

    result = [work_path[0]]
    current_idx = 0

    while current_idx < len(work_path) - 1:
        start = work_path[current_idx]
        found = False

        # Try farthest end first
        for end_idx in range(len(work_path) - 1, current_idx, -1):
            end = work_path[end_idx]

            # Compute control point
            control = compute_terrain_control_point(
                start,
                end,
                map_data,
                control_factor=control_factor,
                invert_gradients=invert_gradients,
            )

            # Generate Bezier points
            num_points = calculate_bezier_points_count(start, end)
            bezier_points = quadratic_bezier_points(
                start, control, end, num_points=num_points
            )

            # Check validity
            if all(position_validator(map_data, pos) for pos in bezier_points):
                # Valid: add the curve points (skip start)
                result.extend(bezier_points[1:])
                current_idx = end_idx
                found = True
                break

        # No valid jump: add next point
        if not found:
            current_idx += 1
            if current_idx < len(work_path):
                result.append(work_path[current_idx])

    # If we reversed, we need to reverse the result back
    return result[::-1] if reverse else result


def compute_terrain_control_point(
    start: Position,
    end: Position,
    map_data: MapData,
    control_factor: float = 2.0,
    invert_gradients: bool = False,
) -> Position:
    """
    Compute a control point for Bezier curves based on terrain gradients.

    This function analyzes the terrain (elevation/depth) gradients at the midpoint
    between start and end positions to create natural-looking curves that follow
    the landscape contours.

    Args:
        start (Position): Start position.
        end (Position): End position.
        map_data (MapData): Map data for terrain analysis.
        control_factor (float): Multiplier for gradient influence (default: 2.0).
        invert_gradients (bool): If True, invert gradient direction (useful for water routes).

    Returns:
        Position: Control point for the curve.
    """
    mid_x = (start.x + end.x) / 2
    mid_y = (start.y + end.y) / 2
    elevation_map = map_data.elevation_map
    grad_x = 0.0
    grad_y = 0.0

    if 0 <= int(mid_x) < len(elevation_map[0]) and 0 <= int(mid_y) < len(elevation_map):
        current_elev = elevation_map[int(mid_y)][int(mid_x)]
        left_elev = (
            elevation_map[int(mid_y)][max(0, int(mid_x) - 1)]
            if int(mid_x) > 0
            else current_elev
        )
        right_elev = (
            elevation_map[int(mid_y)][min(len(elevation_map[0]) - 1, int(mid_x) + 1)]
            if int(mid_x) < len(elevation_map[0]) - 1
            else current_elev
        )
        top_elev = (
            elevation_map[max(0, int(mid_y) - 1)][int(mid_x)]
            if int(mid_y) > 0
            else current_elev
        )
        bottom_elev = (
            elevation_map[min(len(elevation_map) - 1, int(mid_y) + 1)][int(mid_x)]
            if int(mid_y) < len(elevation_map) - 1
            else current_elev
        )
        grad_x = (right_elev - left_elev) / 2
        grad_y = (bottom_elev - top_elev) / 2

        if invert_gradients:
            grad_x = -grad_x
            grad_y = -grad_y

    control_x = mid_x + control_factor * grad_x
    control_y = mid_y + control_factor * grad_y
    return Position(round(control_x), round(control_y))


def a_star_search(
    map_data: MapData,
    start: Position,
    goal: Position,
    position_validator: Callable[[MapData, Position], bool],
) -> list[Position] | None:
    """
    Perform A* search with configurable validation.

    Args:
        map_data (MapData): The map grid.
        start (Position): The start position.
        goal (Position): The goal position.
        position_validator (callable): Function to validate if a position is traversable.

    Returns:
        list[Position] | None: The path if found, None otherwise.
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

            if not position_validator(map_data, neighbor):
                continue

            tile = map_data.get_terrain(neighbor.x, neighbor.y)
            tentative_cost = current_cost + tile.movement_cost

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


def _reconstruct_path(
    current: Position,
    came_from: dict[Position, Position],
) -> list[Position]:
    """
    Reconstruct the path from A* search.

    Args:
        current (Position): The current position.
        came_from (dict[Position, Position]): The came_from dictionary.

    Returns:
        list[Position]: The reconstructed path.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def normalize_array(
    arr: np.ndarray,
    min_val: float | None = None,
    max_val: float | None = None,
    clip: bool = True,
    decimals: int | None = 4,
) -> np.ndarray:
    """
    Normalize a numpy array to the [0, 1] range using min-max scaling.

    Args:
        arr (np.ndarray):
            The input array to normalize.
        min_val (float | None):
            The minimum value to use for normalization. If None, uses np.min(arr).
        max_val (float | None):
            The maximum value to use for normalization. If None, uses np.max(arr).
        clip (bool):
            Whether to clip values to [0, 1] after normalization. Default is True.
        decimals (int | None):
            Number of decimal places to round to. If None, no rounding is applied.
            Default is 4.

    Returns:
        np.ndarray:
            The normalized array.
    """
    if min_val is None:
        min_val = float(np.min(arr))
    if max_val is None:
        max_val = float(np.max(arr))
    if max_val > min_val:
        normalized = (arr - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(arr)
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    if decimals is not None:
        normalized = np.round(normalized, decimals=decimals)
    return normalized
