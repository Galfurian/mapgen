"""
General utility functions for procedural map generation.

This module provides helper functions for generating procedural maps, including
noise generation utilities and geometric calculations.
"""

import random

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


def compute_terrain_control_point(
    start: Position, 
    end: Position, 
    map_data: MapData,
    control_factor: float = 2.0,
    invert_gradients: bool = False
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
