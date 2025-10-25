"""
General utility functions for procedural map generation.

This module provides helper functions for generating procedural maps, including
noise generation utilities.
"""

import random

import noise
import numpy as np


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
