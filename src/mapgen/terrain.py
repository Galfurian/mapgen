"""Terrain generation module for procedural maps."""

import random
from typing import List, Tuple

import noise
import numpy as np


def initialize_level(width: int, height: int) -> List[List[str]]:
    """Initialize a level grid filled with walls.

    Args:
        width (int): The width of the level.
        height (int): The height of the level.

    Returns:
        List[List[str]]: A 2D list representing the level grid filled with walls.
    """
    return [["#"] * width for _ in range(height)]


def initialize_character(
    width: int, height: int, padding: int, wall_countdown: int
) -> dict:
    """Initialize a digging character.

    Args:
        width (int): The width of the level.
        height (int): The height of the level.
        padding (int): The padding around the edges.
        wall_countdown (int): The number of walls to dig.

    Returns:
        dict: A dictionary representing the character state.
    """
    return {
        "wallCountdown": wall_countdown,
        "padding": padding,
        "x": width // 2,
        "y": height // 2,
    }


def dig(level: List[List[str]], character: dict) -> None:
    """Simulate character digging through the level.

    Args:
        level (List[List[str]]): The level grid to modify.
        character (dict): The character state dictionary.
    """
    while character["wallCountdown"] > 0:
        x = character["x"]
        y = character["y"]

        if level[y][x] == "#":
            level[y][x] = " "
            character["wallCountdown"] -= 1

        traverse = random.randint(1, 4)

        if traverse == 1 and x > character["padding"]:
            character["x"] -= 1
        elif traverse == 2 and x < len(level[0]) - 1 - character["padding"]:
            character["x"] += 1
        elif traverse == 3 and y > character["padding"]:
            character["y"] -= 1
        elif traverse == 4 and y < len(level) - 1 - character["padding"]:
            character["y"] += 1


def generate_noise_map(
    width: int,
    height: int,
    scale: float = 50.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Generate a Perlin noise map.

    Args:
        width (int): The width of the noise map.
        height (int): The height of the noise map.
        scale (float): The scale of the noise.
        octaves (int): The number of octaves.
        persistence (float): The persistence value.
        lacunarity (float): The lacunarity value.

    Returns:
        np.ndarray: The generated noise map.
    """
    offset_x = random.uniform(0, 10000)
    offset_y = random.uniform(0, 10000)

    noise_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            noise_map[y, x] = noise.pnoise2(
                (x / scale) + offset_x,
                (y / scale) + offset_y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0,
            )
    return noise_map


def apply_terrain_features(
    level: List[List[str]],
    noise_map: np.ndarray,
    sea_level: float = 0.03,
    mountain_level: float = 0.5,
    forest_threshold: float = 0.1,
) -> Tuple[List[List[str]], np.ndarray]:
    """Apply terrain features based on noise map.

    Args:
        level (List[List[str]]): The level grid to modify.
        noise_map (np.ndarray): The noise map.
        sea_level (float): The threshold for sea level.
        mountain_level (float): The threshold for mountain level.
        forest_threshold (float): The threshold for forest.

    Returns:
        Tuple[List[List[str]], np.ndarray]: The modified level and elevation map.
    """
    height, width = noise_map.shape
    elevation_map = noise_map.copy()

    for y in range(height):
        for x in range(width):
            noise_value = noise_map[y, x]

            if noise_value < sea_level:
                level[y][x] = "W"  # Water
            elif noise_value < mountain_level:
                if noise_value > forest_threshold:
                    level[y][x] = "F"  # Forest
                else:
                    level[y][x] = "P"  # Plains
            else:
                level[y][x] = "M"  # Mountain

    return level, elevation_map


def smooth_terrain(level: List[List[str]], iterations: int = 5) -> List[List[str]]:
    """Smooth the terrain using cellular automata rules.

    Args:
        level (List[List[str]]): The level grid to smooth.
        iterations (int): The number of smoothing iterations.

    Returns:
        List[List[str]]: The smoothed level grid.
    """
    height = len(level)
    width = len(level[0])

    for _ in range(iterations):
        new_level = [row[:] for row in level]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if level[y][x] in ("#", " "):
                    continue

                neighbor_values = [
                    level[y - 1][x - 1],
                    level[y - 1][x],
                    level[y - 1][x + 1],
                    level[y][x - 1],
                    level[y][x + 1],
                    level[y + 1][x - 1],
                    level[y + 1][x],
                    level[y + 1][x + 1],
                ]

                if level[y][x] != "#" and level[y][x] != "W":
                    if neighbor_values.count("M") > 4:
                        new_level[y][x] = "M"
                    elif neighbor_values.count("F") > 5:
                        new_level[y][x] = "F"
                    elif neighbor_values.count("P") > 6:
                        new_level[y][x] = "P"
        level = new_level

    return level
