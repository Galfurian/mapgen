"""Terrain generation module for procedural maps."""

import random

import noise
import numpy as np

from .map import Map, Tile


def initialize_level(width: int, height: int, tiles: dict[str, Tile]) -> Map:
    """Initialize a map grid filled with walls.

    Args:
        width (int): The width of the map.
        height (int): The height of the map.
        tiles (dict[str, Tile]): The tile catalog.

    Returns:
        Map: A Map instance representing the map grid filled with walls.

    """
    return Map([[tiles["wall"] for _ in range(width)] for _ in range(height)])


def initialize_character(
    width: int, height: int, padding: int, wall_countdown: int
) -> dict:
    """Initialize a digging character.

    Args:
        width (int): The width of the map.
        height (int): The height of the map.
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


def dig(map: Map, character: dict, tiles: dict[str, Tile]) -> None:
    """Simulate character digging through the map.

    Args:
        map (Map): The map grid to modify.
        character (dict): The character state dictionary.
        tiles (dict[str, Tile]): The tile catalog.

    """
    while character["wallCountdown"] > 0:
        x = character["x"]
        y = character["y"]

        current_tile = map.get_terrain(x, y)
        if not current_tile.walkable:
            map.set_terrain(x, y, tiles["floor"])
            character["wallCountdown"] -= 1

        traverse = random.randint(1, 4)

        if traverse == 1 and x > character["padding"]:
            character["x"] -= 1
        elif traverse == 2 and x < map.width - 1 - character["padding"]:
            character["x"] += 1
        elif traverse == 3 and y > character["padding"]:
            character["y"] -= 1
        elif traverse == 4 and y < map.height - 1 - character["padding"]:
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
    map: Map,
    noise_map: np.ndarray,
    tiles: dict[str, Tile],
    sea_level: float = 0.03,
    mountain_level: float = 0.5,
    forest_threshold: float = 0.1,
) -> tuple[Map, np.ndarray]:
    """Apply terrain features based on noise map.

    Args:
        map (Map): The map grid to modify.
        noise_map (np.ndarray): The noise map.
        tiles (dict[str, Tile]): The tile catalog.
        sea_level (float): The threshold for sea map.
        mountain_level (float): The threshold for mountain map.
        forest_threshold (float): The threshold for forest.

    Returns:
        Tuple[Map, np.ndarray]: The modified map and elevation map.

    """
    height, width = noise_map.shape
    elevation_map = noise_map.copy()

    for y in range(height):
        for x in range(width):
            noise_value = noise_map[y, x]

            if noise_value < sea_level:
                map.set_terrain(x, y, tiles["water"])  # Water
            elif noise_value < mountain_level:
                if noise_value > forest_threshold:
                    map.set_terrain(x, y, tiles["forest"])  # Forest
                else:
                    map.set_terrain(x, y, tiles["plains"])  # Plains
            else:
                map.set_terrain(x, y, tiles["mountain"])  # Mountain

    return map, elevation_map


def smooth_terrain(map: Map, tiles: dict[str, Tile], iterations: int = 5) -> Map:
    """Smooth the terrain using cellular automata rules.

    Args:
        map (Map): The map grid to smooth.
        tiles (dict[str, Tile]): The tile catalog.
        iterations (int): The number of smoothing iterations.

    Returns:
        Map: The smoothed map grid.

    """
    height = map.height
    width = map.width

    for _ in range(iterations):
        new_grid = [row[:] for row in map.grid]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_tile = map.get_terrain(x, y)
                
                # Skip obstacles (non-walkable tiles)
                if not current_tile.walkable:
                    continue

                # Get neighbor properties
                neighbors = [
                    map.get_terrain(x - 1, y - 1),
                    map.get_terrain(x, y - 1),
                    map.get_terrain(x + 1, y - 1),
                    map.get_terrain(x - 1, y),
                    map.get_terrain(x + 1, y),
                    map.get_terrain(x - 1, y + 1),
                    map.get_terrain(x, y + 1),
                    map.get_terrain(x + 1, y + 1),
                ]

                # Count different neighbor types based on properties
                obstacle_count = sum(1 for n in neighbors if not n.walkable)
                liquid_count = sum(1 for n in neighbors if n.movement_cost >= 2.0 and n.habitability == 0.0)
                difficult_count = sum(1 for n in neighbors if n.movement_cost > 1.0 and n.movement_cost < 2.0)
                elevated_count = sum(1 for n in neighbors if n.elevation_influence > 0.5)

                # Apply smoothing rules based on neighbor majority
                if obstacle_count > 4:
                    new_grid[y][x] = tiles["wall"]
                elif liquid_count > 3:
                    new_grid[y][x] = tiles["water"]
                elif elevated_count > 3:
                    new_grid[y][x] = tiles["mountain"]
                elif difficult_count > 4:
                    new_grid[y][x] = tiles["forest"]
        map.grid = new_grid

    return map
