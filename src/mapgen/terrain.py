"""Terrain generation module for procedural maps."""

import random

import noise
import numpy as np

from . import logger
from .map_data import MapData, Tile


def initialize_level(
    width: int,
    height: int,
    tiles: dict[str, Tile],
) -> MapData:
    """Initialize a map grid filled with walls.

    Args:
        width (int): The width of the map.
        height (int): The height of the map.
        tiles (dict[str, Tile]): The tile catalog.

    Returns:
        MapData: A MapData instance representing the map grid filled with walls.

    """
    return MapData([[tiles["wall"] for _ in range(width)] for _ in range(height)])


def initialize_character(
    width: int,
    height: int,
    padding: int,
    wall_countdown: int,
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


def dig(
    map_data: MapData,
    character: dict,
    tiles: dict[str, Tile],
) -> None:
    """Simulate character digging through the map.

    Args:
        map_data (MapData): The map grid to modify.
        character (dict): The character state dictionary.
        tiles (dict[str, Tile]): The tile catalog.

    """
    initial_countdown = character["wallCountdown"]
    logger.debug(f"Starting terrain digging: {initial_countdown} walls to dig")
    
    while character["wallCountdown"] > 0:
        x = character["x"]
        y = character["y"]

        current_tile = map_data.get_terrain(x, y)
        if not current_tile.walkable:
            map_data.set_terrain(x, y, tiles["floor"])
            character["wallCountdown"] -= 1
            
            # Log progress every 10% completion
            remaining = character["wallCountdown"]
            progress = (initial_countdown - remaining) / initial_countdown
            if progress % 0.1 < 0.01:  # Log roughly every 10%
                logger.debug(f"Digging progress: {progress:.1%} complete ({remaining} walls remaining)")

        traverse = random.randint(1, 4)

        if traverse == 1 and x > character["padding"]:
            character["x"] -= 1
        elif traverse == 2 and x < map_data.width - 1 - character["padding"]:
            character["x"] += 1
        elif traverse == 3 and y > character["padding"]:
            character["y"] -= 1
        elif traverse == 4 and y < map_data.height - 1 - character["padding"]:
            character["y"] += 1
    
    logger.debug("Terrain digging completed")


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
    map_data: MapData,
    noise_map: np.ndarray,
    tiles: dict[str, Tile],
    sea_level: float = 0.03,
    mountain_level: float = 0.5,
    forest_threshold: float = 0.1,
) -> tuple[MapData, np.ndarray]:
    """Apply terrain features based on noise map.

    Args:
        map_data (MapData): The map grid to modify.
        noise_map (np.ndarray): The noise map.
        tiles (dict[str, Tile]): The tile catalog.
        sea_level (float): The threshold for sea map.
        mountain_level (float): The threshold for mountain map.
        forest_threshold (float): The threshold for forest.

    Returns:
        Tuple[MapData, np.ndarray]: The modified map and elevation map.

    """
    height, width = noise_map.shape
    elevation_map = noise_map.copy()

    for y in range(height):
        for x in range(width):
            noise_value = noise_map[y, x]

            if noise_value < sea_level:
                map_data.set_terrain(x, y, tiles["water"])  # Water
            elif noise_value < mountain_level:
                if noise_value > forest_threshold:
                    map_data.set_terrain(x, y, tiles["forest"])  # Forest
                else:
                    map_data.set_terrain(x, y, tiles["plains"])  # Plains
            else:
                map_data.set_terrain(x, y, tiles["mountain"])  # Mountain

    return map_data, elevation_map


def smooth_terrain(
    map_data: MapData,
    tiles: dict[str, Tile],
    iterations: int = 5,
) -> MapData:
    """Smooth the terrain using cellular automata rules.

    Args:
        map_data (MapData): The map grid to smooth.
        tiles (dict[str, Tile]): The tile catalog.
        iterations (int): The number of smoothing iterations.

    Returns:
        MapData: The smoothed map grid.

    """
    height = map_data.height
    width = map_data.width

    for _ in range(iterations):
        new_grid = [row[:] for row in map_data.grid]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_tile = map_data.get_terrain(x, y)

                # Skip obstacles (non-walkable tiles)
                if not current_tile.walkable:
                    continue

                # Get neighbor properties
                neighbors = map_data.get_neighbor_tiles(
                    x, y, walkable_only=False, include_diagonals=True
                )

                # Count different neighbor types based on properties
                obstacle_count = sum(1 for n in neighbors if not n.walkable)
                liquid_count = sum(1 for n in neighbors if not n.can_build_road)
                difficult_count = sum(
                    1
                    for n in neighbors
                    if n.movement_cost > 1.0 and n.movement_cost < 2.0
                )
                elevated_count = sum(
                    1 for n in neighbors if n.elevation_influence > 0.5
                )

                # Apply smoothing rules based on neighbor majority
                if obstacle_count > 4:
                    new_grid[y][x] = tiles["wall"]
                elif liquid_count > 3:
                    new_grid[y][x] = tiles["water"]
                elif elevated_count > 3:
                    new_grid[y][x] = tiles["mountain"]
                elif difficult_count > 4:
                    new_grid[y][x] = tiles["forest"]
        map_data.grid = new_grid

    return map_data
