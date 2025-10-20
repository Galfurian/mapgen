"""Terrain generation module for procedural maps."""

import random

import noise
import numpy as np

from . import logger
from .map_data import MapData


def dig(
    map_data: MapData,
    padding: int,
    initial_x: int,
    initial_y: int,
) -> None:
    """
    Simulate character digging through the map.

    Args:
        map_data (MapData):
            The map grid to modify.
        padding (int):
            The padding around the edges.
        initial_x (int):
            The starting x-coordinate of the character.
        initial_y (int):
            The starting y-coordinate of the character.

    """
    max_countdown = max(100, (map_data.width * map_data.height) // 3)

    logger.debug(f"Starting terrain digging: {max_countdown} walls to dig")

    countdown = 0
    x = initial_x
    y = initial_y
    floor = map_data.tiles.index(next(t for t in map_data.tiles if t.name == "floor"))

    while countdown < max_countdown:
        current_tile = map_data.get_terrain(x, y)
        if not current_tile.walkable:
            map_data.set_terrain(x, y, floor)
            countdown += 1

            # Log progress every 10% completion
            progress = countdown / max_countdown
            remaining = max_countdown - countdown
            if progress % 0.1 < 0.01:
                logger.debug(
                    f"Digging progress: {progress:.1%} complete ({remaining} walls remaining)"
                )

        traverse = random.randint(1, 4)

        if traverse == 1 and x > padding:
            x -= 1
        elif traverse == 2 and x < map_data.width - 1 - padding:
            x += 1
        elif traverse == 3 and y > padding:
            y -= 1
        elif traverse == 4 and y < map_data.height - 1 - padding:
            y += 1

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
    sea_level: float = 0.03,
    mountain_level: float = 0.5,
    forest_threshold: float = 0.1,
) -> None:
    """
    Apply terrain features based on noise map.

    Args:
        map_data (MapData):
            The map grid to modify.
        noise_map (np.ndarray):
            The noise map.
        sea_level (float):
            The threshold for sea map.
        mountain_level (float):
            The threshold for mountain map.
        forest_threshold (float):
            The threshold for forest.

    """
    water = map_data.tiles.index(next(t for t in map_data.tiles if t.name == "water"))
    plains = map_data.tiles.index(next(t for t in map_data.tiles if t.name == "plains"))
    forest = map_data.tiles.index(next(t for t in map_data.tiles if t.name == "forest"))
    mountain = map_data.tiles.index(
        next(t for t in map_data.tiles if t.name == "mountain")
    )

    for y in range(map_data.height):
        for x in range(map_data.width):
            noise_value = noise_map[y, x]
            if noise_value < sea_level:
                map_data.set_terrain(x, y, water)
            elif noise_value < mountain_level:
                if noise_value > forest_threshold:
                    map_data.set_terrain(x, y, forest)
                else:
                    map_data.set_terrain(x, y, plains)
            else:
                map_data.set_terrain(x, y, mountain)


def smooth_terrain(
    map_data: MapData,
    iterations: int = 5,
) -> None:
    """
    Smooth the terrain using cellular automata rules.

    Args:
        map_data (MapData):
            The map grid to smooth.
        iterations (int):
            The number of smoothing iterations.

    """
    wall = map_data.tiles.index(next(t for t in map_data.tiles if t.name == "wall"))
    water = map_data.tiles.index(next(t for t in map_data.tiles if t.name == "water"))
    forest = map_data.tiles.index(next(t for t in map_data.tiles if t.name == "forest"))
    mountain = map_data.tiles.index(
        next(t for t in map_data.tiles if t.name == "mountain")
    )

    for _ in range(iterations):
        new_grid_indices = [row[:] for row in map_data.grid]
        for y in range(1, map_data.height - 1):
            for x in range(1, map_data.width - 1):
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
                    new_grid_indices[y][x] = wall
                elif liquid_count > 3:
                    new_grid_indices[y][x] = water
                elif elevated_count > 3:
                    new_grid_indices[y][x] = mountain
                elif difficult_count > 4:
                    new_grid_indices[y][x] = forest
        map_data.grid = new_grid_indices
