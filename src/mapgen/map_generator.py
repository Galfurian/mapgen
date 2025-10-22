"""Map generation functions."""

import logging
import random
import time
from typing import Optional

import numpy as np

from . import (
    MapData,
    Tile,
    PlacementMethod,
    hydrology,
    roads,
    rivers,
    settlements,
    terrain,
)

logger = logging.getLogger(__name__)


def generate_map(
    width: int = 150,
    height: int = 100,
    padding: int = 2,
    scale: float = 100.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    smoothing_iterations: int = 5,
    settlement_density: float = 0.002,
    min_settlement_radius: float = 0.5,
    max_settlement_radius: float = 1.0,
    seed: Optional[int] = None,
    enable_rainfall: bool = True,
    enable_smoothing: bool = True,
    enable_settlements: bool = True,
    enable_roads: bool = True,
    enable_rivers: bool = True,
    min_river_length: int = 15,
    max_rivers: int = 3,
    rainfall_threshold: float = 0.8,
    elevation_threshold: float = 0.6,
    sea_level: float = 0.0,
) -> MapData:
    """
    Generate a complete procedural fantasy map.

    This function orchestrates all map generation steps in the correct sequence:
    terrain generation, hydrological features, settlements, and roads.

    Args:
        width (int): The width of the map.
        height (int): The height of the map.
        padding (int): Padding around edges.
        scale (float): Noise scale for terrain generation.
        octaves (int): Number of noise octaves.
        persistence (float): Noise persistence.
        lacunarity (float): Noise lacunarity.
        smoothing_iterations (int): Number of terrain smoothing iterations.
        settlement_density (float): Density of settlements (0.0 to 1.0).
        min_settlement_radius (float): Minimum settlement radius.
        max_settlement_radius (float): Maximum settlement radius.
        seed (int): Random seed for reproducible generation. If None, uses random seed.
        enable_rainfall (bool): Whether to generate rainfall data.
        enable_smoothing (bool): Whether to apply terrain smoothing.
        enable_settlements (bool): Whether to generate settlements.
        enable_roads (bool): Whether to generate road networks.
        enable_rivers (bool): Whether to generate rivers.
        min_river_length (int): Minimum length for rivers.
        max_rivers (int): Maximum number of rivers to generate.
        rainfall_threshold (float): Minimum rainfall for river sources.
        elevation_threshold (float): Minimum elevation for river sources.
        sea_level (float): Elevation level for sea (controls land/sea ratio, -1.0 to 1.0).

    Returns:
        MapData: The generated map data.

    Raises:
        ValueError: If any parameter has an invalid value.
    """
    # Validate input parameters
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"Height must be positive, got {height}")
    if padding < 0:
        raise ValueError(f"Padding must be non-negative, got {padding}")
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
    if octaves < 1:
        raise ValueError(f"Octaves must be at least 1, got {octaves}")
    if not (0.0 < persistence < 1.0):
        raise ValueError(f"Persistence must be between 0 and 1, got {persistence}")
    if lacunarity <= 1.0:
        raise ValueError(f"Lacunarity must be greater than 1, got {lacunarity}")
    if smoothing_iterations < 0:
        raise ValueError(
            f"Smoothing iterations must be non-negative, got {smoothing_iterations}"
        )
    if not (0.0 <= settlement_density <= 1.0):
        raise ValueError(
            f"Settlement density must be between 0 and 1, got {settlement_density}"
        )
    if min_settlement_radius <= 0:
        raise ValueError(
            f"Minimum settlement radius must be positive, got {min_settlement_radius}"
        )
    if max_settlement_radius <= 0:
        raise ValueError(
            f"Maximum settlement radius must be positive, got {max_settlement_radius}"
        )
    if min_settlement_radius > max_settlement_radius:
        raise ValueError(
            f"Minimum settlement radius ({min_settlement_radius}) cannot be greater than maximum ({max_settlement_radius})"
        )
    if not (-1.0 <= sea_level <= 1.0):
        raise ValueError(f"Sea level must be between -1 and 1, got {sea_level}")

    start_time = time.time()
    logger.info(f"Starting map generation: {width}*{height}")

    # Set random seed for reproducible generation
    if seed is None:
        seed = random.randint(0, 1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Using random seed: {seed}")

    tiles = get_default_tiles()

    logger.debug("Initializing map data")
    map_data = MapData(
        tiles=tiles,
        grid=[[0 for _ in range(width)] for _ in range(height)],
    )

    logger.debug("Generating noise map")
    terrain_start = time.time()
    terrain.generate_noise_map(
        map_data,
        width,
        height,
        scale,
        octaves,
        persistence,
        lacunarity,
        sea_level,
    )
    terrain_time = time.time() - terrain_start
    logger.debug(f"Noise map generation completed in {terrain_time:.3f}s")

    if enable_rainfall:
        logger.debug("Generating rainfall map")
        rainfall_start = time.time()
        hydrology.generate_rainfall_map(
            map_data,
            width,
            height,
        )
        rainfall_time = time.time() - rainfall_start
        logger.debug(f"Rainfall map generation completed in {rainfall_time:.3f}s")

    # Hydrology: compute accumulation (runoff) map
    logger.debug("Computing water accumulation (runoff) map")
    elevation = np.array(map_data.elevation_map)
    rainfall = np.array(map_data.rainfall_map)
    accumulation = hydrology.compute_accumulation(elevation, rainfall)
    map_data.accumulation_map = accumulation.tolist()
    logger.debug(
        f"Accumulation stats: min={accumulation.min():.3f}, max={accumulation.max():.3f}, mean={accumulation.mean():.3f}"
    )

    logger.debug("Applying terrain features")
    features_start = time.time()
    terrain.apply_terrain_features(map_data)
    features_time = time.time() - features_start
    logger.debug(f"Terrain features applied in {features_time:.3f}s")

    if enable_rivers:
        logger.debug("Generating rivers")
        rivers_start = time.time()
        rivers.generate_rivers(
            map_data,
            min_river_length=min_river_length,
            max_rivers=max_rivers,
            rainfall_threshold=rainfall_threshold,
            elevation_threshold=elevation_threshold,
        )
        rivers_time = time.time() - rivers_start
        logger.debug(f"River generation completed in {rivers_time:.3f}s")

    if enable_smoothing:
        logger.debug("Smoothing terrain")
        smooth_start = time.time()
        terrain.smooth_terrain(
            map_data,
            smoothing_iterations,
        )
        smooth_time = time.time() - smooth_start
        logger.debug(f"Terrain smoothing completed in {smooth_time:.3f}s")

    if enable_rivers and not enable_smoothing:
        # If rivers were enabled but smoothing was already done above, no need to smooth again
        pass
    elif enable_rivers:
        logger.debug("Smoothing terrain after rivers")
        smooth_start = time.time()
        terrain.smooth_terrain(
            map_data,
            smoothing_iterations,
        )
        smooth_time = time.time() - smooth_start
        logger.debug(f"Terrain smoothing completed in {smooth_time:.3f}s")

    if enable_settlements:
        logger.debug("Generating settlements")
        settlements_start = time.time()
        settlements.generate_settlements(
            map_data,
            settlement_density,
            min_settlement_radius,
            max_settlement_radius,
        )
        settlements_time = time.time() - settlements_start
        logger.debug(f"Settlements generated in {settlements_time:.3f}s")
        logger.debug(f"Generated {len(map_data.settlements)} settlements")

    if enable_roads:
        logger.debug("Generating road network")
        roads_start = time.time()
        roads.generate_roads(map_data)
        roads_time = time.time() - roads_start
        logger.debug(f"Road network generated in {roads_time:.3f}s")
        logger.debug(f"Generated road network with {len(map_data.roads)} roads")

    total_time = time.time() - start_time
    logger.info(f"Map generation completed successfully in {total_time:.3f}s")

    return map_data


def get_default_tiles() -> list[Tile]:
    """
    Create the catalog of tiles used for map generation.

    Returns:
        list[Tile]: List of Tile instances used in the map.
    """
    tiles = [
        Tile(
            name="sea",
            description="Sea water terrain",
            walkable=True,
            movement_cost=2.0,
            blocks_line_of_sight=False,
            buildable=False,
            habitability=0.0,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_influence=-0.5,
            smoothing_weight=1.0,
            elevation_min=-1.0,
            elevation_max=-0.5,
            terrain_priority=1,
            smoothing_priority=2,
            symbol="~",
            color=(0.2, 0.5, 1.0),
            resources=[],
            is_water=True,
            is_salt_water=True,
            is_flowing_water=False,
        ),
        Tile(
            name="plains",
            description="Open plains",
            walkable=True,
            movement_cost=1.0,
            blocks_line_of_sight=False,
            buildable=True,
            habitability=0.9,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_influence=0.0,
            smoothing_weight=1.0,
            elevation_min=0.0,
            elevation_max=0.2,
            terrain_priority=2,
            smoothing_priority=0,
            symbol=".",
            color=(0.8, 0.9, 0.6),
            resources=["grain", "herbs"],
        ),
        Tile(
            name="forest",
            description="Forest terrain",
            walkable=True,
            movement_cost=1.2,
            blocks_line_of_sight=False,
            buildable=True,
            habitability=0.7,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_influence=0.0,
            smoothing_weight=1.0,
            elevation_min=0.2,
            elevation_max=0.5,
            terrain_priority=3,
            smoothing_priority=4,
            symbol="F",
            color=(0.2, 0.6, 0.2),
            resources=["wood", "game"],
        ),
        Tile(
            name="mountain",
            description="Mountain terrain",
            walkable=False,
            movement_cost=1.0,
            blocks_line_of_sight=True,
            buildable=False,
            habitability=0.1,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_influence=1.0,
            smoothing_weight=1.0,
            elevation_min=0.5,
            elevation_max=1.0,
            terrain_priority=4,
            smoothing_priority=3,
            symbol="^",
            color=(0.5, 0.4, 0.3),
            resources=["stone", "ore"],
        ),
    ]
    tiles.append(
        Tile(
            name="river",
            description="Flowing river water",
            walkable=True,
            movement_cost=1.5,
            blocks_line_of_sight=False,
            buildable=False,
            habitability=0.0,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_influence=-0.3,
            smoothing_weight=1.0,
            elevation_min=-0.5,
            elevation_max=1.0,
            terrain_priority=1,
            smoothing_priority=1,
            symbol="R",
            color=(0.1, 0.4, 0.9),
            resources=["fish"],
            is_water=True,
            is_salt_water=False,
            is_flowing_water=True,
            placement_method=PlacementMethod.ALGORITHM_BASED,
        )
    )
    return tiles
