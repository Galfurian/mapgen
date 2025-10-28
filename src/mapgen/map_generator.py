"""
Map generation functions for procedural fantasy maps.

This module provides the main entry point for generating complete procedural
fantasy maps. It orchestrates all generation phases including terrain,
hydrology, vegetation, settlements, and infrastructure in the correct order.
"""

import logging
import random
import time

import numpy as np

from . import (
    MapData,
    flora,
    hydrology,
    rivers,
    roads,
    settlements,
    terrain,
    water_routes,
)
from .tile_collections import get_default_tile_collections

logger = logging.getLogger(__name__)


def _validate_map_parameters(
    width: int,
    height: int,
    padding: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    smoothing_iterations: int,
    settlement_density: float,
    min_settlement_radius: float,
    max_settlement_radius: float,
    sea_level: float,
) -> None:
    """Validate all map generation parameters."""
    validations = [
        (width <= 0, f"Width must be positive, got {width}"),
        (height <= 0, f"Height must be positive, got {height}"),
        (padding < 0, f"Padding must be non-negative, got {padding}"),
        (scale <= 0, f"Scale must be positive, got {scale}"),
        (octaves < 1, f"Octaves must be at least 1, got {octaves}"),
        (
            not (0.0 < persistence < 1.0),
            f"Persistence must be between 0 and 1, got {persistence}",
        ),
        (lacunarity <= 1.0, f"Lacunarity must be greater than 1, got {lacunarity}"),
        (
            smoothing_iterations < 0,
            f"Smoothing iterations must be non-negative, got {smoothing_iterations}",
        ),
        (
            not (0.0 <= settlement_density <= 1.0),
            f"Settlement density must be between 0 and 1, got {settlement_density}",
        ),
        (
            min_settlement_radius <= 0,
            f"Minimum settlement radius must be positive, got {min_settlement_radius}",
        ),
        (
            max_settlement_radius <= 0,
            f"Maximum settlement radius must be positive, got {max_settlement_radius}",
        ),
        (
            min_settlement_radius > max_settlement_radius,
            f"Minimum settlement radius ({min_settlement_radius}) cannot be greater than maximum ({max_settlement_radius})",
        ),
        (
            not (-1.0 <= sea_level <= 1.0),
            f"Sea level must be between -1 and 1, got {sea_level}",
        ),
    ]
    for condition, message in validations:
        if condition:
            raise ValueError(message)


def generate_map(
    width: int = 150,
    height: int = 100,
    padding: int = 2,
    scale: float = 100.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    smoothing_iterations: int = 5,
    smoothing_sigma: float = 0.5,
    settlement_density: float = 0.01,
    min_settlement_radius: float = 0.5,
    max_settlement_radius: float = 1.0,
    seed: int | None = None,
    enable_settlements: bool = True,
    enable_roads: bool = True,
    enable_rivers: bool = True,
    enable_vegetation: bool = True,
    min_source_elevation: float = 0.6,
    min_source_rainfall: float = 0.5,
    min_river_length: int = 10,
    sea_level: float = 0.0,
    rainfall_temp_weight: float = 0.3,
    rainfall_humidity_weight: float = 0.4,
    rainfall_orographic_weight: float = 0.3,
    forest_coverage: float = 0.15,
    desert_coverage: float = 0.10,
    max_lake_size: int | None = None,
) -> MapData:
    """
    Generate a complete procedural fantasy map.

    This function orchestrates all map generation steps in the correct sequence:
    terrain generation, hydrological features, vegetation placement,
    settlements, and roads.

    Args:
        width (int):
            The width of the map.
        height (int):
            The height of the map.
        padding (int):
            Padding around edges.
        scale (float):
            Noise scale for terrain generation.
        octaves (int):
            Number of noise octaves.
        persistence (float):
            Noise persistence.
        lacunarity (float):
            Noise lacunarity.
        smoothing_iterations (int):
            Number of terrain smoothing iterations.
        settlement_density (float):
            Density of settlements (0.0 to 1.0).
        min_settlement_radius (float):
            Minimum settlement radius.
        max_settlement_radius (float):
            Maximum settlement radius.
        seed (int):
            Random seed for reproducible generation. If None, uses random seed.
        enable_settlements (bool):
            Whether to generate settlements.
        enable_roads (bool):
            Whether to generate road networks.
        enable_rivers (bool):
            Whether to generate rivers.
        enable_vegetation (bool):
            Whether to place climate-driven vegetation.
        min_source_elevation (float):
            Minimum elevation for river sources (0.0-1.0).
        min_source_rainfall (float):
            Minimum rainfall percentile for sources (0.0-1.0).
        min_river_length (int):
            Minimum path length to place a river.
        sea_level (float):
            Elevation level for sea (controls land/sea ratio, -1.0 to 1.0).
        rainfall_temp_weight (float):
            Weight for temperature influence on rainfall (0.0 to 1.0).
        rainfall_humidity_weight (float):
            Weight for humidity influence on rainfall (0.0 to 1.0).
        rainfall_orographic_weight (float):
            Weight for orographic influence on rainfall (0.0 to 1.0).
        rainfall_variation_strength (float):
            Strength of random variation in rainfall (0.0 to 1.0).
        forest_coverage (float):
            Target forest coverage ratio (0.0 to 1.0).
        desert_coverage (float):
            Target desert coverage ratio (0.0 to 1.0).
        max_lake_size (int | None):
            Maximum size for edge-connected water bodies to be classified as lakes.
            Larger edge-connected bodies remain as salt water seas.
            If None, automatically calculated based on map size.

    Returns:
        MapData:
            The generated map data.

    Raises:
        ValueError:
            If any parameter has an invalid value.

    """
    # Validate parameters
    _validate_map_parameters(
        width,
        height,
        padding,
        scale,
        octaves,
        persistence,
        lacunarity,
        smoothing_iterations,
        settlement_density,
        min_settlement_radius,
        max_settlement_radius,
        sea_level,
    )

    # Record start time for performance measurement.
    start_time = time.time()
    logger.info(f"Starting map generation: {width}*{height}")

    # Set up random seed for reproducible results.
    if seed is None:
        seed = random.randint(0, 1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Using random seed: {seed}")

    # Retrieve organized tile collections.
    tile_collections = get_default_tile_collections()

    logger.debug("Initializing map data")
    # Create the map data object with tiles and empty grid.
    map_data = MapData(
        tiles=tile_collections.all_tiles,
        grid=[[0 for _ in range(width)] for _ in range(height)],
    )

    # Phase 1: Generate elevation map
    logger.debug("Generating elevation map")
    terrain_start = time.time()
    terrain.generate_elevation_map(
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
    logger.debug(f"Elevation map generation completed in {terrain_time:.3f}s")

    # Phase 1.5: Smooth elevation map
    logger.debug("Smoothing elevation map")
    smooth_start = time.time()
    terrain.smooth_elevation_map(
        map_data,
        iterations=smoothing_iterations,
        sigma=smoothing_sigma,
    )
    smooth_time = time.time() - smooth_start
    logger.debug(f"Elevation smoothing completed in {smooth_time:.3f}s")

    # Phase 2: Generate rainfall map (provides climate data for vegetation and
    # analysis)
    logger.debug("Generating rainfall map")
    rainfall_start = time.time()
    hydrology.generate_rainfall_map(
        map_data,
        width,
        height,
        temp_weight=rainfall_temp_weight,
        humidity_weight=rainfall_humidity_weight,
        orographic_weight=rainfall_orographic_weight,
    )
    rainfall_time = time.time() - rainfall_start
    logger.debug(f"Rainfall map generation completed in {rainfall_time:.3f}s")

    # Phase 2.5: Compute water accumulation (runoff)
    logger.debug("Computing water accumulation (runoff) map")
    accumulation_start = time.time()
    hydrology.generate_accumulation_map(map_data)
    accumulation_time = time.time() - accumulation_start
    logger.debug(f"Accumulation map generation completed in {accumulation_time:.3f}s")

    # Phase 3: Apply base terrain (elevation-driven only)
    logger.debug("Applying base terrain features")
    features_start = time.time()
    terrain.apply_base_terrain(map_data, tile_collections.base_terrain)
    features_time = time.time() - features_start
    logger.debug(f"Base terrain applied in {features_time:.3f}s")

    # Phase 3.5: Classify water bodies (seas vs lakes)
    logger.debug("Classifying water bodies as seas or lakes")
    water_classify_start = time.time()
    terrain.classify_water_bodies(map_data, max_lake_size=max_lake_size)
    water_classify_time = time.time() - water_classify_start
    logger.debug(f"Water bodies classified in {water_classify_time:.3f}s")

    # Phase 4: Place vegetation (climate-driven)
    if enable_vegetation:
        logger.debug("Placing climate-driven vegetation")
        vegetation_start = time.time()
        flora.place_vegetation(
            map_data,
            tile_collections.vegetation,
            forest_coverage=forest_coverage,
            desert_coverage=desert_coverage,
        )
        vegetation_time = time.time() - vegetation_start
        logger.debug(f"Vegetation placement completed in {vegetation_time:.3f}s")

    # Phase 5: Generate rivers using simple downhill flow
    if enable_rivers:
        logger.debug("Generating rivers")
        rivers_start = time.time()
        rivers.generate_rivers(
            map_data,
            min_source_elevation=min_source_elevation,
            min_source_rainfall=min_source_rainfall,
            min_river_length=min_river_length,
            sea_level=sea_level,
        )
        rivers_time = time.time() - rivers_start
        logger.debug(f"River generation completed in {rivers_time:.3f}s")

    # Phase 7: Generate settlements
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

    # Phase 8: Generate roads
    if enable_roads:
        logger.debug("Generating road network")
        roads_start = time.time()
        roads.generate_roads(map_data)
        roads_time = time.time() - roads_start
        logger.debug(f"Road network generated in {roads_time:.3f}s")
        logger.debug(f"Generated road network with {len(map_data.roads)} roads")

    # Phase 9: Generate water routes
    if enable_settlements and enable_roads:
        logger.debug("Generating water routes")
        water_routes_start = time.time()
        water_routes.generate_water_routes(map_data)
        water_routes_time = time.time() - water_routes_start
        logger.debug(f"Water routes generated in {water_routes_time:.3f}s")
        logger.debug(f"Generated water routes with {len(map_data.water_routes)} routes")

    end_time = time.time() - start_time
    logger.info(f"Map generation completed in {end_time:.3f}s")

    return map_data
