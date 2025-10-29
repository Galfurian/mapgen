"""
Flora and vegetation placement algorithms.

This module implements climate-driven vegetation placement that runs after
base terrain generation. Vegetation placement uses an organism-based growth
simulation where vegetation spreads, competes, and dies based on environmental
suitability.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .map_data import MapData, Tile

logger = logging.getLogger(__name__)


@dataclass
class VegetationInstance:
    """Represents a single vegetation instance with age and tile ID."""

    tile_id: int
    age: int = 0

    def is_alive(self, max_age: int) -> bool:
        """Check if this vegetation instance is still alive."""
        return self.age < max_age


def calculate_suitability(
    tile: Tile,
    elevation: float,
    rainfall: float,
    temperature: float,
    humidity: float,
    water_proximity: bool = False,
    slope: float = 0.0,
) -> float:
    """
    Calculate suitability score for a vegetation tile at given conditions.

    Returns a score from 0.0 (completely unsuitable) to 1.0 (perfectly suitable).
    """
    if not tile.suitability_params:
        return 0.0

    params = tile.suitability_params

    # Check viability ranges
    temp_min, temp_max = params["temperature_range"]
    if not (temp_min <= temperature <= temp_max):
        return 0.0

    rain_min, rain_max = params["rainfall_range"]
    if not (rain_min <= rainfall <= rain_max):
        return 0.0

    elev_min, elev_max = params["elevation_range"]
    if not (elev_min <= elevation <= elev_max):
        return 0.0

    humid_min, humid_max = params["humidity_range"]
    if not (humid_min <= humidity <= humid_max):
        return 0.0

    # Calculate base suitability (how close to optimal conditions)
    temp_optimal = (temp_min + temp_max) / 2
    rain_optimal = (rain_min + rain_max) / 2
    elev_optimal = (elev_min + elev_max) / 2
    humid_optimal = (humid_min + humid_max) / 2

    temp_score = 1.0 - abs(temperature - temp_optimal) / ((temp_max - temp_min) / 2)
    rain_score = 1.0 - abs(rainfall - rain_optimal) / ((rain_max - rain_min) / 2)
    elev_score = 1.0 - abs(elevation - elev_optimal) / ((elev_max - elev_min) / 2)
    humid_score = 1.0 - abs(humidity - humid_optimal) / ((humid_max - humid_min) / 2)

    base_score = (temp_score + rain_score + elev_score + humid_score) / 4.0

    # Apply environmental modifiers
    score = base_score

    if water_proximity and "water_proximity_bonus" in params:
        score *= params["water_proximity_bonus"]

    if slope > 0.3 and "slope_penalty" in params:  # Steep slope
        score *= params["slope_penalty"]

    return min(1.0, max(0.0, score))


def place_vegetation(
    map_data: MapData,
) -> None:
    """
    Place vegetation using organism-based growth simulation.

    This function simulates vegetation as organisms that grow, compete, and die
    based on environmental suitability. The process includes:
    1. Seeding initial vegetation in highly suitable areas
    2. Multiple growth iterations with spreading, competition, and mortality
    3. Emergent patterns that create realistic vegetation distributions

    Args:
        map_data (MapData):
            The map data to modify. Must have elevation, rainfall, temperature,
            and humidity maps.
    """
    if not map_data.elevation_map:
        logger.warning("Elevation map is required for vegetation placement")
        return
    if not map_data.rainfall_map:
        logger.warning("Rainfall map is required for vegetation placement")
        return
    if not map_data.temperature_map:
        logger.warning("Temperature map is recommended for vegetation placement")
        return
    if not map_data.humidity_map:
        logger.warning("Humidity map is required for vegetation placement")
        return

    vegetation_tiles = map_data.find_tiles_by_properties(
        is_vegetation=True,
    )
    if not vegetation_tiles:
        logger.warning("No vegetation tiles found in tile catalog")
        return

    # Find tiles that can host vegetation (like plains)
    host_tiles = map_data.find_tiles_by_properties(
        can_host_vegetation=True,
    )
    if not host_tiles:
        logger.warning("No tiles that can host vegetation found")
        return

    logger.debug("Starting organism-based vegetation placement")

    # Group vegetation tiles by tile ID
    vegetation_by_id = {}
    for veg_tile in vegetation_tiles:
        if veg_tile.suitability_params:
            vegetation_by_id[veg_tile.id] = veg_tile

    if not vegetation_by_id:
        logger.warning("No vegetation tiles with suitability parameters found")
        return

    # Initialize vegetation tracking
    vegetation_grid: List[List[Optional[VegetationInstance]]] = [
        [None for _ in range(map_data.width)] for _ in range(map_data.height)
    ]

    # Phase 1: Seeding - Place initial vegetation in highly suitable areas
    seeded_count = 0
    for y in range(map_data.height):
        for x in range(map_data.width):
            current_tile = map_data.get_terrain(x, y)
            if not current_tile.can_host_vegetation:
                continue

            elevation = map_data.elevation_map[y][x]
            rainfall = map_data.rainfall_map[y][x]
            temperature = map_data.temperature_map[y][x]
            humidity = map_data.humidity_map[y][x]

            # Check each vegetation type for seeding
            for tile_id, veg_tile in vegetation_by_id.items():
                suitability = calculate_suitability(
                    veg_tile,
                    elevation,
                    rainfall,
                    temperature,
                    humidity,
                )

                seed_threshold = veg_tile.suitability_params.get("seed_threshold", 0.8)
                if suitability >= seed_threshold:
                    # Seed this vegetation type
                    vegetation_grid[y][x] = VegetationInstance(tile_id)
                    seeded_count += 1
                    break  # Only seed one type per location

    # Phase 2: Growth simulation - Multiple iterations
    logger.debug(f"Seeded {seeded_count} initial vegetation instances")

    # More iterations for better spread
    growth_iterations = 120

    for iteration in range(growth_iterations):
        logger.debug(f"Growth iteration {iteration + 1}/{growth_iterations}")

        # Create a copy of the current vegetation grid for this iteration
        new_vegetation_grid = [[veg for veg in row] for row in vegetation_grid]

        # Age all existing vegetation
        for y in range(map_data.height):
            for x in range(map_data.width):
                if vegetation_grid[y][x] is not None:
                    vegetation_grid[y][x].age += 1

        # Attempt growth to adjacent tiles
        growth_attempts = 0
        successful_growth = 0

        for y in range(map_data.height):
            for x in range(map_data.width):
                if not vegetation_grid[y][x]:
                    continue

                veg_instance = vegetation_grid[y][x]
                if veg_instance is None:
                    continue

                veg_tile = vegetation_by_id.get(veg_instance.tile_id)
                if not veg_tile:
                    continue

                params = veg_tile.suitability_params
                growth_prob = params.get("growth_probability", 0.2)

                # Try to spread to adjacent tiles
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Cardinal directions
                    ny, nx = y + dy, x + dx

                    if not (0 <= ny < map_data.height and 0 <= nx < map_data.width):
                        continue

                    # Only spread to host tiles that don't already have vegetation
                    target_tile = map_data.get_terrain(nx, ny)
                    if (
                        not target_tile.can_host_vegetation
                        or new_vegetation_grid[ny][nx]
                    ):
                        continue

                    # Calculate suitability at target location
                    elevation = map_data.elevation_map[ny][nx]
                    rainfall = map_data.rainfall_map[ny][nx]
                    temperature = (
                        map_data.temperature_map[ny][nx]
                        if map_data.temperature_map
                        else 0.5
                    )
                    humidity = map_data.humidity_map[ny][nx]

                    suitability = calculate_suitability(
                        veg_tile, elevation, rainfall, temperature, humidity
                    )

                    # Attempt growth based on suitability and growth probability
                    if suitability > 0.2 and random.random() < (
                        growth_prob * suitability
                    ):
                        growth_attempts += 1

                        # Check for competition if another type wants this spot
                        competing_types = []
                        for other_id, other_tile in vegetation_by_id.items():
                            if other_id == veg_instance.tile_id:
                                continue

                            other_suitability = calculate_suitability(
                                other_tile, elevation, rainfall, temperature, humidity
                            )

                            if other_suitability > 0.3:
                                competing_types.append(
                                    (
                                        other_id,
                                        other_suitability,
                                        other_tile.suitability_params.get(
                                            "competition_strength", 0.5
                                        ),
                                    )
                                )

                        # If no competition, place vegetation
                        if not competing_types:
                            new_vegetation_grid[ny][nx] = VegetationInstance(
                                veg_instance.tile_id
                            )
                            successful_growth += 1
                        else:
                            # Competition resolution
                            our_strength = params.get("competition_strength", 0.5)
                            our_score = suitability * our_strength

                            winner = veg_instance.tile_id
                            best_score = our_score

                            for (
                                comp_id,
                                comp_suitability,
                                comp_strength,
                            ) in competing_types:
                                comp_score = comp_suitability * comp_strength
                                if comp_score > best_score:
                                    winner = comp_id
                                    best_score = comp_score

                            if winner == veg_instance.tile_id:
                                new_vegetation_grid[ny][nx] = VegetationInstance(
                                    veg_instance.tile_id
                                )
                                successful_growth += 1

        logger.debug(
            f"  Growth attempts: {growth_attempts}, successful: {successful_growth}"
        )

        # Apply mortality
        mortality_events = 0
        for y in range(map_data.height):
            for x in range(map_data.width):
                if not vegetation_grid[y][x]:
                    continue

                veg_instance = vegetation_grid[y][x]
                if veg_instance is None:
                    continue

                veg_tile = vegetation_by_id.get(veg_instance.tile_id)
                if not veg_tile:
                    continue

                params = veg_tile.suitability_params
                mortality_rate = params.get("mortality_rate", 0.05)
                max_age = params.get("max_age", 20)

                # Calculate local suitability for mortality check
                elevation = map_data.elevation_map[y][x]
                rainfall = map_data.rainfall_map[y][x]
                temperature = (
                    map_data.temperature_map[y][x] if map_data.temperature_map else 0.5
                )
                humidity = map_data.humidity_map[y][x]

                suitability = calculate_suitability(
                    veg_tile, elevation, rainfall, temperature, humidity
                )

                # Calculate clustering bonus - vegetation survives better in groups
                clustering_bonus = 0.0
                neighbor_count = 0
                same_type_neighbors = 0
                
                # Check all 8 neighboring tiles
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:  # Skip center tile
                            continue
                            
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < map_data.height and 0 <= nx < map_data.width:
                            neighbor_count += 1
                            neighbor_veg = vegetation_grid[ny][nx]
                            if (neighbor_veg is not None and 
                                neighbor_veg.tile_id == veg_instance.tile_id):
                                same_type_neighbors += 1
                
                # Clustering bonus: reduce mortality when surrounded by similar vegetation
                if neighbor_count > 0:
                    clustering_factor = same_type_neighbors / neighbor_count
                    clustering_bonus = clustering_factor * params.get("clustering_strength", 0.3)
                
                # Higher mortality in unsuitable conditions and for old vegetation
                age_factor = veg_instance.age / max_age
                suitability_factor = 1.0 - suitability

                effective_mortality = (
                    mortality_rate + (age_factor * 0.1) + (suitability_factor * 0.2)
                )
                
                # Apply clustering bonus (reduces mortality)
                effective_mortality = max(0.0, effective_mortality - clustering_bonus)

                if random.random() < effective_mortality or veg_instance.age >= max_age:
                    new_vegetation_grid[y][x] = None
                    mortality_events += 1

        logger.debug(f"  Mortality events: {mortality_events}")

        # Update vegetation grid for next iteration
        vegetation_grid = new_vegetation_grid

    # Phase 3: Apply vegetation to map
    final_vegetation_count = 0
    for y in range(map_data.height):
        for x in range(map_data.width):
            if vegetation_grid[y][x] is not None:
                veg_instance = vegetation_grid[y][x]
                if veg_instance is not None:
                    veg_tile = vegetation_by_id.get(veg_instance.tile_id)
                    if veg_tile:
                        map_data.set_terrain(x, y, veg_tile)
                        final_vegetation_count += 1

    logger.debug(f"Vegetation placement complete: {final_vegetation_count} final tiles")
