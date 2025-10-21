"""Generator classes for procedural map generation."""

import logging
from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from .map_data import MapData, PlacementMethod
from .rivers import generate_rivers, RiverGenerator, RiverConfig
from .hydrology import detect_lakes


logger = logging.getLogger(__name__)


class GenerationConfig(Protocol):
    """Protocol for generation configuration."""
    pass


class TerrainConfig(GenerationConfig):
    """Configuration for terrain generation."""
    pass


class LakeConfig(GenerationConfig):
    """Configuration for lake generation."""
    
    def __init__(
        self,
        min_accumulation: float = 10.0,
        min_lake_size: int = 5,
        max_elevation: float = -0.1,
    ):
        self.min_accumulation = min_accumulation
        self.min_lake_size = min_lake_size
        self.max_elevation = max_elevation


class BaseGenerator(ABC):
    """Abstract base class for all map generators."""

    @abstractmethod
    def generate(self, map_data: MapData, config: GenerationConfig | None = None) -> None:
        """Generate features for the map.

        Args:
            map_data: The map data to modify
            config: Configuration for this generator (optional)
        """
        pass


class TerrainGenerator(BaseGenerator):
    """Generator for base terrain features based on elevation and climate."""

    def generate(self, map_data: MapData, config: GenerationConfig | None = None) -> None:
        """
        Apply base terrain features based on elevation and rainfall.

        Only assigns terrain-based tiles (mountains, plains, forests, sea).
        Algorithm-based tiles (rivers, lakes) are handled by other generators.

        Args:
            map_data: The map data to modify
            config: Terrain generation configuration
        """
        sorted_tiles = sorted(
            [tile for tile in map_data.tiles if tile.placement_method == PlacementMethod.TERRAIN_BASED],
            key=lambda t: t.terrain_priority,
            reverse=True,
        )

        def apply_suitable_tile(x: int, y: int) -> bool:
            """
            Apply the most suitable terrain-based tile for the given position.
            """
            elevation = map_data.get_elevation(x, y)
            rainfall = map_data.get_rainfall(x, y)

            # Filter tiles based on elevation - only terrain-based tiles
            suitable_tiles = [
                tile for tile in sorted_tiles
                if tile.elevation_min <= elevation <= tile.elevation_max
            ]

            if not suitable_tiles:
                return False

            # If we have multiple suitable tiles, use rainfall to choose
            if len(suitable_tiles) > 1:
                # Use rainfall to influence tile selection
                # High rainfall favors tiles that thrive in wet conditions (forests)
                # Low rainfall favors tiles that thrive in dry conditions (plains)
                wet_preferred_tiles = [
                    tile for tile in suitable_tiles
                    if "wood" in tile.resources or "game" in tile.resources
                ]
                dry_preferred_tiles = [
                    tile for tile in suitable_tiles
                    if "grain" in tile.resources or "herbs" in tile.resources
                ]

                if wet_preferred_tiles and dry_preferred_tiles:
                    # We have both wet and dry preferring tiles
                    if rainfall > 0.6:
                        # High rainfall - prefer wet tiles (forests)
                        chosen_tile = max(wet_preferred_tiles, key=lambda t: t.terrain_priority)
                    elif rainfall < 0.3:
                        # Low rainfall - prefer dry tiles (plains)
                        chosen_tile = max(dry_preferred_tiles, key=lambda t: t.terrain_priority)
                    else:
                        # Moderate rainfall - use highest priority
                        chosen_tile = max(suitable_tiles, key=lambda t: t.terrain_priority)
                else:
                    # For other cases, use highest priority
                    chosen_tile = max(suitable_tiles, key=lambda t: t.terrain_priority)
            else:
                chosen_tile = suitable_tiles[0]

            map_data.set_terrain(x, y, chosen_tile)
            return True

        for y in range(map_data.height):
            for x in range(map_data.width):
                if not apply_suitable_tile(x, y):
                    logger.warning(
                        f"No suitable terrain tile found for elevation {map_data.get_elevation(x, y)} at ({x}, {y})"
                    )


class LakeGenerator(BaseGenerator):
    """Generator for lake features based on water accumulation."""

    def generate(self, map_data: MapData, config: GenerationConfig | None = None) -> None:
        """
        Detect and apply lakes to the map based on water accumulation.

        Lakes are detected in low elevation areas with high water accumulation.

        Args:
            map_data: The map data to modify
            config: Lake generation configuration
        """
        if config is None:
            config = LakeConfig()
        
        if not isinstance(config, LakeConfig):
            raise ValueError(f"LakeGenerator requires LakeConfig, got {type(config)}")
        
        # Convert map data to numpy arrays for lake detection
        elevation_array = np.array(map_data.elevation_map)
        accumulation_array = np.array(map_data.accumulation_map)
        
        # Detect lakes
        lakes = detect_lakes(
            elevation=elevation_array,
            accumulation=accumulation_array,
            min_accumulation=config.min_accumulation,
            min_lake_size=config.min_lake_size,
            max_elevation=config.max_elevation,
        )
        
        # Find lake tiles (still fresh water)
        lake_tiles = map_data.find_tiles_by_properties(
            is_water=True, is_salt_water=False, is_flowing_water=False
        )
        
        if not lake_tiles:
            logger.warning("No lake tiles found in tile catalog")
            return
        
        # Use the first matching lake tile
        lake_tile = lake_tiles[0]
        
        # Apply lake tiles to detected lake positions
        for lake in lakes:
            for position in lake.tiles:
                # Only replace non-flowing water tiles with lakes
                current_tile = map_data.get_terrain(position.x, position.y)
                if not current_tile.is_flowing_water:
                    map_data.set_terrain(position.x, position.y, lake_tile)
            
            # Store the lake in map_data
            map_data.lakes.append(lake)
            
            logger.debug(f"Applied lake with {len(lake.tiles)} tiles at center {lake.center}")


class MapBuilder:
    """Orchestrator for map generation using separate generator classes."""
    
    def __init__(self):
        self.generators: list[tuple[BaseGenerator, GenerationConfig | None]] = []
    
    def add_generator(self, generator: BaseGenerator, config: GenerationConfig | None = None) -> None:
        """Add a generator to the build pipeline.
        
        Args:
            generator: The generator to add
            config: Configuration for the generator
        """
        self.generators.append((generator, config))
    
    def build_map(self, map_data: MapData) -> None:
        """Build the complete map using all registered generators.
        
        Args:
            map_data: The map data to build upon
        """
        logger.info(f"Building map with {len(self.generators)} generators")
        
        for i, (generator, config) in enumerate(self.generators):
            generator_name = generator.__class__.__name__
            logger.debug(f"Running generator {i+1}/{len(self.generators)}: {generator_name}")
            generator.generate(map_data, config)
        
        logger.info("Map building complete")