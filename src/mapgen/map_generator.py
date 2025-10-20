"""Main map generator module."""

import random

import numpy as np

from . import (
    MapData,
    Tile,
    logger,
    roads,
    settlements,
    terrain,
)


class MapGenerator:
    """Procedural fantasy map generator.

    This class generates fantasy maps using procedural techniques including
    terrain generation, settlements, and road networks.
    """

    # Map generation parameters.
    width: int
    height: int
    padding: int
    scale: float
    octaves: int
    persistence: float
    lacunarity: float
    smoothing_iterations: int
    settlement_density: float
    min_settlement_radius: float
    max_settlement_radius: float
    seed: int

    def __init__(
        self,
        width: int = 150,
        height: int = 100,
        padding: int = 2,
        scale: float = 50.0,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        smoothing_iterations: int = 5,
        settlement_density: float = 0.002,
        min_settlement_radius: float = 0.5,
        max_settlement_radius: float = 1.0,
        seed: int = random.randint(0, 1_000_000),
    ):
        """Initialize the map generator.

        Args:
            width (int): The width of the map.
            height (int): The height of the map.
            padding (int): Padding around edges.
            scale (float): Noise scale.
            octaves (int): Number of noise octaves.
            persistence (float): Noise persistence.
            lacunarity (float): Noise lacunarity.
            smoothing_iterations (int): Number of smoothing iterations.
            settlement_density (float): Density of settlements.
            min_settlement_radius (float): Minimum settlement radius.
            max_settlement_radius (float): Maximum settlement radius.
            seed (int): Random seed for reproducible generation.

        """
        # Map generation parameters.
        self.width = width
        self.height = height
        self.padding = padding
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.smoothing_iterations = smoothing_iterations
        self.settlement_density = settlement_density
        self.min_settlement_radius = min_settlement_radius
        self.max_settlement_radius = max_settlement_radius
        self.seed = seed

    def generate(self) -> MapData:
        """
        Generate the complete map.

        This method performs all steps to generate a procedural fantasy map
        including terrain, settlements, and roads.

        Returns:
            MapData:
                The generated map data.

        """
        logger.info(f"Starting map generation: {self.width}*{self.height}")

        # Set random seed for reproducible generation
        random.seed(self.seed)
        np.random.seed(self.seed)
        logger.debug(f"Using random seed: {self.seed}")

        tiles = self.get_default_tiles()

        logger.debug("Initializing map level")
        map_data = MapData(
            tiles=tiles,
            grid=[[0 for _ in range(self.width)] for _ in range(self.height)],
        )

        logger.debug("Generating noise map")
        terrain.generate_noise_map(
            map_data,
            self.width,
            self.height,
            self.scale,
            self.octaves,
            self.persistence,
            self.lacunarity,
        )

        logger.debug("Applying terrain features")
        terrain.apply_terrain_features(
            map_data,
        )

        logger.debug("Smoothing terrain")
        terrain.smooth_terrain(
            map_data,
            self.smoothing_iterations,
        )

        logger.debug("Generating settlements")
        settlements.generate_settlements(
            map_data,
            self.settlement_density,
            self.min_settlement_radius,
            self.max_settlement_radius,
        )
        logger.debug(f"Generated {len(map_data.settlements)} settlements")

        logger.debug("Generating road network")
        roads.generate_roads(
            map_data,
        )
        logger.debug(f"Generated road network with {len(map_data.roads)} roads")

        logger.info("Map generation completed successfully")

        return map_data

    @staticmethod
    def get_default_tiles() -> list[Tile]:
        """
        Create the catalog of tiles used for map generation.

        Returns:
            list[Tile]:
                List of Tile instances used in the map.

        """
        tiles = [
            Tile(
                name="floor",
                description="Open floor space",
                walkable=True,
                movement_cost=1.0,
                blocks_line_of_sight=False,
                buildable=True,
                habitability=0.3,
                road_buildable=True,
                elevation_penalty=0.0,
                elevation_influence=0.0,
                smoothing_weight=1.0,
                elevation_min=0.0,
                elevation_max=0.0,
                terrain_priority=0,
                smoothing_priority=0,
                diggable=True,
                symbol=".",
                color=(0.9, 0.9, 0.9),
                resources=[],
            ),
            Tile(
                name="water",
                description="Water terrain",
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
                elevation_max=0.03,
                terrain_priority=1,
                smoothing_priority=2,
                symbol="~",
                color=(0.2, 0.5, 1.0),
                resources=[],
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
                elevation_min=0.03,
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
        return tiles
