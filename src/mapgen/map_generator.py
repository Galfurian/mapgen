"""Main map generator module."""

import networkx as nx
import numpy as np

from . import (
    MapData,
    Settlement,
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
    sea_level: float
    mountain_level: float
    forest_threshold: float
    smoothing_iterations: int
    settlement_density: float
    min_settlement_radius: float
    max_settlement_radius: float
    # Tiles information.
    tiles: dict[str, Tile] | None
    default_tile: Tile | None
    # Generated data placeholders.
    map_data: MapData | None
    noise_map: np.ndarray | None
    settlements: list[Settlement] | None
    roads_graph: nx.Graph | None

    def __init__(
        self,
        width: int = 150,
        height: int = 100,
        padding: int = 2,
        scale: float = 50.0,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        sea_level: float = 0.03,
        mountain_level: float = 0.5,
        forest_threshold: float = 0.1,
        smoothing_iterations: int = 5,
        settlement_density: float = 0.002,
        min_settlement_radius: float = 0.5,
        max_settlement_radius: float = 1.0,
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
            sea_level (float): Sea level threshold.
            mountain_level (float): Mountain level threshold.
            forest_threshold (float): Forest threshold.
            smoothing_iterations (int): Number of smoothing iterations.
            settlement_density (float): Density of settlements.
            min_settlement_radius (float): Minimum settlement radius.
            max_settlement_radius (float): Maximum settlement radius.

        """
        # Map generation parameters.
        self.width = width
        self.height = height
        self.padding = padding
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.sea_level = sea_level
        self.mountain_level = mountain_level
        self.forest_threshold = forest_threshold
        self.smoothing_iterations = smoothing_iterations
        self.settlement_density = settlement_density
        self.min_settlement_radius = min_settlement_radius
        self.max_settlement_radius = max_settlement_radius
        # Generated data placeholders.
        self.map_data = None
        self.noise_map = None
        self.elevation_map = None
        self.settlements = None
        self.roads_graph = None

    def generate(self) -> None:
        """
        Generate the complete map.

        This method performs all steps to generate a procedural fantasy map
        including terrain, settlements, and roads.
        """
        logger.info(f"Starting map generation: {self.width}x{self.height}")

        tiles, default_index = self._get_default_tiles()

        # Initialize map.
        logger.debug("Initializing map level")
        self.map_data = MapData(
            tiles=tiles,
            grid=[
                [default_index for _ in range(self.width)] for _ in range(self.height)
            ],
        )

        # Dig
        logger.debug("Digging terrain")
        terrain.dig(
            map_data=self.map_data,
            padding=self.padding,
            initial_x=self.width // 2,
            initial_y=self.height // 2,
        )

        # Generate noise
        logger.debug("Generating noise map")
        self.noise_map = terrain.generate_noise_map(
            self.width,
            self.height,
            self.scale,
            self.octaves,
            self.persistence,
            self.lacunarity,
        )

        # Apply terrain features
        logger.debug("Applying terrain features")
        terrain.apply_terrain_features(
            self.map_data,
            self.noise_map,
            self.sea_level,
            self.mountain_level,
            self.forest_threshold,
        )

        # Smooth terrain
        logger.debug("Smoothing terrain")
        terrain.smooth_terrain(
            self.map_data,
            self.smoothing_iterations,
        )

        # Generate settlements
        logger.debug("Generating settlements")
        self.settlements = settlements.generate_settlements(
            self.map_data,
            self.noise_map,
            self.settlement_density,
            self.min_settlement_radius,
            self.max_settlement_radius,
        )
        logger.info(f"Generated {len(self.settlements)} settlements")

        # Generate roads
        logger.debug("Generating road network")
        self.roads_graph = roads.generate_roads(
            self.settlements,
            self.map_data,
            self.noise_map,
        )
        logger.info(f"Generated road network with {len(self.roads_graph.edges)} roads")
        logger.info("Map generation completed successfully")

    def _get_default_tiles(self) -> tuple[list[Tile], int]:
        """
        Create the catalog of tiles used for map generation.

        Returns:
            dict[str, Tile]: Dictionary mapping tile names to Tile instances.
            str: The name of the default tile.

        """
        tiles = [
            Tile(
                name="wall",
                description="Impassable wall",
                walkable=False,
                movement_cost=1.0,
                blocks_line_of_sight=True,
                buildable=False,
                habitability=0.0,
                road_buildable=False,
                elevation_penalty=0.0,
                elevation_influence=0.0,
                smoothing_weight=1.0,
                symbol="#",
                color=(0.0, 0.0, 0.0),
                resources=[],
            ),
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
                symbol="~",
                color=(0.2, 0.5, 1.0),
                resources=[],
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
                symbol="F",
                color=(0.2, 0.6, 0.2),
                resources=["wood", "game"],
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
                symbol=".",
                color=(0.8, 0.9, 0.6),
                resources=["grain", "herbs"],
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
                symbol="^",
                color=(0.5, 0.4, 0.3),
                resources=["stone", "ore"],
            ),
        ]
        return tiles, 0  # Default tile is "wall"
