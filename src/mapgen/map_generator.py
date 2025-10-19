"""Main map generator module."""

import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from . import (
    logger,
    roads,
    settlements,
    terrain,
    visualization,
    MapData,
    Settlement,
    Tile,
)


class MapGenerator:
    """Procedural fantasy map generator.

    This class generates fantasy maps using procedural techniques including
    terrain generation, settlements, and road networks.
    """

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
        seed: int | None = None,
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

        # Tile catalog - centralized tile definitions
        self.tiles = self._create_tile_catalog()

    def _create_tile_catalog(self) -> dict[str, Tile]:
        """Create the catalog of tiles used for map generation.

        Returns:
            dict[str, Tile]: Dictionary mapping tile names to Tile instances.

        """
        return {
            "wall": Tile(
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
            "floor": Tile(
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
            "water": Tile(
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
            "forest": Tile(
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
            "plains": Tile(
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
            "mountain": Tile(
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
        }

    def generate(self) -> MapData:
        """Generate the complete map.

        This method performs all steps to generate a procedural fantasy map
        including terrain, settlements, and roads.

        Returns:
            MapData: The generated map data containing terrain, settlements, and roads.
        """
        logger.info(f"Starting map generation: {self.width}x{self.height}")

        # Initialize map
        logger.debug("Initializing map level")
        map_data = terrain.initialize_level(
            self.width,
            self.height,
            self.tiles,
        )

        # Dig
        logger.debug("Digging terrain")
        terrain.dig(
            map_data=map_data,
            tiles=self.tiles,
            padding=self.padding,
            initial_x=self.width // 2,
            initial_y=self.height // 2,
        )

        # Generate noise
        logger.debug("Generating noise map")
        noise_map = terrain.generate_noise_map(
            self.width,
            self.height,
            self.scale,
            self.octaves,
            self.persistence,
            self.lacunarity,
        )

        # Apply terrain features
        logger.debug("Applying terrain features")
        map_data, elevation_map = terrain.apply_terrain_features(
            map_data,
            noise_map,
            self.tiles,
            self.sea_level,
            self.mountain_level,
            self.forest_threshold,
        )

        # Smooth terrain
        logger.debug("Smoothing terrain")
        map_data = terrain.smooth_terrain(
            map_data, self.tiles, self.smoothing_iterations
        )

        # Generate settlements
        logger.debug("Generating settlements")
        generated_settlements = settlements.generate_settlements(
            map_data,
            noise_map,
            self.settlement_density,
            self.min_settlement_radius,
            self.max_settlement_radius,
        )
        logger.info(
            f"Generated {len(generated_settlements) if generated_settlements else 0} settlements"
        )

        # Generate roads
        logger.debug("Generating road network")
        roads_graph = roads.generate_roads(
            generated_settlements, map_data, elevation_map
        )
        logger.info(
            f"Generated road network with {len(roads_graph.edges) if roads_graph else 0} roads"
        )

        # Create complete MapData with all components
        map_data.noise_map = noise_map
        map_data.elevation_map = elevation_map
        map_data.settlements = generated_settlements
        map_data.roads_graph = roads_graph

        logger.info("Map generation completed successfully")
        return map_data

    def plot(self, map_data: MapData) -> Figure:
        """Plot the generated map.

        Args:
            map_data: The map data to plot.

        Returns:
            Figure: The matplotlib figure of the map.

        Raises:
            ValueError: If required map data is missing.

        """
        if map_data.noise_map is None:
            raise ValueError(
                "Map data missing noise_map. Generate a complete map first."
            )
        if map_data.settlements is None:
            raise ValueError(
                "Map data missing settlements. Generate a complete map first."
            )
        if map_data.roads_graph is None:
            raise ValueError(
                "Map data missing roads_graph. Generate a complete map first."
            )
        if map_data.elevation_map is None:
            raise ValueError(
                "Map data missing elevation_map. Generate a complete map first."
            )

        return visualization.plot_map(
            map_data,
            map_data.noise_map,
            map_data.settlements,
            map_data.roads_graph,
            map_data.elevation_map,
        )
