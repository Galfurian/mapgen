"""Main map generator module."""

import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from . import roads, settlements, terrain, visualization
from .map import Map, Tile, Settlement


class MapGenerator:
    """Procedural fantasy map generator.

    This class generates fantasy maps using procedural techniques including
    terrain generation, settlements, and road networks.
    """

    def __init__(
        self,
        width: int = 150,
        height: int = 100,
        wall_countdown: int = 8000,
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
            wall_countdown (int): Number of walls to dig.
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
        self.wall_countdown = wall_countdown
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

        self.map: Map | None = None
        self.noise_map: np.ndarray | None = None
        self.elevation_map: np.ndarray | None = None
        self.settlements: list[Settlement] | None = None
        self.roads_graph: nx.Graph | None = None

    def _create_tile_catalog(self) -> dict[str, Tile]:
        """Create the catalog of tiles used for map generation.

        Returns:
            dict[str, Tile]: Dictionary mapping tile names to Tile instances.
        """
        return {
            "wall": Tile(
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
                name="wall",
                description="Impassable wall",
                resources=[],
            ),
            "floor": Tile(
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
                name="floor",
                description="Open floor space",
                resources=[],
            ),
            "water": Tile(
                walkable=True,
                movement_cost=2.0,
                blocks_line_of_sight=False,
                buildable=False,
                habitability=0.0,
                road_buildable=True,
                elevation_penalty=0.0,
                elevation_influence=-0.5,
                smoothing_weight=1.0,
                symbol="~",
                color=(0.2, 0.5, 1.0),
                name="water",
                description="Water terrain",
                resources=[],
            ),
            "forest": Tile(
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
                name="forest",
                description="Forest terrain",
                resources=["wood", "game"],
            ),
            "plains": Tile(
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
                name="plains",
                description="Open plains",
                resources=["grain", "herbs"],
            ),
            "mountain": Tile(
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
                name="mountain",
                description="Mountain terrain",
                resources=["stone", "ore"],
            ),
        }

    def generate(self) -> None:
        """Generate the complete map.

        This method performs all steps to generate a procedural fantasy map
        including terrain, settlements, and roads.
        """
        # Initialize map
        self.map = terrain.initialize_level(self.width, self.height, self.tiles)

        # Initialize character
        character = terrain.initialize_character(
            self.width, self.height, self.padding, self.wall_countdown
        )

        # Dig
        terrain.dig(self.map, character, self.tiles)

        # Generate noise
        self.noise_map = terrain.generate_noise_map(
            self.width,
            self.height,
            self.scale,
            self.octaves,
            self.persistence,
            self.lacunarity,
        )

        # Apply terrain features
        self.map, self.elevation_map = terrain.apply_terrain_features(
            self.map,
            self.noise_map,
            self.tiles,
            self.sea_level,
            self.mountain_level,
            self.forest_threshold,
        )

        # Smooth terrain
        self.map = terrain.smooth_terrain(self.map, self.tiles, self.smoothing_iterations)

        # Generate settlements
        self.settlements = settlements.generate_settlements(
            self.map,
            self.noise_map,
            self.settlement_density,
            self.min_settlement_radius,
            self.max_settlement_radius,
        )

        # Generate roads
        self.roads_graph = roads.generate_roads(
            self.settlements, self.map, self.elevation_map
        )

    def plot(self) -> Figure:
        """Plot the generated map.

        Returns:
            Figure: The matplotlib figure of the map.

        Raises:
            ValueError: If the map has not been generated yet.

        """
        if self.map is None:
            raise ValueError("Map not generated yet. Call generate() first.")
        assert self.noise_map is not None
        assert self.settlements is not None
        assert self.roads_graph is not None
        assert self.elevation_map is not None
        return visualization.plot_map(
            self.map,
            self.noise_map,
            self.settlements,
            self.roads_graph,
            self.elevation_map,
        )