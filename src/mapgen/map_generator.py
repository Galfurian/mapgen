"""Main map generator module."""

import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from . import roads, settlements, terrain, visualization
from .map import Map


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

        self.map: Map | None = None
        self.noise_map: np.ndarray | None = None
        self.elevation_map: np.ndarray | None = None
        self.settlements: list[dict] | None = None
        self.roads_graph: nx.Graph | None = None

    def generate(self) -> None:
        """Generate the complete map.

        This method performs all steps to generate a procedural fantasy map
        including terrain, settlements, and roads.
        """
        # Initialize map
        self.map = terrain.initialize_level(self.width, self.height)

        # Initialize character
        character = terrain.initialize_character(
            self.width, self.height, self.padding, self.wall_countdown
        )

        # Dig
        terrain.dig(self.map, character)

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
            self.sea_level,
            self.mountain_level,
            self.forest_threshold,
        )

        # Smooth terrain
        self.map = terrain.smooth_terrain(self.map, self.smoothing_iterations)

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
