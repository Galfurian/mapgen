"""Data models for the map generator."""

import json
from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np

from . import logger


from pydantic import BaseModel, Field


class RoadType(Enum):
    """Enumeration of different road types for visualization and gameplay.

    Road types determine how roads are rendered and can affect gameplay mechanics.
    This enum can be extended with additional types like BRIDGE, TUNNEL, HIGHWAY, etc.

    Attributes:
        LAND: Standard land roads (rendered as curved solid lines)
        WATER: Roads that cross water bodies (rendered as straight dotted lines)

    """

    LAND = "land"
    WATER = "water"


class Tile(BaseModel):
    """
    Represents a single tile in the map with all properties needed for
    generation.

    This class encapsulates all the properties that drive map generation
    algorithms: settlement placement, pathfinding, road building, visualization,
    etc.

    Tiles are designed to be generic - the specific terrain semantics are
    defined by the properties, not hardcoded types. This allows the generator to
    work with any tile-based world: fantasy maps, dungeons, space, urban areas,
    etc.

    Attributes:
        name (str):
            Human-readable name for this tile type.
        description (str):
            Detailed description of this tile type.
        walkable (bool):
            Whether units can move through this tile.
        movement_cost (float):
            Base cost for pathfinding algorithms (higher = harder to traverse).
        blocks_line_of_sight (bool):
            Whether this tile blocks line of sight for visibility calculations.
        buildable (bool):
            Whether settlements and buildings can be constructed on this tile.
        habitability (float):
            How suitable this tile is for settlements (0.0 to 1.0).
        road_buildable (bool):
            Whether roads can be built on or through this tile.
        elevation_penalty (float):
            Additional pathfinding cost due to elevation changes.
        elevation_influence (float):
            How much this tile affects terrain elevation during generation.
        smoothing_weight (float):
            How much this tile participates in terrain smoothing algorithms.
        symbol (str):
            Character symbol used for text-based map representation.
        color (tuple[float, float, float]):
            RGB color tuple for visualization (0.0 to 1.0).
        resources (list[str]):
            list of resources available on this tile type.

    """

    # Metadata
    name: str = Field(
        description="Human-readable name for this tile type.",
    )
    description: str = Field(
        description="Detailed description of this tile type.",
    )
    # Core properties that drive all algorithms
    walkable: bool = Field(
        description="Whether units can move through this tile.",
    )
    movement_cost: float = Field(
        description="Base cost for pathfinding algorithms (higher = harder to traverse).",
    )
    blocks_line_of_sight: bool = Field(
        description="Whether this tile blocks line of sight for visibility calculations.",
    )
    # Settlement and building
    buildable: bool = Field(
        description="Whether settlements and buildings can be constructed on this tile.",
    )
    habitability: float = Field(
        description="How suitable this tile is for settlements (0.0 to 1.0).",
    )
    # Road generation
    road_buildable: bool = Field(
        description="Whether roads can be built on or through this tile.",
    )
    elevation_penalty: float = Field(
        description="Additional pathfinding cost due to elevation changes.",
    )
    # Terrain generation
    elevation_influence: float = Field(
        description="How much this tile affects terrain elevation during generation.",
    )
    smoothing_weight: float = Field(
        description="How much this tile participates in terrain smoothing algorithms.",
    )
    # Visualization (optional - can be computed from other properties)
    symbol: str = Field(
        description="Character symbol used for text-based map representation.",
    )
    color: tuple[float, float, float] = Field(
        description="RGB color tuple for visualization (0.0 to 1.0).",
    )
    # Resources and features.
    resources: list[str] = Field(
        default_factory=list,
        description="List of resources available on this tile type.",
    )

    def __hash__(self) -> int:
        """Return hash based on all tile properties."""
        return hash(
            (
                self.walkable,
                self.movement_cost,
                self.blocks_line_of_sight,
                self.buildable,
                self.habitability,
                self.road_buildable,
                self.elevation_penalty,
                self.elevation_influence,
                self.smoothing_weight,
                self.symbol,
                self.color,
                self.name,
                self.description,
                tuple(self.resources),
            )
        )

    # Computed properties
    @property
    def is_walkable(self) -> bool:
        """Check if this tile can be walked on."""
        return self.walkable

    @property
    def can_build_settlement(self) -> bool:
        """Check if settlements can be built on this tile."""
        return self.buildable and self.habitability > 0.5

    @property
    def can_build_road(self) -> bool:
        """Check if roads can be built on this tile."""
        return self.road_buildable

    @property
    def pathfinding_cost(self) -> float:
        """Get the total cost for pathfinding algorithms."""
        return self.movement_cost + self.elevation_penalty


class Position(BaseModel):
    """
    Represents a 2D coordinate position.

    This class encapsulates x and y coordinates for positions in the map grid.
    It's immutable (frozen) to ensure coordinate integrity.

    Attributes:
        x (int):
            The x-coordinate.
        y (int):
            The y-coordinate.

    """

    x: int = Field(
        description="The x-coordinate.",
    )
    y: int = Field(
        description="The y-coordinate.",
    )

    def distance_to(self, other) -> float:
        """Calculate Euclidean distance to another position.

        Args:
            other (Position): The other position.

        Returns:
            float: The Euclidean distance.

        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def manhattan_distance_to(self, other) -> int:
        """Calculate Manhattan distance to another position.

        Args:
            other (Position): The other position.

        Returns:
            int: The Manhattan distance.

        """
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class Settlement:
    """Represents a settlement on the map.

    Settlements are population centers that can be connected by roads.
    They have a position, size, and connectivity properties that affect
    road generation and map visualization.

    Attributes:
        x (int): X coordinate of the settlement center.
        y (int): Y coordinate of the settlement center.
        radius (float): Radius of the settlement (affects size and connectivity).
        name (str): Human-readable name of the settlement.
        connectivity (int): Road connectivity factor (higher = more likely to have roads).

    """

    x: int
    y: int
    radius: float
    name: str
    connectivity: int

    @property
    def position(self) -> Position:
        """Get the settlement position as a Position object."""
        return Position(x=self.x, y=self.y)

    def to_json(self) -> dict:
        """Convert this settlement to a JSON-serializable dictionary.

        Returns:
            dict: JSON-serializable representation of this settlement.
        """
        return {
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "name": self.name,
            "connectivity": self.connectivity,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Settlement":
        """Create a Settlement instance from a JSON dictionary.

        Args:
            data: JSON dictionary containing settlement properties.

        Returns:
            Settlement: New Settlement instance.
        """
        return cls(
            x=data["x"],
            y=data["y"],
            radius=data["radius"],
            name=data["name"],
            connectivity=data["connectivity"],
        )


@dataclass
class MapData:
    """Represents a 2D map grid with terrain data.

    This class encapsulates the map data and provides convenient methods
    for accessing and manipulating terrain information.

    Attributes:
        grid (list[list[Tile]]): The 2D grid of terrain tiles.
        noise_map (np.ndarray | None): The noise map used for terrain generation and visualization.
        elevation_map (np.ndarray | None): The elevation map derived from terrain features.
        settlements (list[Settlement] | None): List of settlements on the map.
        roads_graph (nx.Graph | None): The road network connecting settlements.

    """

    grid: list[list[Tile]]
    noise_map: np.ndarray | None = None
    elevation_map: np.ndarray | None = None
    settlements: list[Settlement] | None = None
    roads_graph: nx.Graph | None = None

    @property
    def height(self) -> int:
        """Get the height of the map."""
        return len(self.grid)

    @property
    def width(self) -> int:
        """Get the width of the map."""
        if self.height == 0:
            return 0
        return len(self.grid[0])

    def get_terrain(self, x: int, y: int) -> Tile:
        """Get the terrain tile at the specified coordinates.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.

        Returns:
            Tile: The terrain tile at the coordinates.

        Raises:
            IndexError: If coordinates are out of bounds.

        """
        return self.grid[y][x]

    def set_terrain(self, x: int, y: int, terrain: Tile) -> None:
        """Set the terrain tile at the specified coordinates.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            terrain (Tile): The terrain tile to set.

        Raises:
            IndexError: If coordinates are out of bounds.

        """
        self.grid[y][x] = terrain

    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if the given coordinates are within the map bounds.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.

        Returns:
            bool: True if coordinates are valid, False otherwise.

        """
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(
        self,
        x: int,
        y: int,
        walkable_only: bool = True,
        include_diagonals: bool = False,
    ) -> list[Position]:
        """Get neighboring positions within map boundaries.

        Args:
            x: The x coordinate.
            y: The y coordinate.
            walkable_only: If True, only return walkable neighbors.
            include_diagonals: If True, include diagonal neighbors (8 directions), otherwise only cardinal directions (4 directions).

        Returns:
            list[Position]: List of valid neighboring positions.

        """
        # Cardinal directions (4-way)
        neighbors = [
            Position(x=x - 1, y=y),  # left
            Position(x=x + 1, y=y),  # right
            Position(x=x, y=y - 1),  # up
            Position(x=x, y=y + 1),  # down
        ]

        # Add diagonal directions if requested (8-way)
        if include_diagonals:
            neighbors.extend(
                [
                    Position(x=x - 1, y=y - 1),  # top-left
                    Position(x=x + 1, y=y - 1),  # top-right
                    Position(x=x - 1, y=y + 1),  # bottom-left
                    Position(x=x + 1, y=y + 1),  # bottom-right
                ]
            )

        valid_neighbors = []
        for neighbor in neighbors:
            # If the neighbor is out of bounds, skip it.
            if not self.is_valid_position(neighbor.x, neighbor.y):
                continue
            # If walkable_only is True, skip non-walkable tiles.
            terrain = self.get_terrain(neighbor.x, neighbor.y)
            if walkable_only and not terrain.is_walkable:
                continue
            # If we passed all checks, add to valid neighbors.
            valid_neighbors.append(neighbor)

        return valid_neighbors

    def get_neighbor_tiles(
        self,
        x: int,
        y: int,
        walkable_only: bool = True,
        include_diagonals: bool = False,
    ) -> list[Tile]:
        """Get neighboring tiles within map boundaries.

        Args:
            x: The x coordinate.
            y: The y coordinate.
            walkable_only: If True, only return walkable neighbors.
            include_diagonals: If True, include diagonal neighbors (8 directions), otherwise only cardinal directions (4 directions).

        Returns:
            list[Tile]: List of valid neighboring tiles.

        """
        return [
            self.get_terrain(pos.x, pos.y)
            for pos in self.get_neighbors(x, y, walkable_only, include_diagonals)
        ]

    def to_json(self) -> dict:
        """Convert this map data to a JSON-serializable dictionary.

        Returns:
            dict: JSON-serializable representation of this map data.
        """
        # Collect unique tiles and assign sequential IDs
        unique_tiles: list[Tile] = []
        tile_to_id = {}

        for row in self.grid:
            for tile in row:
                if tile not in tile_to_id:
                    tile_id = len(unique_tiles)
                    tile_to_id[tile] = tile_id
                    unique_tiles.append(tile)

        # Create grid as comma-separated string of tile IDs
        grid_rows = []
        for row in self.grid:
            row_ids = [str(tile_to_id[tile]) for tile in row]
            grid_rows.append(",".join(row_ids))

        result = {
            "width": self.width,
            "height": self.height,
            "tiles": [tile.model_dump_json() for tile in unique_tiles],
            "grid": "\n".join(grid_rows),
        }

        # Add optional fields if they exist
        if self.noise_map is not None:
            result["noise_map"] = np.round(self.noise_map, 4).tolist()

        if self.elevation_map is not None:
            result["elevation_map"] = np.round(self.elevation_map, 4).tolist()

        if self.settlements is not None:
            result["settlements"] = [
                settlement.to_json() for settlement in self.settlements
            ]

        if self.roads_graph is not None:
            # Convert networkx graph to serializable format
            edges_data = []
            for u, v, data in self.roads_graph.edges(data=True):
                edge_data = {
                    "u": u,
                    "v": v,
                    "data": {
                        "path": [pos.to_json() for pos in data["path"]],
                        "type": data.get("type", RoadType.LAND).value,
                    },
                }
                edges_data.append(edge_data)
            result["roads_graph"] = edges_data

        return result

    @classmethod
    def from_json(cls, data: dict) -> "MapData":
        """Create a MapData instance from a JSON dictionary.

        Args:
            data: JSON dictionary containing map data.

        Returns:
            MapData: New MapData instance.
        """
        # Reconstruct tiles
        tiles = [Tile.model_validate_json(tile_data) for tile_data in data["tiles"]]

        # Reconstruct the grid from comma-separated strings
        grid = []
        for row_str in data["grid"].split("\n"):
            if row_str.strip():  # Skip empty lines
                row_ids = [int(id_str) for id_str in row_str.split(",")]
                row = [tiles[tile_id] for tile_id in row_ids]
                grid.append(row)

        # Create the basic MapData
        map_data = cls(grid=grid)

        # Add optional fields if they exist
        if "noise_map" in data:
            map_data.noise_map = np.array(data["noise_map"])

        if "elevation_map" in data:
            map_data.elevation_map = np.array(data["elevation_map"])

        if "settlements" in data:
            map_data.settlements = [
                Settlement.from_json(s_data) for s_data in data["settlements"]
            ]

        if "roads_graph" in data:
            # Reconstruct networkx graph
            graph = nx.Graph()
            for edge_data in data["roads_graph"]:
                u = edge_data["u"]
                v = edge_data["v"]
                path_data = edge_data["data"]["path"]
                path = [
                    Position.model_validate_json(pos_data) for pos_data in path_data
                ]
                road_type = RoadType(edge_data["data"]["type"])
                graph.add_edge(u, v, path=path, type=road_type)
            map_data.roads_graph = graph

        return map_data

    def save_to_json(self, filepath: str) -> None:
        """Save the map data to a JSON file.

        Args:
            filepath (str): Path to the file where the map will be saved.

        """
        logger.info(f"Saving map data to {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)
            logger.info(f"Map data saved successfully in {filepath}")

    @classmethod
    def load_from_json(cls, filepath: str) -> "MapData":
        """
        Load map data from a JSON file.

        Args:
            filepath (str): Path to the JSON file to load from.

        Returns:
            MapData: The loaded map data.

        """
        logger.info(f"Loading map data from {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def __getitem__(self, key: int) -> list[Tile]:
        """Get a row from the grid."""
        return self.grid[key]

    def __setitem__(self, key: int, value: list[Tile]) -> None:
        """Set a row in the grid."""
        self.grid[key] = value

    def __len__(self) -> int:
        """Get the height of the map."""
        return self.height
