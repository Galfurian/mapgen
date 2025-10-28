"""
Data models for the map generator.

This module defines the core data structures used throughout the map generation
process, including tiles, positions, settlements, roads, and the main map data
container.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class Position:
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

    x: int
    y: int

    def distance_to(self, other: Position) -> float:
        """Calculate Euclidean distance to another position.

        Args:
            other (Position):
                The other position.

        Returns:
            float:
                The Euclidean distance.

        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def manhattan_distance_to(self, other: Position) -> int:
        """
        Calculate Manhattan distance to another position.

        Args:
            other (Position):
                The other position.

        Returns:
            int:
                The Manhattan distance.

        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __hash__(self) -> int:
        """Return hash based on coordinates."""
        return hash((self.x, self.y))


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
        elevation_min (float):
            Minimum elevation for terrain generation.
        elevation_max (float):
            Maximum elevation for terrain generation.
        terrain_priority (int):
            Priority for terrain assignment (higher = preferred).
        smoothing_priority (int):
            Priority for smoothing assignment (higher = preferred).
        diggable (bool):
            Whether this tile can be dug into (e.g., floor).
        symbol (str):
            Character symbol used for text-based map representation.
        color (tuple[float, float, float]):
            RGB color tuple for visualization (0.0 to 1.0).

    """

    # Metadata
    id: int = Field(
        description="Unique identifier for this tile.",
    )
    name: str = Field(
        description="Human-readable name for this tile type.",
    )
    description: str = Field(
        description="Detailed description of this tile type.",
    )
    is_base_tile: bool = Field(
        description="Whether this tile is a base terrain tile.",
    )
    # Core properties that drive all algorithms
    walkable: bool = Field(
        description="Whether units can move through this tile.",
    )
    movement_cost: float = Field(
        description="Base cost for pathfinding algorithms (higher = harder to traverse).",
    )
    # Settlement and building
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
    elevation_min: float = Field(
        default=0.0,
        description="Minimum elevation for terrain generation.",
    )
    elevation_max: float = Field(
        default=1.0,
        description="Maximum elevation for terrain generation.",
    )
    terrain_priority: int = Field(
        default=0,
        description="Priority for terrain assignment (higher = preferred).",
    )
    diggable: bool = Field(
        default=False,
        description="Whether this tile can be dug into (e.g., floor).",
    )
    # Visualization (optional - can be computed from other properties)
    symbol: str = Field(
        description="Character symbol used for text-based map representation.",
    )
    color: tuple[float, float, float] = Field(
        description="RGB color tuple for visualization (0.0 to 1.0).",
    )
    # Vegetation properties
    is_vegetation: bool = Field(
        default=False,
        description="Whether this tile is covered by vegetation.",
    )
    # Water properties
    is_water: bool = Field(
        default=False,
        description="Whether this tile represents any kind of water.",
    )
    is_salt_water: bool = Field(
        default=False,
        description="Whether this water tile is salt water (ocean, sea).",
    )
    is_flowing_water: bool = Field(
        default=False,
        description="Whether this water tile represents flowing water (rivers, streams).",
    )
    # Vegetation support
    can_host_vegetation: bool = Field(
        default=False,
        description="Whether this base terrain tile can support vegetation growth.",
    )

    def __hash__(self) -> int:
        """Return hash based on all tile properties."""
        return hash(
            (
                self.walkable,
                self.movement_cost,
                self.habitability,
                self.road_buildable,
                self.elevation_penalty,
                self.elevation_min,
                self.elevation_max,
                self.terrain_priority,
                self.diggable,
                self.symbol,
                self.color,
                self.name,
                self.description,
                self.is_water,
                self.is_salt_water,
                self.is_flowing_water,
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
        return self.habitability > 0.5

    @property
    def can_build_road(self) -> bool:
        """Check if roads can be built on this tile."""
        return self.road_buildable

    @property
    def pathfinding_cost(self) -> float:
        """Get the total cost for pathfinding algorithms."""
        return self.movement_cost + self.elevation_penalty

    @property
    def is_fresh_water(self) -> bool:
        """Check if this tile is fresh water."""
        return self.is_water and not self.is_salt_water

    @property
    def is_still_water(self) -> bool:
        """Check if this tile is still water."""
        return self.is_water and not self.is_flowing_water


class BodyOfWater(BaseModel):
    """
    Represents a body of water on the map.

    Attributes:
        body_of_water_type (BodyOfWaterType):
            The type of body of water.
        tiles (list[Position]):
            List of tiles occupied by this body of water.
    """

    is_salt_water: bool = Field(
        description="Whether this body of water is salt water, or not.",
    )
    tiles: list[Position] = Field(
        default_factory=list,
        description="List of tiles occupied by this body of water.",
    )


class Settlement(BaseModel):
    """
    Represents a settlement on the map.

    Settlements are population centers that can be connected by roads. They have
    a position, size, and connectivity properties that affect road generation
    and map visualization.

    Attributes:
        name (str):
            Human-readable name of the settlement.
        position (Position):
            The (x, y) coordinates of the settlement center.
        radius (float):
            Radius of the settlement (affects size and connectivity).
        connectivity (int):
            Road connectivity factor (higher = more likely to have roads).

    """

    name: str = Field(
        description="Human-readable name of the settlement.",
    )
    position: Position = Field(
        description="The (x, y) coordinates of the settlement center.",
    )
    radius: float = Field(
        description="Radius of the settlement (affects size and connectivity).",
    )
    connectivity: int = Field(
        description="Road connectivity factor (higher = more likely to have roads).",
    )
    is_harbor: bool = Field(
        default=False,
        description="Whether this settlement is a harbor.",
    )

    def distance_to(self, other: Settlement) -> float:
        """Calculate Euclidean distance to another position.

        Args:
            other (Settlement):
                The other settlement.

        Returns:
            float:
                The Euclidean distance.

        """
        return self.position.distance_to(other.position)


class Road(BaseModel):
    """
    Represents a road connecting two settlements.

    Attributes:
        start_settlement (str):
            Name of the starting settlement.
        end_settlement (str):
            Name of the ending settlement.
        path (list[Position]):
            List of positions forming the road path.

    """

    start_settlement: str = Field(
        description="Name of the starting settlement.",
    )
    end_settlement: str = Field(
        description="Name of the ending settlement.",
    )
    path: list[Position] = Field(
        default_factory=list,
        description="List of positions forming the road path.",
    )


class WaterRoute(BaseModel):
    """
    Represents a water route connecting two harbors.

    Attributes:
        start_harbor (str):
            Name of the starting harbor.
        end_harbor (str):
            Name of the ending harbor.
        path (list[Position]):
            List of positions forming the water route path.

    """

    start_harbor: str = Field(
        description="Name of the starting harbor.",
    )
    end_harbor: str = Field(
        description="Name of the ending harbor.",
    )
    path: list[Position] = Field(
        default_factory=list,
        description="List of positions forming the water route path.",
    )


class MapData(BaseModel):
    """
    Represents a 2D map grid with terrain data.

    This class encapsulates the map data and provides convenient methods for
    accessing and manipulating terrain information.

    Attributes:
        tiles (list[Tile]):
            List of unique tile types used in the map.
        grid (list[list[int]]):
            The 2D grid of tile indices.
        layers (dict[str, list[list[float]]]):
            Dictionary of spatial layers (terrain grid, elevation, rainfall,
            etc.).
        settlements (list[Settlement]):
            List of settlements on the map.
        roads (list[Road]):
            List of roads connecting settlements.

    """

    sea_level: float = Field(
        default=0.0,
        description="Elevation threshold below which tiles are considered ocean.",
    )
    tiles: list[Tile] = Field(
        default_factory=list,
        description="List of unique tile types used in the map.",
    )
    grid: list[list[int]] = Field(
        default_factory=list,
        description="The 2D grid of tile indices.",
    )
    layers: dict[str, list[list[float]]] = Field(
        default_factory=dict,
        description="Dictionary of spatial layers containing terrain and environmental data.",
    )
    settlements: list[Settlement] = Field(
        default_factory=list,
        description="List of settlements on the map.",
    )
    roads: list[Road] = Field(
        default_factory=list,
        description="List of roads connecting settlements.",
    )
    water_routes: list[WaterRoute] = Field(
        default_factory=list,
        description="List of water routes connecting harbors.",
    )
    bodies_of_water: list[BodyOfWater] = Field(
        default_factory=list,
        description="List of bodies of water on the map.",
    )

    @property
    def elevation_map(self) -> list[list[float]]:
        """Get the elevation map layer."""
        return self.layers.get("elevation", [])

    @elevation_map.setter
    def elevation_map(self, value: list[list[float]]) -> None:
        """Set the elevation map layer."""
        self.layers["elevation"] = value

    @property
    def rainfall_map(self) -> list[list[float]]:
        """Get the rainfall map layer."""
        return self.layers.get("rainfall", [])

    @rainfall_map.setter
    def rainfall_map(self, value: list[list[float]]) -> None:
        """Set the rainfall map layer."""
        self.layers["rainfall"] = value

    @property
    def temperature_map(self) -> list[list[float]]:
        """Get the temperature map layer."""
        return self.layers.get("temperature", [])

    @temperature_map.setter
    def temperature_map(self, value: list[list[float]]) -> None:
        """Set the temperature map layer."""
        self.layers["temperature"] = value

    @property
    def humidity_map(self) -> list[list[float]]:
        """Get the humidity map layer."""
        return self.layers.get("humidity", [])

    @humidity_map.setter
    def humidity_map(self, value: list[list[float]]) -> None:
        """Set the humidity map layer."""
        self.layers["humidity"] = value

    @property
    def orographic_map(self) -> list[list[float]]:
        """Get the orographic map layer."""
        return self.layers.get("orographic", [])

    @orographic_map.setter
    def orographic_map(self, value: list[list[float]]) -> None:
        """Set the orographic map layer."""
        self.layers["orographic"] = value

    @property
    def accumulation_map(self) -> list[list[float]]:
        """Get the accumulation map layer."""
        return self.layers.get("accumulation", [])

    @accumulation_map.setter
    def accumulation_map(self, value: list[list[float]]) -> None:
        """Set the accumulation map layer."""
        self.layers["accumulation"] = value

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

    def get_elevation(self, x: int, y: int) -> float:
        """
        Get the elevation at the specified coordinates.

        Args:
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.

        Returns:
            float:
                The elevation at the coordinates (0.0 if elevation data not
                available).

        Raises:
            IndexError:
                If coordinates are out of bounds when elevation data exists.

        """
        if not self.elevation_map:
            return 0.0
        return self.elevation_map[y][x]

    def get_terrain(self, x: int, y: int) -> Tile:
        """
        Get the terrain tile at the specified coordinates.

        Args:
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.

        Returns:
            Tile:
                The terrain tile at the coordinates.

        Raises:
            IndexError:
                If coordinates are out of bounds.

        """
        return self.tiles[self.grid[y][x]]

    def set_terrain(self, x: int, y: int, terrain: Tile) -> None:
        """
        Set the terrain tile at the specified coordinates.

        Args:
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.
            terrain (Tile):
                The terrain tile to set.

        Raises:
            IndexError:
                If coordinates are out of bounds.

        """
        self.grid[y][x] = terrain.id

    def is_valid_position(self, x: int, y: int) -> bool:
        """
        Check if the given coordinates are within the map bounds.

        Args:
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.

        Returns:
            bool:
                True if coordinates are valid, False otherwise.

        """
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(
        self,
        x: int,
        y: int,
        walkable_only: bool = True,
        include_diagonals: bool = False,
    ) -> list[Position]:
        """
        Get neighboring positions within map boundaries.

        Args:
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.
            walkable_only (bool):
                If True, only return walkable neighbors.
            include_diagonals (bool):
                If True, include diagonal neighbors (8 directions), otherwise
                only cardinal directions (4 directions).

        Returns:
            list[Position]:
                List of valid neighboring positions.

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
        """
        Get neighboring tiles within map boundaries.

        Args:
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.
            walkable_only (bool):
                If True, only return walkable neighbors.
            include_diagonals (bool):
                If True, include diagonal neighbors (8 directions), otherwise
                only cardinal directions (4 directions).

        Returns:
            list[Tile]:
                List of valid neighboring tiles.

        """
        return [
            self.get_terrain(pos.x, pos.y)
            for pos in self.get_neighbors(x, y, walkable_only, include_diagonals)
        ]

    def __getitem__(self, key: int) -> list[Tile]:
        """Get a row from the grid.

        Args:
            key (int):
                The row index.

        Returns:
            list[Tile]:
                The row of tiles.

        Raises:
            IndexError:
                If key is out of bounds.

        """
        return [self.tiles[i] for i in self.grid[key]]

    def __setitem__(self, key: int, value: list[Tile]) -> None:
        """Set a row in the grid.

        Args:
            key (int):
                The row index.
            value (list[Tile]):
                The row of tiles to set.

        Raises:
            IndexError:
                If key is out of bounds.

        """
        self.grid[key] = [tile.id for tile in value]

    def __len__(self) -> int:
        """Get the height of the map."""
        return self.height

    def save_to_json(self, filepath: str) -> None:
        """Save the map data to a JSON file.

        Args:
            filepath (str):
                Path to the file where the map will be saved.

        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_json(cls, filepath: str) -> MapData:
        """
        Load map data from a JSON file.

        Args:
            filepath (str):
                Path to the JSON file to load from.

        Returns:
            MapData:
                The loaded map data.

        """
        with open(filepath, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    @property
    def tiles_grid(self) -> list[list[Tile]]:
        """Get the grid as list of list of tiles."""
        return [[self.tiles[i] for i in row] for row in self.grid]

    def get_rainfall(self, x: int, y: int) -> float:
        """
        Get the rainfall at the specified coordinates.

        Args:
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.

        Returns:
            float:
                The rainfall at the coordinates (0.0 if rainfall data not
                available).

        Raises:
            IndexError:
                If coordinates are out of bounds when rainfall data exists.

        """
        if not self.rainfall_map:
            return 0.0
        return self.rainfall_map[y][x]

    def find_tiles_by_properties(self, **properties) -> list[Tile]:
        """
        Find tiles that match the given properties.

        Args:
            **properties:
                Keyword arguments representing tile properties to match.

        Returns:
            list[Tile]:
                List of tiles that match all the specified properties.

        Example:
            # Find all flowing water tiles flowing_water =
            map_data.find_tiles_by_properties(is_flowing_water=True)

            # Find all walkable, buildable tiles buildable_tiles =
            map_data.find_tiles_by_properties(
                walkable=True, buildable=True
            )

        """
        # Initialize list for matching tiles.
        matching_tiles = []
        # Check each tile.
        for tile in self.tiles:
            # Assume tile matches initially.
            matches = True
            # Check each property.
            for prop_name, prop_value in properties.items():
                # If tile doesn't have the property, no match.
                if not hasattr(tile, prop_name):
                    matches = False
                    break
                # If property value doesn't match, no match.
                if getattr(tile, prop_name) != prop_value:
                    matches = False
                    break
            # If all properties match, add to list.
            if matches:
                matching_tiles.append(tile)
        return matching_tiles

    def find_road(
        self,
        settlement_a: Settlement,
        settlement_b: Settlement,
    ) -> Road | None:
        """
        Find the road between two settlements.

        Args:
            settlement_a (Settlement):
                The first settlement.
            settlement_b (Settlement):
                The second settlement.

        Returns:
            Road | None:
                The road between the two settlements, or None if not found.
        """
        for road in self.roads:
            if (
                road.start_settlement == settlement_a.name
                and road.end_settlement == settlement_b.name
            ) or (
                road.start_settlement == settlement_b.name
                and road.end_settlement == settlement_a.name
            ):
                return road
        return None

    def find_water_route(
        self,
        settlement_a: Settlement,
        settlement_b: Settlement,
    ) -> WaterRoute | None:
        """
        Find the water route between two settlements.

        Args:
            settlement_a (Settlement):
                The first settlement.
            settlement_b (Settlement):
                The second settlement.

        Returns:
            WaterRoute | None:
                The water route between the two settlements, or None if not
                found.
        """
        for route in self.water_routes:
            if (
                route.start_harbor == settlement_a.name
                and route.end_harbor == settlement_b.name
            ) or (
                route.start_harbor == settlement_b.name
                and route.end_harbor == settlement_a.name
            ):
                return route
        return None


def get_default_tile_collections() -> list[Tile]:
    """
    Get a default set of tile collections for map generation.
    """
    return [
        Tile(
            id=0,
            name="sea",
            description="Sea water terrain",
            is_base_tile=True,
            walkable=True,
            movement_cost=2.0,
            habitability=0.0,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_min=-1.0,
            elevation_max=+0.0,
            terrain_priority=1,
            symbol="~",
            color=(0.2, 0.5, 1.0),
            is_water=True,
            is_salt_water=True,
            is_flowing_water=False,
        ),
        Tile(
            id=1,
            name="coast",
            description="Coastal shoreline terrain",
            is_base_tile=True,
            walkable=True,
            movement_cost=1.5,
            habitability=0.6,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_min=0.0,
            elevation_max=0.05,
            terrain_priority=2,
            symbol="c",
            color=(0.9647, 0.8627, 0.7412),
        ),
        Tile(
            id=2,
            name="plains",
            description="Open plains",
            is_base_tile=True,
            walkable=True,
            movement_cost=1.0,
            habitability=0.9,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_min=0.05,
            elevation_max=0.5,
            terrain_priority=3,
            symbol=".",
            color=(0.8, 0.9, 0.6),
            can_host_vegetation=True,
        ),
        Tile(
            id=3,
            name="mountain",
            description="Mountain terrain",
            is_base_tile=True,
            walkable=False,
            movement_cost=1.0,
            habitability=0.1,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_min=0.5,
            elevation_max=1.0,
            terrain_priority=5,
            symbol="^",
            color=(0.5, 0.4, 0.3),
        ),
        Tile(
            id=4,
            name="forest",
            description="Dense forest with tall trees",
            is_base_tile=False,
            walkable=True,
            movement_cost=1.2,
            habitability=0.7,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_min=0.0,
            elevation_max=0.6,
            terrain_priority=4,
            symbol="F",
            color=(0.2, 0.6, 0.2),
            is_vegetation=True,
        ),
        Tile(
            id=5,
            name="river",
            description="Flowing river water",
            is_base_tile=False,
            walkable=True,
            movement_cost=1.5,
            habitability=0.0,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_min=-0.5,
            elevation_max=1.0,
            terrain_priority=1,
            symbol="R",
            color=(0.2, 0.6, 0.7),
            is_water=True,
            is_salt_water=False,
            is_flowing_water=True,
        ),
        Tile(
            id=6,
            name="lake",
            description="Fresh water lake",
            is_base_tile=False,
            walkable=True,
            movement_cost=1.5,
            habitability=0.0,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_min=-0.5,
            elevation_max=1.0,
            terrain_priority=1,
            symbol="L",
            color=(0.3, 0.6, 0.7),
            is_water=True,
            is_salt_water=False,
            is_flowing_water=False,
        ),
    ]
