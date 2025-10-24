"""
Tile collections for organizing map tiles by placement method.

This module provides a structured way to organize tiles into distinct categories
based on how they are placed on the map, improving code organization and
separation of concerns.
"""

from dataclasses import dataclass, field

from .map_data import Tile, PlacementMethod


@dataclass
class TileCollections:
    """
    Organized collections of tiles by placement phase and type.

    This class separates tiles into four distinct categories based on how they
    are placed during map generation, enabling better organization and
    separation of concerns.

    Attributes:
        base_terrain (list[Tile]):
            Base terrain tiles placed using elevation data (e.g., sea, coast,
            plains, mountains). These form the foundation of the map.
        vegetation (list[Tile]):
            Vegetation tiles placed using climate data (e.g., forests, deserts,
            grasslands). These are placed after base terrain and follow
            ecological rules based on rainfall, temperature, and elevation.
        water_features (list[Tile]):
            Water feature tiles placed by algorithms (e.g., rivers, lakes,
            wetlands). These use flow-based and hydrological algorithms.
        infrastructure (list[Tile]):
            Infrastructure tiles placed by algorithms (e.g., roads, settlements,
            structures). These use pathfinding and placement algorithms.

    """

    base_terrain: list[Tile] = field(default_factory=list)
    vegetation: list[Tile] = field(default_factory=list)
    water_features: list[Tile] = field(default_factory=list)
    infrastructure: list[Tile] = field(default_factory=list)

    @property
    def all_tiles(self) -> list[Tile]:
        """
        Get all tiles in a single flat list.

        This property provides backward compatibility with code that expects
        a flat list of tiles. Tiles are returned in the order: base terrain,
        vegetation, water features, infrastructure.

        Returns:
            list[Tile]:
                All tiles from all collections concatenated together.

        """
        return (
            self.base_terrain
            + self.vegetation
            + self.water_features
            + self.infrastructure
        )

    def get_tile_by_name(self, name: str) -> Tile | None:
        """
        Find a tile by its name across all collections.

        Args:
            name (str):
                The name of the tile to find.

        Returns:
            Tile | None:
                The tile with the given name, or None if not found.

        """
        for tile in self.all_tiles:
            if tile.name == name:
                return tile
        return None

    def count_tiles(self) -> dict[str, int]:
        """
        Count the number of tiles in each collection.

        Returns:
            dict[str, int]:
                A dictionary mapping collection names to tile counts.

        """
        return {
            "base_terrain": len(self.base_terrain),
            "vegetation": len(self.vegetation),
            "water_features": len(self.water_features),
            "infrastructure": len(self.infrastructure),
            "total": len(self.all_tiles),
        }


def get_default_tile_collections() -> TileCollections:
    """
    Create organized tile collections for map generation.

    This function returns tiles organized by placement method, separating
    base terrain from vegetation and other features for better organization
    and more realistic placement.

    Returns:
        TileCollections:
            Organized tile collections with base_terrain, vegetation,
            water_features, and infrastructure.

    """
    collections = TileCollections()

    # Base terrain tiles (elevation-driven)
    collections.base_terrain.extend(
        [
            Tile(
                name="sea",
                description="Sea water terrain",
                walkable=True,
                movement_cost=2.0,
                blocks_line_of_sight=False,
                buildable=False,
                habitability=0.0,
                road_buildable=False,
                elevation_penalty=0.0,
                elevation_min=-1.0,
                elevation_max=+0.0,
                terrain_priority=1,
                symbol="~",
                color=(0.2, 0.5, 1.0),
                resources=[],
                is_water=True,
                is_salt_water=True,
                is_flowing_water=False,
                placement_method=PlacementMethod.TERRAIN_BASED,
            ),
            Tile(
                name="coast",
                description="Coastal shoreline terrain",
                walkable=True,
                movement_cost=1.5,
                blocks_line_of_sight=False,
                buildable=True,
                habitability=0.6,
                road_buildable=True,
                elevation_penalty=0.0,
                elevation_min=0.0,
                elevation_max=0.05,
                terrain_priority=2,
                symbol="c",
                color=(0.9647, 0.8627, 0.7412),
                resources=["fish", "salt"],
                placement_method=PlacementMethod.TERRAIN_BASED,
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
                elevation_min=0.05,
                elevation_max=0.5,
                terrain_priority=3,
                symbol=".",
                color=(0.8, 0.9, 0.6),
                resources=["grain", "herbs"],
                placement_method=PlacementMethod.TERRAIN_BASED,
                can_host_vegetation=True,
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
                elevation_min=0.5,
                elevation_max=1.0,
                terrain_priority=5,
                symbol="^",
                color=(0.5, 0.4, 0.3),
                resources=["stone", "ore"],
                placement_method=PlacementMethod.TERRAIN_BASED,
            ),
        ]
    )

    # Vegetation tiles (climate-driven)
    collections.vegetation.extend(
        [
            Tile(
                name="forest",
                description="Dense forest with tall trees",
                walkable=True,
                movement_cost=1.2,
                blocks_line_of_sight=False,
                buildable=True,
                habitability=0.7,
                road_buildable=True,
                elevation_penalty=0.0,
                elevation_min=0.0,
                elevation_max=0.6,
                terrain_priority=4,
                symbol="F",
                color=(0.2, 0.6, 0.2),
                resources=["wood", "game"],
                placement_method=PlacementMethod.ALGORITHM_BASED,
            ),
            Tile(
                name="grassland",
                description="Grassy terrain with scattered vegetation",
                walkable=True,
                movement_cost=1.0,
                blocks_line_of_sight=False,
                buildable=True,
                habitability=0.8,
                road_buildable=True,
                elevation_penalty=0.0,
                elevation_min=0.0,
                elevation_max=0.4,
                terrain_priority=3,
                symbol="g",
                color=(0.6, 0.8, 0.4),
                resources=["herbs", "game"],
                placement_method=PlacementMethod.ALGORITHM_BASED,
            ),
            Tile(
                name="desert",
                description="Arid desert terrain",
                walkable=True,
                movement_cost=1.3,
                blocks_line_of_sight=False,
                buildable=True,
                habitability=0.3,
                road_buildable=True,
                elevation_penalty=0.0,
                elevation_min=0.0,
                elevation_max=0.5,
                terrain_priority=3,
                symbol="d",
                color=(0.9, 0.8, 0.5),
                resources=[],
                placement_method=PlacementMethod.ALGORITHM_BASED,
            ),
        ]
    )

    # Water features (algorithm-based)
    collections.water_features.extend(
        [
            Tile(
                name="river",
                description="Flowing river water",
                walkable=True,
                movement_cost=1.5,
                blocks_line_of_sight=False,
                buildable=False,
                habitability=0.0,
                road_buildable=False,
                elevation_penalty=0.0,
                elevation_min=-0.5,
                elevation_max=1.0,
                terrain_priority=1,
                symbol="R",
                color=(0.1, 0.4, 0.9),
                resources=["fish"],
                is_water=True,
                is_salt_water=False,
                is_flowing_water=True,
                placement_method=PlacementMethod.ALGORITHM_BASED,
            ),
        ]
    )

    # Infrastructure tiles (currently empty, for future use)
    # collections.infrastructure.extend([...])

    return collections
