"""
Tile collections for organizing map tiles by placement method.

This module provides a structured way to organize tiles into distinct categories
based on how they are placed on the map, improving code organization and
separation of concerns.
"""

from dataclasses import dataclass, field

from .map_data import Tile


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
