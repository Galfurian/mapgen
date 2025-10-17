"""Data models for the map generator."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Position:
    """Represents a 2D coordinate position.

    This class encapsulates x and y coordinates for positions in the map grid.
    It's immutable (frozen) to ensure coordinate integrity.
    """

    x: int
    """The x-coordinate."""
    y: int
    """The y-coordinate."""

    def __iter__(self):
        """Allow tuple unpacking: x, y = position."""
        yield self.x
        yield self.y

    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position.

        Args:
            other (Position): The other position.

        Returns:
            float: The Euclidean distance.
        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def manhattan_distance_to(self, other: "Position") -> int:
        """Calculate Manhattan distance to another position.

        Args:
            other (Position): The other position.

        Returns:
            int: The Manhattan distance.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class Level:
    """Represents a 2D level grid with terrain data.

    This class encapsulates the level data and provides convenient methods
    for accessing and manipulating terrain information.
    """

    grid: List[List[str]]
    """The 2D grid of terrain types."""

    @property
    def height(self) -> int:
        """Get the height of the level."""
        return len(self.grid)

    @property
    def width(self) -> int:
        """Get the width of the level."""
        if self.height == 0:
            return 0
        return len(self.grid[0])

    def get_terrain(self, x: int, y: int) -> str:
        """Get the terrain type at the specified coordinates.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.

        Returns:
            str: The terrain type at the coordinates.

        Raises:
            IndexError: If coordinates are out of bounds.
        """
        return self.grid[y][x]

    def set_terrain(self, x: int, y: int, terrain: str) -> None:
        """Set the terrain type at the specified coordinates.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            terrain (str): The terrain type to set.

        Raises:
            IndexError: If coordinates are out of bounds.
        """
        self.grid[y][x] = terrain

    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if the given coordinates are within the level bounds.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.

        Returns:
            bool: True if coordinates are valid, False otherwise.
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def __getitem__(self, key: int) -> List[str]:
        """Get a row from the grid."""
        return self.grid[key]

    def __setitem__(self, key: int, value: List[str]) -> None:
        """Set a row in the grid."""
        self.grid[key] = value

    def __len__(self) -> int:
        """Get the height of the level."""
        return self.height