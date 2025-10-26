"""
Settlement generation module for procedural map generation.

This module handles the generation and placement of settlements on the map.
It includes functions for finding suitable locations, generating settlement
names, and ensuring settlements do not overlap while respecting terrain
constraints and density parameters.
"""

import logging
import random

from onymancer import generate

from .map_data import MapData, Position, Settlement

logger = logging.getLogger(__name__)


def generate_settlements(
    map_data: MapData,
    settlement_density: float = 0.002,
    min_radius: float = 0.5,
    max_radius: float = 1.0,
) -> None:
    """
    Generate settlements on suitable terrain.

    Args:
        map_data (MapData):
            The terrain map grid.
        settlement_density (float):
            The probability of placing a settlement on suitable terrain.
        min_radius (float):
            The minimum radius of settlements.
        max_radius (float):
            The maximum radius of settlements.

    """
    # Validate input parameters
    if not (0.0 <= settlement_density <= 1.0):
        raise ValueError(
            f"Settlement density must be between 0.0 and 1.0, got {settlement_density}"
        )
    if min_radius <= 0:
        raise ValueError(f"Minimum radius must be positive, got {min_radius}")
    if max_radius <= 0:
        raise ValueError(f"Maximum radius must be positive, got {max_radius}")
    if min_radius > max_radius:
        raise ValueError(
            f"Minimum radius ({min_radius}) cannot be greater than maximum radius ({max_radius})"
        )

    logger.debug(f"Generating settlements with density {settlement_density}")

    # Find all suitable positions
    suitable_positions = _find_suitable_settlement_positions(map_data)

    # Place settlements at suitable positions
    _place_settlements_at_positions(
        map_data,
        suitable_positions,
        settlement_density,
        min_radius,
        max_radius,
    )

    # Designate harbors
    _designate_harbors(map_data)


def _generate_settlement_name() -> str:
    """Generate a random fantasy settlement name.

    Returns
        str:
            A randomly generated settlement name.

    """
    # Define various patterns for different name styles
    patterns: list[str] = [
        "sVs",  # Simple: Ael, Bor, etc.
        "sVsV",  # Longer: Aelon, Borin, etc.
        "VsV",  # Starting with vowel: Elor, Orin, etc.
        "sVsVs",  # Compound: Aelbor, Thorin, etc.
        "BVs",  # With consonant cluster: Bran, Thor, etc.
        "sVB",  # Ending with cluster: Aeld, Dorn, etc.
    ]

    # Randomly select a pattern
    pattern = random.choice(patterns)

    # Generate seed for reproducibility if needed, but use random for variety
    seed = random.randint(0, 1000000)

    name = generate(pattern, seed)

    # Capitalize first letter
    return name.capitalize()


def _is_position_suitable_for_settlement(
    map_data: MapData,
    position: Position,
) -> bool:
    """Check if a position is suitable for settlement placement.

    Args:
        map_data (MapData):
            The terrain map grid.
        position (Position):
            The position to check.

    Returns:
        bool:
            True if the position can have a settlement, False otherwise.

    Raises:
        ValueError: If position is out of bounds.

    """
    if not map_data.is_valid_position(position.x, position.y):
        raise ValueError(f"Position {position} is out of bounds")

    # Get the tile at the position.
    tile = map_data.get_terrain(position.x, position.y)
    # Check if the tile allows settlement building.
    return tile.can_build_settlement


def _does_settlement_overlaps(
    map_data: MapData,
    position: Position,
    min_radius: float,
    max_radius: float,
) -> bool:
    """
    Check if a potential settlement overlaps with existing settlements.

    Args:
        map_data (MapData):
            The terrain map grid.
        position (Position):
            The position of the potential settlement.
        min_radius (float):
            The radius of the potential settlement.
        max_radius (float):
            The maximum possible radius for settlements.

    Returns:
        bool:
            True if there is an overlap, False otherwise.

    Raises:
        ValueError:
            If radius or max_radius are negative.

    """
    if min_radius < 0:
        raise ValueError(f"Settlement radius must be non-negative, got {min_radius}")
    if max_radius < 0:
        raise ValueError(f"Maximum radius must be non-negative, got {max_radius}")
    # Check distance to each existing settlement.
    for existing in map_data.settlements:
        distance = position.distance_to(existing.position)
        if distance < pow(min_radius + max_radius, 6):
            return True
    return False


def _create_settlement(
    position: Position,
    min_radius: float,
    max_radius: float,
) -> Settlement:
    """Create a new settlement at the specified position.

    Args:
        position (Position):
            The position of the settlement.
        min_radius (float):
            The minimum radius for the settlement.
        max_radius (float):
            The maximum radius for the settlement.

    Returns:
        Settlement:
            The created settlement.

    Raises:
        ValueError: If min_radius > max_radius or if radii are negative.

    """
    if min_radius < 0 or max_radius < 0:
        raise ValueError(
            f"Radii must be non-negative, got min={min_radius}, max={max_radius}"
        )
    if min_radius > max_radius:
        raise ValueError(
            f"Minimum radius ({min_radius}) cannot be greater than maximum radius ({max_radius})"
        )

    # Generate a random radius within the range.
    radius = random.uniform(min_radius, max_radius)
    # Generate a name for the settlement.
    name = _generate_settlement_name()
    # Calculate connectivity based on radius.
    connectivity = int(radius * 5)

    return Settlement(
        name=name,
        position=position,
        radius=radius,
        connectivity=connectivity,
    )


def _find_suitable_settlement_positions(
    map_data: MapData,
) -> list[Position]:
    """Find all positions on the map that are suitable for settlement placement.

    Args:
        map_data (MapData):
            The terrain map grid.

    Returns:
        list[Position]:
            List of positions suitable for settlements.

    """
    suitable_positions = []

    # Iterate over all positions on the map.
    for y in range(map_data.height):
        for x in range(map_data.width):
            position = Position(x=x, y=y)
            if _is_position_suitable_for_settlement(map_data, position):
                suitable_positions.append(position)

    return suitable_positions


def _place_settlements_at_positions(
    map_data: MapData,
    positions: list[Position],
    settlement_density: float,
    min_radius: float,
    max_radius: float,
) -> None:
    """
    Place settlements at suitable positions with given constraints.

    Args:
        map_data (MapData):
            The terrain map grid.
        positions (list[Position]):
            List of positions to consider for settlement placement.
        settlement_density (float):
            The probability of placing a settlement at each position.
        min_radius (float):
            The minimum radius for settlements.
        max_radius (float):
            The maximum radius for settlements.

    """
    # Attempt to place a settlement at each suitable position.
    for position in positions:
        # Skip if random chance doesn't allow placement.
        if random.random() >= settlement_density:
            continue
        # Skip if settlement would overlap with existing ones.
        if _does_settlement_overlaps(map_data, position, min_radius, max_radius):
            continue
        # Add the new settlement to the map.
        map_data.settlements.append(
            _create_settlement(
                position,
                min_radius,
                max_radius,
            )
        )


def _designate_harbors(map_data: MapData) -> None:
    """
    Designate settlements as harbors by moving coastal ones closer to water
    or electing inland ones to become harbors.

    Args:
        map_data (MapData):
            The map data containing settlements and terrain.

    """
    # Find coastal settlements (those adjacent to water)
    coastal_settlements = []
    for settlement in map_data.settlements:
        pos = settlement.position
        neighbors = map_data.get_neighbors(pos.x, pos.y, include_diagonals=True)
        for neighbor in neighbors:
            if map_data.get_terrain(neighbor.x, neighbor.y).is_salt_water:
                coastal_settlements.append(settlement)
                break

    if coastal_settlements:
        # Move coastal settlements closer to the coast
        for settlement in coastal_settlements:
            _move_settlement_closer_to_coast(map_data, settlement)
            settlement.is_harbor = True
    else:
        # Elect some settlements to be harbors by moving them to the coast
        num_harbors = min(3, len(map_data.settlements) // 2)  # Up to 3 or half
        candidates = sorted(
            map_data.settlements, key=lambda s: s.connectivity, reverse=True
        )[:num_harbors]
        for settlement in candidates:
            _move_settlement_to_coast(map_data, settlement)
            settlement.is_harbor = True


def _move_settlement_closer_to_coast(map_data: MapData, settlement: Settlement) -> None:
    """
    Move a coastal settlement closer to the water if possible.

    Args:
        map_data (MapData):
            The map data.
        settlement (Settlement):
            The settlement to move.

    """
    pos = settlement.position
    # Find direction to nearest water
    nearest_water = None
    min_dist = float("inf")
    for y in range(map_data.height):
        for x in range(map_data.width):
            if map_data.get_terrain(x, y).is_salt_water:
                dist = ((x - pos.x) ** 2 + (y - pos.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_water = Position(x, y)

    if nearest_water:
        # Move towards the water by 1 tile
        dx = nearest_water.x - pos.x
        dy = nearest_water.y - pos.y
        dist = (dx**2 + dy**2) ** 0.5
        if dist > 0:
            new_x = pos.x + int(dx / dist)
            new_y = pos.y + int(dy / dist)
            new_pos = Position(new_x, new_y)
            if (
                map_data.is_valid_position(new_x, new_y)
                and map_data.get_terrain(new_x, new_y).can_build_settlement
                and not _does_settlement_overlaps(
                    map_data, new_pos, settlement.radius, settlement.radius
                )
            ):
                settlement.position = new_pos


def _move_settlement_to_coast(map_data: MapData, settlement: Settlement) -> None:
    """
    Move a settlement to the nearest coastal position.

    Args:
        map_data (MapData):
            The map data.
        settlement (Settlement):
            The settlement to move.

    """
    pos = settlement.position
    # Find nearest coastal position (buildable with water neighbor)
    nearest_coast = None
    min_dist = float("inf")
    for y in range(map_data.height):
        for x in range(map_data.width):
            if map_data.get_terrain(x, y).can_build_settlement:
                neighbors = map_data.get_neighbors(x, y, include_diagonals=True)
                has_water = any(
                    map_data.get_terrain(n.x, n.y).is_salt_water for n in neighbors
                )
                if has_water:
                    dist = ((x - pos.x) ** 2 + (y - pos.y) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_coast = Position(x, y)

    if nearest_coast:
        settlement.position = nearest_coast
