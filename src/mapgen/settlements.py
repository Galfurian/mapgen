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
    -------
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
    Designate settlements as harbors by selecting those near salt water
    and moving them to coastal positions.

    Args:
        map_data (MapData):
            The map data containing settlements and terrain.

    """
    logger.debug(
        f"Starting harbor designation for {len(map_data.settlements)} settlements"
    )
    # Find settlements near salt water, sorted by distance
    settlements_with_distance = []
    for settlement in map_data.settlements:
        distance = _get_distance_to_salt_water(map_data, settlement.position)
        if distance is not None:  # Only include settlements that can reach salt water
            settlements_with_distance.append((settlement, distance))
            logger.debug(
                f"  {settlement.name} at ({settlement.position.x}, {settlement.position.y}): distance to salt water = {distance:.1f}"
            )

    logger.debug(f"Found {len(settlements_with_distance)} settlements near salt water")

    # Sort by distance (closest first)
    settlements_with_distance.sort(key=lambda x: x[1])

    # Select harbors: scale based on total settlements and map size
    # More settlements = more harbors, larger maps = more harbors
    map_area = map_data.width * map_data.height
    base_harbors = max(1, len(map_data.settlements) // 4)  # ~1 harbor per 4 settlements
    area_bonus = max(0, (map_area // 5000) - 1)  # Bonus for large maps
    num_harbors = min(base_harbors + area_bonus, len(settlements_with_distance))

    logger.debug(f"Map size: {map_data.width}x{map_data.height} ({map_area} tiles)")
    logger.debug(f"Total settlements: {len(map_data.settlements)}")
    logger.debug(
        f"Selecting {num_harbors} harbors (base: {base_harbors}, area bonus: {area_bonus})"
    )

    selected_settlements = [s for s, _ in settlements_with_distance[:num_harbors]]

    logger.debug(
        f"Selected {len(selected_settlements)} harbors: {[s.name for s in selected_settlements]}"
    )

    # Move selected settlements to the coast and mark as harbors
    for settlement in selected_settlements:
        logger.debug(f"Moving {settlement.name} to coast")
        _move_settlement_to_coast(map_data, settlement)
        settlement.is_harbor = True
        logger.debug(f"  Moved to ({settlement.position.x}, {settlement.position.y})")


def _get_distance_to_salt_water(map_data: MapData, position: Position) -> float | None:
    """
    Get the distance from a position to the nearest salt water.

    Args:
        map_data (MapData):
            The map data.
        position (Position):
            The position to check.

    Returns:
        float | None:
            The distance to nearest salt water, or None if no salt water exists.

    """
    min_dist = float("inf")
    found_water = False

    for y in range(map_data.height):
        for x in range(map_data.width):
            if map_data.get_terrain(x, y).is_salt_water:
                found_water = True
                dist = ((x - position.x) ** 2 + (y - position.y) ** 2) ** 0.5
                min_dist = min(min_dist, dist)

    return min_dist if found_water else None


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
