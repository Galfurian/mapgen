"""Settlement generation module."""

import random

from onymancer import generate

from . import logger
from .map_data import MapData, Position, Settlement


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

    Raises:
        ValueError: If map_data dimensions don't match noise_map dimensions.

    """
    logger.debug(f"Generating settlements with density {settlement_density}")

    # Find all suitable positions
    suitable_positions = _find_suitable_settlement_positions(map_data)

    # Place settlements at suitable positions
    settlements = _place_settlements_at_positions(
        map_data,
        suitable_positions,
        settlement_density,
        min_radius,
        max_radius,
    )

    return settlements


def _generate_settlement_name() -> str:
    """Generate a random fantasy settlement name.

    Returns:
        str: A randomly generated settlement name.

    """
    # Define various patterns for different name styles
    patterns = [
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
        map_data: The terrain map grid.
        position: The position to check.

    Returns:
        bool: True if the position can have a settlement, False otherwise.

    """
    if not map_data.is_valid_position(position.x, position.y):
        raise ValueError(f"Position {position} is out of bounds")

    tile = map_data.get_terrain(position.x, position.y)
    return tile.can_build_settlement


def _should_place_settlement(settlement_density: float) -> bool:
    """Determine if a settlement should be placed based on probability.

    Args:
        settlement_density: The probability of placing a settlement (0.0 to 1.0).

    Returns:
        bool: True if a settlement should be placed.

    Raises:
        ValueError: If settlement_density is not between 0.0 and 1.0.

    """
    if not (0.0 <= settlement_density <= 1.0):
        raise ValueError(
            f"Settlement density must be between 0.0 and 1.0, got {settlement_density}"
        )

    return random.random() < settlement_density


def _does_settlement_overlaps(
    map_data: MapData,
    position: Position,
    min_radius: float,
    max_radius: float,
) -> bool:
    """
    Check if a potential settlement overlaps with existing settlements.

    Args:
        map_data:
            The terrain map grid.
        position:
            The position of the potential settlement.
        min_radius:
            The radius of the potential settlement.
        max_radius:
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
    for existing in map_data.settlements:
        distance = position.distance_to(existing.position)
        if distance < min_radius + max_radius:
            return True
    return False


def _create_settlement(
    position: Position,
    min_radius: float,
    max_radius: float,
) -> Settlement:
    """Create a new settlement at the specified position.

    Args:
        position: The position of the settlement.
        min_radius: The minimum radius for the settlement.
        max_radius: The maximum radius for the settlement.

    Returns:
        Settlement: The created settlement.

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

    radius = random.uniform(min_radius, max_radius)
    name = _generate_settlement_name()
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
        map_data: The terrain map grid.

    Returns:
        List[Position]: List of positions suitable for settlements.

    """
    suitable_positions = []

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
        map_data:
            The terrain map grid.
        positions:
            List of positions to consider for settlement placement.
        settlement_density:
            The probability of placing a settlement at each position.
        min_radius:
            The minimum radius for settlements.
        max_radius:
            The maximum radius for settlements.

    """
    for position in positions:
        if not _should_place_settlement(settlement_density):
            continue
        if _does_settlement_overlaps(map_data, position, min_radius, max_radius):
            continue
        map_data.settlements.append(
            _create_settlement(
                position,
                min_radius,
                max_radius,
            )
        )
