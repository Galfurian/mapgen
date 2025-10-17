"""Settlement generation module."""

import random

import numpy as np
from onymancer import generate

from .map_data import MapData, Settlement


def generate_settlement_name() -> str:
    """Generate a random fantasy settlement name.

    Returns:
        str: A randomly generated settlement name.

    """
    # Define various patterns for different name styles
    patterns = [
        "sVs",      # Simple: Ael, Bor, etc.
        "sVsV",     # Longer: Aelon, Borin, etc.
        "VsV",      # Starting with vowel: Elor, Orin, etc.
        "sVsVs",    # Compound: Aelbor, Thorin, etc.
        "BVs",      # With consonant cluster: Bran, Thor, etc.
        "sVB",      # Ending with cluster: Aeld, Dorn, etc.
    ]

    # Randomly select a pattern
    pattern = random.choice(patterns)

    # Generate seed for reproducibility if needed, but use random for variety
    seed = random.randint(0, 1000000)

    name = generate(pattern, seed)

    # Capitalize first letter
    return name.capitalize()


def generate_settlements(
    map_data: MapData,
    noise_map: np.ndarray,
    settlement_density: float = 0.002,
    min_radius: float = 0.5,
    max_radius: float = 1.0,
) -> list[Settlement]:
    """Generate settlements on suitable terrain.

    Args:
        map_data (MapData): The terrain map grid.
        noise_map (np.ndarray): The noise map array.
        settlement_density (float): The probability of placing a settlement on suitable terrain.
        min_radius (float): The minimum radius of settlements.
        max_radius (float): The maximum radius of settlements.

    Returns:
        List[Settlement]: A list of generated settlements.

    """
    height = map_data.height
    width = map_data.width

    settlements: list[Settlement] = []
    for y in range(height):
        for x in range(width):
            tile = map_data.get_terrain(x, y)
            if tile.can_build_settlement:
                if random.random() < settlement_density:
                    # Check for overlaps
                    is_overlapping = False
                    for existing in settlements:
                        distance = (
                            (x - existing.x) ** 2
                            + (y - existing.y) ** 2
                        ) ** 0.5
                        if distance < existing.radius + max_radius:
                            is_overlapping = True
                            break

                    if not is_overlapping:
                        radius = random.uniform(min_radius, max_radius)
                        name = generate_settlement_name()
                        settlements.append(
                            Settlement(
                                x=x,
                                y=y,
                                radius=radius,
                                name=name,
                                connectivity=int(radius * 5),
                            )
                        )
    return settlements
