"""Settlement generation module."""

import random

import numpy as np

from .map import Map, Settlement


def generate_settlement_name() -> str:
    """Generate a random fantasy settlement name.

    Returns:
        str: A randomly generated settlement name.

    """
    prefixes = [
        "North",
        "South",
        "East",
        "West",
        "New",
        "Old",
        "Bright",
        "Dark",
        "Green",
        "Silver",
        "Golden",
        "Whispering",
        "Silent",
        "Hidden",
        "Sunlit",
        "Starlit",
        "Crimson",
        "Azure",
        "Iron",
        "Stone",
        "Frost",
        "Ember",
    ]
    suffixes = [
        "wood",
        "haven",
        "gate",
        "ville",
        "burgh",
        "ton",
        "ford",
        "brook",
        "glen",
        "hollow",
        "vale",
        "Reach",
        "Hold",
        "Keep",
        "Crest",
        "Crag",
        "Falls",
        "Ridge",
        "Spire",
        "Moor",
        "Fells",
        "Run",
        "Pass",
        "Wold",
        "Downs",
    ]
    middles = [
        "High",
        "Low",
        "Deep",
        "Red",
        "White",
        "Black",
        "Gray",
        "Green",
        "Silver",
        "Golden",
        "Blue",
        "Swift",
        "Cold",
        "Burning",
        "Silent",
        "Ancient",
    ]

    name_parts = [random.choice(prefixes)]
    if random.random() < 0.5:
        name_parts.append(random.choice(middles))
    name_parts.append(random.choice(suffixes))
    return " ".join(name_parts)


def generate_settlements(
    map: Map,
    noise_map: np.ndarray,
    settlement_density: float = 0.002,
    min_radius: float = 0.5,
    max_radius: float = 1.0,
) -> list[Settlement]:
    """Generate settlements on suitable terrain.

    Args:
        map (Map): The terrain map grid.
        noise_map (np.ndarray): The noise map array.
        settlement_density (float): The probability of placing a settlement on suitable terrain.
        min_radius (float): The minimum radius of settlements.
        max_radius (float): The maximum radius of settlements.

    Returns:
        List[Settlement]: A list of generated settlements.

    """
    height = map.height
    width = map.width

    settlements: list[Settlement] = []
    for y in range(height):
        for x in range(width):
            tile = map.get_terrain(x, y)
            if tile.can_build_settlement:  # Uses habitability > 0.5
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
