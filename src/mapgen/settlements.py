"""Settlement generation module."""

import random
from typing import TypedDict


class Settlement(TypedDict):
    """Represents a settlement in the map.

    Attributes:
        x (int): The x-coordinate of the settlement.
        y (int): The y-coordinate of the settlement.
        radius (float): The radius of the settlement.
        name (str): The name of the settlement.
        connectivity (int): The connectivity value of the settlement.

    """

    x: int
    y: int
    radius: float
    name: str
    connectivity: int


import numpy as np


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
    level: list[list[str]],
    noise_map: np.ndarray,
    settlement_density: float = 0.002,
    min_radius: float = 0.5,
    max_radius: float = 1.0,
) -> list[dict[str, int | float | str]]:
    """Generate settlements on suitable terrain.

    Args:
        level (List[List[str]]): The terrain level grid.
        noise_map (np.ndarray): The noise map array.
        settlement_density (float): The probability of placing a settlement on suitable terrain.
        min_radius (float): The minimum radius of settlements.
        max_radius (float): The maximum radius of settlements.

    Returns:
        List[Settlement]: A list of generated settlements.

    """
    height = len(level)
    width = len(level[0])

    settlements: list[dict[str, int | float | str]] = []
    for y in range(height):
        for x in range(width):
            if level[y][x] in ("P", "F"):  # Plains or Forest
                if random.random() < settlement_density:
                    # Check for overlaps
                    is_overlapping = False
                    for existing in settlements:
                        distance = (
                            (x - int(existing["x"])) ** 2
                            + (y - int(existing["y"])) ** 2
                        ) ** 0.5
                        if distance < float(existing["radius"]) + max_radius:
                            is_overlapping = True
                            break

                    if not is_overlapping:
                        radius = random.uniform(min_radius, max_radius)
                        name = generate_settlement_name()
                        settlements.append(
                            {
                                "x": x,
                                "y": y,
                                "radius": radius,
                                "name": name,
                                "connectivity": int(radius * 5),
                            }
                        )
    return settlements
