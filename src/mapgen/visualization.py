"""Visualization module for maps."""

from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from scipy.interpolate import interp1d


def apply_curves_to_path(
    path: List[Tuple[int, int]],
    elevation_map: np.ndarray,
    num_control_points: int = 5,
    smoothing_factor: float = 0.5,
) -> List[Tuple[int, int]]:
    """Apply curves to a path based on elevation.

    Args:
        path (List[Tuple[int, int]]): The path to curve.
        elevation_map (np.ndarray): The elevation map.
        num_control_points (int): Number of control points.
        smoothing_factor (float): Smoothing factor.

    Returns:
        List[Tuple[int, int]]: The curved path.
    """
    if len(path) <= 2:
        return path

    x = np.array([p[0] for p in path])
    y = np.array([p[1] for p in path])
    distances = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    distances = np.insert(distances, 0, 0)

    total_distance = distances[-1]
    control_point_distances = np.linspace(0, total_distance, num=num_control_points)

    control_points_x = interp1d(distances, x, kind="linear")(control_point_distances)
    control_points_y = interp1d(distances, y, kind="linear")(control_point_distances)

    for i in range(1, num_control_points - 1):
        cx, cy = int(round(control_points_x[i])), int(round(control_points_y[i]))
        left_x = max(0, cx - 1)
        right_x = min(elevation_map.shape[1] - 1, cx + 1)
        top_y = max(0, cy - 1)
        bottom_y = min(elevation_map.shape[0] - 1, cy + 1)

        gradient_x = (elevation_map[cy, right_x] - elevation_map[cy, left_x]) / 2
        gradient_y = (elevation_map[bottom_y, cx] - elevation_map[top_y, cx]) / 2

        elevation_diff = abs(
            elevation_map[cy, cx]
            - elevation_map[int(control_points_y[i + 1]), int(control_points_x[i + 1])]
        )
        adjusted_smoothing_factor = smoothing_factor * (1 + elevation_diff)

        control_points_x[i] += adjusted_smoothing_factor * gradient_x
        control_points_y[i] += adjusted_smoothing_factor * gradient_y

    f_x = interp1d(control_point_distances, control_points_x, kind="cubic")
    f_y = interp1d(control_point_distances, control_points_y, kind="cubic")

    new_path_distances = np.linspace(0, total_distance, num=len(path))
    new_x = f_x(new_path_distances)
    new_y = f_y(new_path_distances)
    curved_path = [(int(round(x)), int(round(y))) for x, y in zip(new_x, new_y)]
    return curved_path


def is_coastal(level: List[List[str]], x: int, y: int, max_distance: int = 2) -> bool:
    """Check if a point is within distance from water.

    Args:
        level (List[List[str]]): The level grid.
        x (int): The x coordinate.
        y (int): The y coordinate.
        max_distance (int): Maximum distance to check.

    Returns:
        bool: True if coastal, False otherwise.
    """
    height = len(level)
    width = len(level[0])

    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and level[ny][nx] == "W":
                return True
    return False


def plot_level(
    level: List[List[str]],
    noise_map: np.ndarray,
    settlements: List[Dict],
    roads_graph: nx.Graph,
    elevation_map: np.ndarray,
) -> Figure:
    """Plot the level with terrain, settlements, and roads.

    Args:
        level (List[List[str]]): The level grid.
        noise_map (np.ndarray): The noise map.
        settlements (List[Dict]): List of settlements.
        roads_graph (nx.Graph): The roads graph.
        elevation_map (np.ndarray): The elevation map.

    Returns:
        Figure: The matplotlib figure.
    """
    height = len(level)
    width = len(level[0])

    color_map = {
        "#": (0.2, 0.2, 0.2),  # Wall
        " ": (1.0, 1.0, 1.0),  # Empty
        "W": (0.2, 0.5, 1.0),  # Water
        "M": (0.5, 0.5, 0.5),  # Mountain
        "F": (0.2, 0.7, 0.2),  # Forest
        "P": (0.9, 0.8, 0.6),  # Plains
    }

    rgb_values = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            terrain_type = level[y][x]
            base_color = color_map.get(terrain_type, (1.0, 1.0, 1.0))
            noise_value = noise_map[y, x]
            shade_factor = (noise_value + 1) / 2
            shaded_color = tuple(c * shade_factor for c in base_color)
            rgb_values[y, x, :] = shaded_color

    fig, ax = plt.subplots()
    im = ax.imshow(rgb_values)

    # Contour lines
    contour_levels = 10
    contour_colors = "k"
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    ax.contour(
        X, Y, noise_map, levels=contour_levels, colors=contour_colors, linewidths=0.5
    )

    # Plot settlements
    existing_texts = []
    for settlement in settlements:
        x = settlement["x"]
        y = settlement["y"]
        radius = settlement["radius"]

        circle = patches.Circle(
            (x, y), radius, facecolor="white", edgecolor="black", linewidth=1
        )
        ax.add_patch(circle)

        font_size = int(radius * 10)

        possible_positions = [
            (x, y + 2),
            (x, y - 2),
            (x + 2, y),
            (x - 2, y),
            (x + 2, y + 2),
            (x + 2, y - 2),
            (x - 2, y + 2),
            (x - 2, y - 2),
        ]

        possible_positions.sort(
            key=lambda pos: ((pos[0] - x) ** 2 + (pos[1] - y) ** 2) ** 0.5
        )

        text_x, text_y = None, None
        for pos in possible_positions:
            if (
                ((pos[0] - x) ** 2 + (pos[1] - y) ** 2) ** 0.5 > radius
                and 0 <= pos[0] < width
                and 0 <= pos[1] < height
            ):
                is_overlapping = False
                for existing_text_x, existing_text_y in existing_texts:
                    if (
                        abs(pos[0] - existing_text_x) < font_size
                        and abs(pos[1] - existing_text_y) < font_size
                    ):
                        is_overlapping = True
                        break

                if not is_overlapping:
                    text_x, text_y = pos
                    break

        if text_x is not None and text_y is not None:
            ax.text(
                text_x,
                text_y,
                settlement["name"],
                color="white",
                fontsize=font_size,
                rotation=90,
                bbox={
                    "facecolor": "black",
                    "edgecolor": "black",
                    "alpha": 1,
                    "pad": 0.5,
                },
                ha="center",
                va="center",
            )
            existing_texts.append((text_x, text_y))

    # Plot roads
    for u, v, data in roads_graph.edges(data=True):
        if "path" in data:
            path = data["path"]
            x_coords, y_coords = zip(*path)
            if data["type"] == "water":
                ax.plot(
                    x_coords, y_coords, color="gray", linestyle="dotted", linewidth=2
                )
            else:
                curved_path = apply_curves_to_path(path, elevation_map)
                ax.plot(*zip(*curved_path), color="gray", linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    return fig
