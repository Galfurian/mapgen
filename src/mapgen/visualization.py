"""Visualization module for maps."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patches
from matplotlib.figure import Figure
from scipy.interpolate import interp1d

from .map_data import MapData, Position, RoadType, Settlement


def apply_curves_to_path(
    path: list[Position],
    elevation_map: np.ndarray,
    num_control_points: int = 5,
    smoothing_factor: float = 0.5,
) -> list[Position]:
    """Apply curves to a path based on elevation.

    Args:
        path (List[Position]): The path to curve.
        elevation_map (np.ndarray): The elevation map.
        num_control_points (int): Number of control points.
        smoothing_factor (float): Smoothing factor.

    Returns:
        List[Position]: The curved path.

    """
    if len(path) <= 2:
        return path

    x = np.array([p.x for p in path])
    y = np.array([p.y for p in path])
    distances = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    distances = np.insert(distances, 0, 0)

    total_distance = distances[-1]
    control_point_distances = np.linspace(0, total_distance, num=num_control_points)

    control_points_x = interp1d(distances, x, kind="linear")(
        control_point_distances
    ).astype(float)
    control_points_y = interp1d(distances, y, kind="linear")(
        control_point_distances
    ).astype(float)

    for i in range(1, num_control_points - 1):
        cx, cy = round(control_points_x[i]), round(control_points_y[i])
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

    f_x = interp1d(
        control_point_distances,
        control_points_x,
        kind="cubic",
    )
    f_y = interp1d(
        control_point_distances,
        control_points_y,
        kind="cubic",
    )

    new_path_distances = np.linspace(0, total_distance, num=len(path))
    new_x = f_x(new_path_distances)
    new_y = f_y(new_path_distances)
    curved_path = [
        Position(round(x), round(y)) for x, y in zip(new_x, new_y, strict=True)
    ]
    return curved_path


def is_coastal(
    map_data: MapData,
    x: int,
    y: int,
    max_distance: int = 2,
) -> bool:
    """Check if a point is within distance from water.

    Args:
        map_data (MapData): The map grid.
        x (int): The x coordinate.
        y (int): The y coordinate.
        max_distance (int): Maximum distance to check.

    Returns:
        bool: True if coastal, False otherwise.

    """
    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            nx, ny = x + dx, y + dy
            tile = map_data.get_terrain(nx, ny)
            if not tile.can_build_road:
                return True
    return False


def plot_map(
    map_data: MapData,
    noise_map: np.ndarray,
    settlements: list[Settlement],
    roads_graph: nx.Graph,
    elevation_map: np.ndarray,
) -> Figure:
    """Plot the map with terrain, settlements, and roads.

    Args:
        map_data (MapData): The map grid.
        noise_map (np.ndarray): The noise map.
        settlements (list[Settlement]): List of settlements.
        roads_graph (nx.Graph): The roads graph.
        elevation_map (np.ndarray): The elevation map.

    Returns:
        Figure: The matplotlib figure.

    """
    height = map_data.height
    width = map_data.width

    rgb_values = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            tile = map_data.get_terrain(x, y)
            base_color = tile.color  # Uses tile's color property
            noise_value = noise_map[y, x]
            shade_factor = (noise_value + 1) / 2
            shaded_color = tuple(c * shade_factor for c in base_color)
            rgb_values[y, x, :] = shaded_color

    fig, ax = plt.subplots()
    ax.imshow(rgb_values)

    # Contour lines
    contour_levels = 10
    contour_colors = "k"
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    ax.contour(
        X,
        Y,
        noise_map,
        levels=contour_levels,
        colors=contour_colors,
        linewidths=0.5,
    )

    # Plot roads first (so settlements appear on top)
    for _, _, data in roads_graph.edges(data=True):
        if "path" in data:
            path = data["path"]
            x_coords = [pos.x for pos in path]
            y_coords = [pos.y for pos in path]
            if data.get("type") == RoadType.WATER:
                ax.plot(
                    x_coords,
                    y_coords,
                    color="gray",
                    linestyle="dotted",
                    linewidth=2,
                    zorder=1,
                )
            else:
                curved_path = apply_curves_to_path(path, elevation_map)
                curved_x = [pos.x for pos in curved_path]
                curved_y = [pos.y for pos in curved_path]
                ax.plot(curved_x, curved_y, color="gray", linewidth=2, zorder=1)

    # Plot settlements
    existing_texts: list[tuple[int, int]] = []
    for settlement in settlements:
        x = settlement.x
        y = settlement.y
        radius = settlement.radius

        circle = patches.Circle(
            (x, y),
            radius,
            facecolor="white",
            edgecolor="black",
            linewidth=1,
            zorder=3,
        )
        ax.add_patch(circle)

        font_size = int(radius * 6)  # Reduced from 10 to 6 for smaller text

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
                settlement.name,
                color="white",
                fontsize=font_size,
                rotation=0,
                bbox={
                    "facecolor": "gray",
                    "edgecolor": "none",
                    "alpha": 0.7,
                    "pad": 0.3,
                },
                ha="center",
                va="center",
                zorder=3,
            )
            existing_texts.append((text_x, text_y))

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    return fig
