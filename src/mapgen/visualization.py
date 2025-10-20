"""Visualization module for maps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d

from .map_data import MapData, Position


def _apply_curves_to_path(
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
        Position(x=round(x), y=round(y)) for x, y in zip(new_x, new_y, strict=True)
    ]
    return curved_path


def plot_base_terrain(ax: Axes, map_data: MapData) -> None:
    """
    Plot the base terrain layer with elevation-based shading.

    Args:
        ax (plt.Axes): The matplotlib axes to plot on.
        map_data (MapData): The map data containing terrain and elevation.

    """
    elevation_map = np.array(map_data.elevation_map)
    rgb_values = np.zeros((map_data.height, map_data.width, 3))

    for y in range(map_data.height):
        for x in range(map_data.width):
            tile = map_data.get_terrain(x, y)
            base_color = tile.color
            elevation = elevation_map[y, x]
            shade_factor = (elevation + 1) / 2
            shaded_color = tuple(c * shade_factor for c in base_color)
            rgb_values[y, x, :] = shaded_color

    ax.imshow(rgb_values)


def plot_contour_lines(
    ax: Axes,
    map_data: MapData,
    levels: int = 10,
    colors: str = "k",
    linewidths: float = 0.5,
) -> None:
    """
    Plot elevation contour lines on the map.

    Args:
        ax (plt.Axes): The matplotlib axes to plot on.
        map_data (MapData): The map data containing elevation.
        levels (int): Number of contour levels.
        colors (str): Color of the contour lines.
        linewidths (float): Width of the contour lines.

    """
    elevation_map = np.array(map_data.elevation_map)
    X, Y = np.meshgrid(np.arange(map_data.width), np.arange(map_data.height))
    ax.contour(X, Y, elevation_map, levels=levels, colors=colors, linewidths=linewidths)


def plot_roads(
    ax: Axes,
    map_data: MapData,
    color: str = "brown",
    linewidth: float = 2,
    zorder: int = 1,
) -> None:
    """
    Plot roads on the map.

    Args:
        ax (plt.Axes): The matplotlib axes to plot on.
        map_data (MapData): The map data containing roads.
        color (str): Color of the roads.
        linewidth (float): Width of the road lines.
        zorder (int): Z-order for layering.

    """
    elevation_map = np.array(map_data.elevation_map)
    for road in map_data.roads:
        path = road.path
        curved_path = _apply_curves_to_path(path, elevation_map)
        curved_x = [pos.x for pos in curved_path]
        curved_y = [pos.y for pos in curved_path]
        ax.plot(curved_x, curved_y, color=color, linewidth=linewidth, zorder=zorder)


def plot_settlements(ax: Axes, map_data: MapData, zorder: int = 3) -> None:
    """
    Plot settlements with circles and labels on the map.

    Args:
        ax (plt.Axes): The matplotlib axes to plot on.
        map_data (MapData): The map data containing settlements.
        zorder (int): Z-order for layering.

    """
    existing_texts: list[tuple[int, int]] = []
    for settlement in map_data.settlements:
        x = settlement.position.x
        y = settlement.position.y
        radius = settlement.radius

        circle = patches.Circle(
            (x, y),
            radius,
            facecolor="white",
            edgecolor="black",
            linewidth=1,
            zorder=zorder,
        )
        ax.add_patch(circle)

        font_size = int(radius * 6)

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
                and 0 <= pos[0] < map_data.width
                and 0 <= pos[1] < map_data.height
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
                zorder=zorder,
            )
            existing_texts.append((text_x, text_y))


def plot_map(
    map_data: MapData,
    enable_contours: bool = True,
    enable_roads: bool = True,
    enable_settlements: bool = True,
) -> Figure:
    """
    Plot the complete map with all layers: terrain, contours, roads, and settlements.

    Args:
        map_data (MapData): The map data to visualize.

    Returns:
        Figure: The matplotlib figure containing the complete map.

    """
    fig, ax = plt.subplots()

    # Plot each layer in order
    plot_base_terrain(ax, map_data)
    if enable_contours:
        plot_contour_lines(ax, map_data)
    if enable_roads:
        plot_roads(ax, map_data)
    if enable_settlements:
        plot_settlements(ax, map_data)

    # Configure axes
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    return fig
