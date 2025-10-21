"""Visualization module for maps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

from .map_data import MapData, Position


def _apply_curves_to_path(
    path: list[Position],
    elevation_map: np.ndarray,
    num_control_points: int = 5,
    smoothing_factor: float = 0.5,
) -> list[Position]:
    """
    Apply curves to a path based on elevation.

    Args:
        path (List[Position]):
            The path to curve.
        elevation_map (np.ndarray):
            The elevation map.
        num_control_points (int):
            Number of control points.
        smoothing_factor (float):
            Smoothing factor.

    Returns:
        List[Position]:
            The curved path.

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
        ax (plt.Axes):
            The matplotlib axes to plot on.
        map_data (MapData):
            The map data containing terrain and elevation.

    """
    rgb_values = np.zeros((map_data.height, map_data.width, 3))

    for y in range(map_data.height):
        for x in range(map_data.width):
            tile = map_data.get_terrain(x, y)
            # Compute shade factor based on elevation.
            shade_factor = 0.5 + 0.5 * map_data.get_elevation(x, y)
            # Generate shaded color.
            shaded_color = tuple(c * shade_factor for c in tile.color)
            # Replace with shaded color.
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
        ax (plt.Axes):
            The matplotlib axes to plot on.
        map_data (MapData):
            The map data containing elevation.
        levels (int):
            Number of contour levels.
        colors (str):
            Color of the contour lines.
        linewidths (float):
            Width of the contour lines.

    """
    X, Y = np.meshgrid(
        np.arange(map_data.width),
        np.arange(map_data.height),
    )
    Z = np.array(map_data.elevation_map)
    ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        colors=colors,
        linewidths=linewidths,
    )


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
        ax (plt.Axes):
            The matplotlib axes to plot on.
        map_data (MapData):
            The map data containing roads.
        color (str):
            Color of the roads.
        linewidth (float):
            Width of the road lines.
        zorder (int):
            Z-order for layering.

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
        ax (plt.Axes):
            The matplotlib axes to plot on.
        map_data (MapData):
            The map data containing settlements.
        zorder (int):
            Z-order for layering.

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
    Plot the complete map with all layers: terrain, contours, roads, and
    settlements.

    Args:
        map_data (MapData):
            The map data to visualize.

    Returns:
        Figure:
            The matplotlib figure containing the complete map.

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


def plot_3d_map(
    map_data: MapData,
    enable_settlements: bool = True,
    enable_roads: bool = True,
    enable_legend: bool = False,
    colormap: str = "terrain",
    elevation_scale: float = 1.0,
) -> Figure:
    """
    Create a stunning 3D visualization of the map with elevation.

    This function generates an interactive 3D terrain map showing elevation,
    settlements as markers, and roads as lines in 3D space.

    Args:
        map_data (MapData):
            The map data to visualize.
        enable_settlements (bool):
            Whether to show settlements as 3D markers.
        enable_roads (bool):
            Whether to show roads as 3D lines.
        enable_legend (bool):
            Whether to show the legend (default: False for cleaner 3D view).
        colormap (str):
            Matplotlib colormap for terrain coloring (e.g., 'terrain', 'viridis').
        elevation_scale (float):
            Scale factor for elevation exaggeration (1.0 = realistic).

    Returns:
        Figure:
            The matplotlib figure containing the 3D map.

    """
    # Create 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # Convert elevation map to numpy array
    if not map_data.elevation_map:
        raise ValueError(
            "Map data missing elevation_map - cannot create 3D visualization"
        )

    elevation_array = np.array(map_data.elevation_map)
    height, width = elevation_array.shape

    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Scale elevation for better visualization
    scaled_elevation = elevation_array * elevation_scale

    # Calculate maximum elevation for positioning settlements above terrain
    max_elevation = np.max(scaled_elevation)
    settlement_height = max_elevation + (
        elevation_scale * 0.5
    )  # Position settlements well above terrain
    label_height = max_elevation + (
        elevation_scale * 0.7
    )  # Position labels even higher

    # Plot the 3D terrain surface
    surf = ax.plot_surface(
        x_coords,
        y_coords,
        scaled_elevation,
        cmap=colormap,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
    )

    # Add settlements as 3D markers
    if enable_settlements and map_data.settlements:
        settlement_x = [s.position.x for s in map_data.settlements]
        settlement_y = [s.position.y for s in map_data.settlements]
        # Fixed height above terrain.
        settlement_z = [settlement_height] * len(map_data.settlements)
        settlement_sizes = [
            s.radius * 20 for s in map_data.settlements
        ]  # Scale for visibility

        # Add vertical lines from terrain to settlements for better visibility
        for settlement in map_data.settlements:
            terrain_z = scaled_elevation[settlement.position.y, settlement.position.x]
            ax.plot(
                [settlement.position.x, settlement.position.x],
                [settlement.position.y, settlement.position.y],
                [terrain_z, settlement_height],
                color="red",
                linewidth=1,
                alpha=0.5,
                linestyle="--",
            )

        ax.scatter(
            xs=settlement_x,
            ys=settlement_y,
            zs=settlement_z,
            c="red",
            s=settlement_sizes,
            alpha=0.9,
            edgecolors="darkred",
            linewidth=1,
            label="Settlements",
        )

        # Add settlement labels at fixed height above settlements
        for settlement in map_data.settlements:
            ax.text(
                settlement.position.x,
                settlement.position.y,
                label_height,
                settlement.name,
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
            )

    # Add roads as 3D lines
    if enable_roads and map_data.roads:
        for road in map_data.roads:
            if len(road.path) > 1:
                road_x = [pos.x for pos in road.path]
                road_y = [pos.y for pos in road.path]
                road_z = [
                    scaled_elevation[pos.y, pos.x] + 0.05 for pos in road.path
                ]  # Slightly above terrain

                ax.plot(
                    road_x,
                    road_y,
                    road_z,
                    color="brown",
                    linewidth=2,
                    alpha=0.7,
                    label=(
                        "Roads" if road == map_data.roads[0] else ""
                    ),  # Only label first road
                )

    # Add roads as 3D lines
    if enable_roads and map_data.roads:
        for road in map_data.roads:
            if len(road.path) > 1:
                road_x = [pos.x for pos in road.path]
                road_y = [pos.y for pos in road.path]
                road_z = [
                    scaled_elevation[pos.y, pos.x] + 0.05 for pos in road.path
                ]  # Slightly above terrain

                ax.plot(
                    road_x,
                    road_y,
                    road_z,
                    color="brown",
                    linewidth=2,
                    alpha=0.7,
                    label=(
                        "Roads" if road == map_data.roads[0] else ""
                    ),  # Only label first road
                )

    # Configure the 3D view
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Elevation")
    ax.set_title("3D Fantasy Map Visualization")

    # Set equal aspect ratio and nice viewing angle
    ax.set_box_aspect([width / height, 1, 0.3])  # Compress Z axis for better viewing
    ax.view_init(elev=30, azim=45)  # Nice isometric view

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation")

    # Add legend if we have elements and legend is enabled
    if enable_legend and (
        (enable_settlements and map_data.settlements)
        or (enable_roads and map_data.roads)
    ):
        ax.legend()

    plt.tight_layout()
    return fig


def get_ascii_map(map_data: MapData) -> str:
    """
    Generate an ASCII representation of the map.

    Args:
        map_data (MapData):
            The map data to visualize.

    Returns:
        str:
            The ASCII map as a string.

    """
    lines = []
    for y in range(map_data.height):
        line = ""
        for x in range(map_data.width):
            tile = map_data.get_terrain(x, y)
            line += tile.symbol
        lines.append(line)
    return "\n".join(lines)


def plot_elevation_map(
    map_data: MapData,
    colormap: str = "terrain",
    title: str = "Elevation Map",
) -> Figure:
    """
    Plot the elevation map as a standalone visualization.

    Args:
        map_data (MapData):
            The map data containing elevation.
        colormap (str):
            Matplotlib colormap for elevation coloring.
        title (str):
            Title for the plot.

    Returns:
        Figure:
            The matplotlib figure containing the elevation map.

    """
    if not map_data.elevation_map:
        raise ValueError("Map data missing elevation_map")

    fig, ax = plt.subplots(figsize=(10, 8))

    elevation_array = np.array(map_data.elevation_map)
    im = ax.imshow(elevation_array, cmap=colormap, origin="upper")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("Elevation")

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    return fig


def plot_rainfall_map(
    map_data: MapData,
    colormap: str = "Blues",
    title: str = "Rainfall Map",
) -> Figure:
    """
    Plot the rainfall map as a standalone visualization.

    Args:
        map_data (MapData):
            The map data containing rainfall.
        colormap (str):
            Matplotlib colormap for rainfall coloring.
        title (str):
            Title for the plot.

    Returns:
        Figure:
            The matplotlib figure containing the rainfall map.

    """
    if not map_data.rainfall_map:
        raise ValueError("Map data missing rainfall_map")

    fig, ax = plt.subplots(figsize=(10, 8))

    rainfall_array = np.array(map_data.rainfall_map)
    im = ax.imshow(rainfall_array, cmap=colormap, origin="upper")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("Rainfall Intensity")

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    return fig
