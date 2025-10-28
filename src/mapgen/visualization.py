"""
Visualization utilities for the map generator.

This module provides functions for creating various visualizations of generated
maps, including 2D plots, 3D terrain views, and ASCII representations of
different map layers like elevation, rainfall, and temperature.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.figure import Figure

from .map_data import MapData

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


def plot_map(
    map_data: MapData,
    enable_contours: bool = True,
    enable_roads: bool = True,
    enable_settlements: bool = True,
    enable_water_routes: bool = True,
) -> Figure:
    """
    Plot the complete map with all layers: terrain, contours, roads, and
    settlements.

    Args:
        map_data (MapData):
            The map data to visualize.
        enable_contours (bool):
            Whether to show elevation contours.
        enable_roads (bool):
            Whether to show roads.
        enable_settlements (bool):
            Whether to show settlements.
        enable_water_routes (bool):
            Whether to show water routes.

    Returns:
        Figure:
            The matplotlib figure containing the complete map.

    """
    fig, ax = plt.subplots()

    # Plot base terrain layer with elevation-based shading
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

    # Plot elevation contour lines
    if enable_contours:
        X, Y = np.meshgrid(
            np.arange(map_data.width),
            np.arange(map_data.height),
        )
        Z = np.array(map_data.elevation_map)
        ax.contour(
            X,
            Y,
            Z,
            levels=10,
            colors="k",
            linewidths=0.5,
        )

    # Plot roads
    if enable_roads:
        for road in map_data.roads:
            # Get original path coordinates.
            x_coords = [pos.x for pos in road.path]
            y_coords = [pos.y for pos in road.path]
            # Plot the curved road.
            ax.plot(
                x_coords,
                y_coords,
                linewidth=1,
                color="brown",
                linestyle="-",
                zorder=1,
            )
            ax.scatter(
                x_coords,
                y_coords,
                color="brown",
                s=1,
                zorder=2,
            )

    # Plot water routes
    if enable_water_routes:
        for water_route in map_data.water_routes:
            # Get original path coordinates.
            x_coords = [pos.x for pos in water_route.path]
            y_coords = [pos.y for pos in water_route.path]
            ax.plot(
                x_coords,
                y_coords,
                linewidth=1,
                color="blue",
                linestyle="-",
                zorder=1,
            )
            ax.scatter(
                x_coords,
                y_coords,
                color="blue",
                s=1,
                zorder=2,
            )

    # Plot settlements with circles and labels
    if enable_settlements:
        existing_texts: list[tuple[int, int]] = []
        for settlement in map_data.settlements:
            x = settlement.position.x
            y = settlement.position.y
            radius = settlement.radius

            circle = patches.Circle(
                (x, y),
                radius,
                facecolor="blue" if settlement.is_harbor else "white",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
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
                distance = ((pos[0] - x) ** 2 + (pos[1] - y) ** 2) ** 0.5
                if (
                    distance > radius
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
                    zorder=3,
                )
                existing_texts.append((text_x, text_y))

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
            Matplotlib colormap for terrain coloring (e.g., 'terrain',
            'viridis').
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
        road_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "gray",
        ]
        for i, road in enumerate(map_data.roads):
            if len(road.path) > 1:
                road_x = [pos.x for pos in road.path]
                road_y = [pos.y for pos in road.path]
                road_z = [
                    scaled_elevation[pos.y, pos.x] + 0.05 for pos in road.path
                ]  # Slightly above terrain

                color = road_colors[i % len(road_colors)]
                ax.plot(
                    road_x,
                    road_y,
                    road_z,
                    color=color,
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


def plot_temperature_map(
    map_data: MapData,
    colormap: str = "RdYlBu_r",
    title: str = "Temperature Map",
) -> Figure:
    """
    Plot the temperature map as a standalone visualization.

    Args:
        map_data (MapData):
            The map data containing temperature.
        colormap (str):
            Matplotlib colormap for temperature coloring (default: RdYlBu_r for
            hot=red, cold=blue).
        title (str):
            Title for the plot.

    Returns:
        Figure:
            The matplotlib figure containing the temperature map.

    """
    if not map_data.temperature_map:
        raise ValueError("Map data missing temperature_map")

    fig, ax = plt.subplots(figsize=(10, 8))

    temperature_array = np.array(map_data.temperature_map)
    im = ax.imshow(temperature_array, cmap=colormap, origin="upper")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("Temperature")

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    return fig


def plot_accumulation_map(
    map_data: MapData,
    colormap: str = "Blues",
    title: str = "Water Accumulation Map",
) -> Figure:
    """
    Plot the water accumulation map as a standalone visualization.

    This shows areas where water tends to accumulate based on terrain and
    rainfall patterns. Higher values indicate areas prone to water accumulation
    (valleys, low-lying areas), while lower values show areas where water
    flows through quickly.

    Args:
        map_data (MapData):
            The map data containing accumulation information.
        colormap (str):
            Matplotlib colormap for accumulation coloring (default: Blues).
        title (str):
            Title for the plot.

    Returns:
        Figure:
            The matplotlib figure containing the accumulation map.

    """
    if not map_data.accumulation_map:
        raise ValueError("Map data missing accumulation_map")

    fig, ax = plt.subplots(figsize=(10, 8))

    accumulation_array = np.array(map_data.accumulation_map)
    im = ax.imshow(accumulation_array, cmap=colormap, origin="upper")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("Water Accumulation")

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xticks([])
    ax.set_yticks([])

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


def get_ascii_elevation_map(map_data: MapData) -> str:
    """
    Generate an ASCII representation of the elevation map using digits 0-9.

    Elevation values are normalized to a 0-9 scale where:
        - 0 represents lowest elevations (deepest ocean)
        - 9 represents highest elevations (mountain peaks)

    This is useful for debugging and analyzing elevation patterns without
    requiring graphical output.

    Args:
        map_data (MapData):
            The map data containing the elevation map.

    Returns:
        str:
            The ASCII elevation map as a string with digits 0-9.

    Raises:
        ValueError:
            If elevation_map is not available in map_data.

    """
    if not map_data.elevation_map:
        raise ValueError("Elevation map is not available in map_data")

    elevation_array = np.array(map_data.elevation_map)
    # Normalize elevation from -1.0 to 1.0 range to 0-9 scale First shift from
    # [-1, 1] to [0, 2], then scale to [0, 9]
    elevation_normalized = ((elevation_array + 1.0) * 4.5).astype(int)
    elevation_normalized = np.clip(elevation_normalized, 0, 9)

    lines = []
    for y in range(map_data.height):
        line = ""
        for x in range(map_data.width):
            line += str(elevation_normalized[y, x])
        lines.append(line)
    return "\n".join(lines)


def get_ascii_rainfall_map(map_data: MapData) -> str:
    """
    Generate an ASCII representation of the rainfall map using digits 0-9.

    This visualization maps rainfall values to single digits where:
        - 0 represents very dry areas (no rain)
        - 9 represents very wet areas (maximum rainfall)

    Args:
        map_data (MapData):
            The map data containing rainfall information.

    Returns:
        str:
            The ASCII rainfall map as a string with digits 0-9.

    Raises:
        ValueError:
            If rainfall_map is not available in map_data.

    """
    if not map_data.rainfall_map:
        raise ValueError("Rainfall map is not available in map_data")

    rainfall_array = np.array(map_data.rainfall_map)
    # Normalize to 0-9 scale
    rainfall_normalized = (rainfall_array * 9).astype(int)
    rainfall_normalized = np.clip(rainfall_normalized, 0, 9)

    lines = []
    for y in range(map_data.height):
        line = ""
        for x in range(map_data.width):
            line += str(rainfall_normalized[y, x])
        lines.append(line)
    return "\n".join(lines)


def get_ascii_temperature_map(map_data: MapData) -> str:
    """
    Generate an ASCII representation of the temperature map using digits 0-9.

    Temperature values are normalized to a 0-9 scale where:
        - 0 represents coldest temperatures
        - 9 represents hottest temperatures

    This is useful for debugging and analyzing temperature patterns without
    requiring graphical output.

    Args:
        map_data (MapData):
            The map data containing the temperature map.

    Returns:
        str:
            The ASCII temperature map as a string with digits 0-9.

    Raises:
        ValueError:
            If temperature_map is not available in map_data.

    """
    if not map_data.temperature_map:
        raise ValueError("Temperature map is not available in map_data")

    temperature_array = np.array(map_data.temperature_map)
    # Normalize to 0-9 scale
    temperature_normalized = (temperature_array * 9).astype(int)
    temperature_normalized = np.clip(temperature_normalized, 0, 9)

    lines = []
    for y in range(map_data.height):
        line = ""
        for x in range(map_data.width):
            line += str(temperature_normalized[y, x])
        lines.append(line)
    return "\n".join(lines)


def get_ascii_accumulation_map(map_data: MapData) -> str:
    """
    Generate an ASCII representation of the water accumulation map using digits 0-9.

    Accumulation values are normalized to a 0-9 scale where:
        - 0 represents areas with minimal water accumulation
        - 9 represents areas with maximum water accumulation (valleys, wetlands)

    This visualization helps identify drainage patterns and areas prone to
    flooding or wetland formation.

    Args:
        map_data (MapData):
            The map data containing the accumulation map.

    Returns:
        str:
            The ASCII accumulation map as a string with digits 0-9.

    Raises:
        ValueError:
            If accumulation_map is not available in map_data.

    """
    if not map_data.accumulation_map:
        raise ValueError("Accumulation map is not available in map_data")

    accumulation_array = np.array(map_data.accumulation_map)
    # Normalize to 0-9 scale (accumulation can be > 1.0, so we need to handle this)
    max_accumulation = np.max(accumulation_array)
    if max_accumulation > 0:
        accumulation_normalized = (accumulation_array / max_accumulation * 9).astype(int)
    else:
        accumulation_normalized = np.zeros_like(accumulation_array, dtype=int)
    accumulation_normalized = np.clip(accumulation_normalized, 0, 9)

    lines = []
    for y in range(map_data.height):
        line = ""
        for x in range(map_data.width):
            line += str(accumulation_normalized[y, x])
        lines.append(line)
    return "\n".join(lines)
