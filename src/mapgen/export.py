"""
Export utilities for MapGen.

This module provides functions for exporting sub-maps and other export-related
functionality.
"""

from __future__ import annotations

from mapgen.map_data import MapData, Position, Settlement, Road, WaterRoute


def extract_sub_map(
    map_data: MapData,
    top_left_x: int,
    top_left_y: int,
    bottom_right_x: int,
    bottom_right_y: int,
) -> MapData:
    """
    Extract a sub-map from the given map data.

    Creates a new MapData object containing only the data within the specified
    rectangular region. All coordinates are adjusted relative to the top-left
    corner of the sub-map.

    Args:
        map_data (MapData):
            The original map data.
        top_left_x (int):
            The x-coordinate of the top-left corner.
        top_left_y (int):
            The y-coordinate of the top-left corner.
        bottom_right_x (int):
            The x-coordinate of the bottom-right corner.
        bottom_right_y (int):
            The y-coordinate of the bottom-right corner.

    Returns:
        MapData:
            A new MapData object containing the sub-map.

    Raises:
        ValueError:
            If the specified region is invalid or out of bounds.

    """
    # Validate coordinates
    if (
        top_left_x < 0
        or top_left_y < 0
        or bottom_right_x >= map_data.width
        or bottom_right_y >= map_data.height
        or top_left_x > bottom_right_x
        or top_left_y > bottom_right_y
    ):
        raise ValueError("Invalid sub-map coordinates")

    new_width = bottom_right_x - top_left_x + 1
    new_height = bottom_right_y - top_left_y + 1

    # Extract grid
    new_grid = [
        [map_data.grid[y + top_left_y][x + top_left_x] for x in range(new_width)]
        for y in range(new_height)
    ]

    # Extract layers
    new_layers = {}
    for layer_name, layer_data in map_data.layers.items():
        new_layers[layer_name] = [
            [layer_data[y + top_left_y][x + top_left_x] for x in range(new_width)]
            for y in range(new_height)
        ]

    # Extract settlements within the region
    new_settlements = []
    for settlement in map_data.settlements:
        if (
            top_left_x <= settlement.position.x <= bottom_right_x
            and top_left_y <= settlement.position.y <= bottom_right_y
        ):
            new_position = Position(
                settlement.position.x - top_left_x,
                settlement.position.y - top_left_y,
            )
            new_settlements.append(
                Settlement(
                    name=settlement.name,
                    position=new_position,
                    radius=settlement.radius,
                    connectivity=settlement.connectivity,
                    is_harbor=settlement.is_harbor,
                )
            )

    # Extract roads that are entirely within the region
    new_roads = []
    for road in map_data.roads:
        # Check if all path positions are within the region
        if all(
            top_left_x <= pos.x <= bottom_right_x
            and top_left_y <= pos.y <= bottom_right_y
            for pos in road.path
        ):
            new_path = [
                Position(pos.x - top_left_x, pos.y - top_left_y) for pos in road.path
            ]
            new_roads.append(
                Road(
                    start_settlement=road.start_settlement,
                    end_settlement=road.end_settlement,
                    path=new_path,
                )
            )

    # Extract water routes that are entirely within the region
    new_water_routes = []
    for water_route in map_data.water_routes:
        # Check if all path positions are within the region
        if all(
            top_left_x <= pos.x <= bottom_right_x
            and top_left_y <= pos.y <= bottom_right_y
            for pos in water_route.path
        ):
            new_path = [
                Position(pos.x - top_left_x, pos.y - top_left_y)
                for pos in water_route.path
            ]
            new_water_routes.append(
                WaterRoute(
                    start_harbor=water_route.start_harbor,
                    end_harbor=water_route.end_harbor,
                    path=new_path,
                )
            )

    # Create the new MapData
    return MapData(
        tiles=map_data.tiles,  # Tiles list is shared
        grid=new_grid,
        layers=new_layers,
        settlements=new_settlements,
        roads=new_roads,
        water_routes=new_water_routes,
    )
