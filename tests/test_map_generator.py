"""Tests for map generator."""

import os
import tempfile

import pytest

from mapgen import MapGenerator
from mapgen.map_data import MapData, Tile


def test_map_generator_initialization() -> None:
    """Test MapGenerator initialization."""
    generator = MapGenerator(width=50, height=50)
    assert generator.width == 50
    assert generator.height == 50


def test_map_generator_generate() -> None:
    """Test map generation."""
    generator = MapGenerator(width=50, height=50, wall_countdown=1000)
    map_data = generator.generate()
    assert map_data is not None
    assert map_data.noise_map is not None
    assert map_data.elevation_map is not None
    assert map_data.settlements is not None
    assert map_data.roads_graph is not None
    assert map_data.height == 50
    assert map_data.width == 50


def test_map_generator_plot_without_generate() -> None:
    """Test plotting without generating raises error."""
    generator = MapGenerator()
    # Create a minimal MapData without required fields
    from mapgen.map_data import MapData, Tile
    grid = [[Tile(walkable=True, movement_cost=1.0, blocks_line_of_sight=False, buildable=True, habitability=0.5, road_buildable=True, elevation_penalty=0.0, elevation_influence=0.0, smoothing_weight=1.0, symbol=".", color=(0.5, 0.5, 0.5), name="test", description="test", resources=[])]]
    map_data = MapData(grid=grid)
    with pytest.raises(ValueError, match="Map data missing noise_map"):
        generator.plot(map_data)


def test_map_data_save_load() -> None:
    """Test saving and loading map data in compact format."""
    # Create a simple test map
    tile1 = Tile(
        walkable=True,
        movement_cost=1.0,
        blocks_line_of_sight=False,
        buildable=True,
        habitability=0.8,
        road_buildable=True,
        elevation_penalty=0.0,
        elevation_influence=0.0,
        smoothing_weight=1.0,
        symbol=".",
        color=(0.5, 0.5, 0.5),
        name="grass",
        description="Grassland",
        resources=["food"],
    )
    tile2 = Tile(
        walkable=False,
        movement_cost=2.0,
        blocks_line_of_sight=True,
        buildable=False,
        habitability=0.0,
        road_buildable=False,
        elevation_penalty=1.0,
        elevation_influence=0.5,
        smoothing_weight=0.5,
        symbol="#",
        color=(0.2, 0.2, 0.2),
        name="mountain",
        description="Mountain",
        resources=["ore"],
    )

    grid = [[tile1, tile2], [tile2, tile1]]
    map_data = MapData(grid=grid)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        # Save in compact format
        map_data.save_to_json(temp_path)

        # Load back
        loaded = MapData.load_from_json(temp_path)

        # Verify
        assert loaded.width == 2
        assert loaded.height == 2
        assert loaded.get_terrain(0, 0).name == "grass"
        assert loaded.get_terrain(1, 0).name == "mountain"
        assert loaded.get_terrain(0, 1).name == "mountain"
        assert loaded.get_terrain(1, 1).name == "grass"

    finally:
        os.unlink(temp_path)


def test_map_data_backward_compatibility() -> None:
    """Test loading old JSON files without mode field (backward compatibility)."""
    # Create a simple test map
    tile1 = Tile(
        walkable=True,
        movement_cost=1.0,
        blocks_line_of_sight=False,
        buildable=True,
        habitability=0.8,
        road_buildable=True,
        elevation_penalty=0.0,
        elevation_influence=0.0,
        smoothing_weight=1.0,
        symbol=".",
        color=(0.5, 0.5, 0.5),
        name="grass",
        description="Grassland",
        resources=["food"],
    )

    grid = [[tile1, tile1], [tile1, tile1]]
    map_data = MapData(grid=grid)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        # Save the map
        map_data.save_to_json(temp_path)

        # Load back
        loaded = MapData.load_from_json(temp_path)

        # Verify
        assert loaded.width == 2
        assert loaded.height == 2
        assert loaded.get_terrain(0, 0).name == "grass"

    finally:
        os.unlink(temp_path)


def test_map_data_many_tiles() -> None:
    """Test that the system can handle many unique tiles."""
    # Create many unique tiles
    tiles = []
    for i in range(100):
        tile = Tile(
            walkable=i % 2 == 0,
            movement_cost=float(i),
            blocks_line_of_sight=False,
            buildable=True,
            habitability=0.5,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_influence=0.0,
            smoothing_weight=1.0,
            symbol=str(i % 10),
            color=(float(i % 3) / 3, float((i + 1) % 3) / 3, float((i + 2) % 3) / 3),
            name=f"tile_{i}",
            description=f"Tile {i}",
            resources=[f"resource_{i % 5}"],
        )
        tiles.append(tile)

    # Create a grid with all unique tiles (10 rows x 10 cols = 100 tiles)
    grid = []
    tile_index = 0
    for row in range(10):
        grid_row = []
        for col in range(10):
            grid_row.append(tiles[tile_index])
            tile_index += 1
        grid.append(grid_row)

    map_data = MapData(grid=grid)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        # Should work fine with many tiles
        map_data.save_to_json(temp_path)

        # Load back and verify
        loaded = MapData.load_from_json(temp_path)
        assert loaded.width == 10
        assert loaded.height == 10
        assert loaded.get_terrain(0, 0).name == "tile_0"
        assert loaded.get_terrain(9, 9).name == "tile_99"

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
