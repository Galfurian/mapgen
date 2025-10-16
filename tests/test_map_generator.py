"""Tests for map generator."""

import pytest

from mapgen import MapGenerator


def test_map_generator_initialization():
    """Test MapGenerator initialization."""
    generator = MapGenerator(width=50, height=50)
    assert generator.width == 50
    assert generator.height == 50
    assert generator.level is None


def test_map_generator_generate():
    """Test map generation."""
    generator = MapGenerator(width=50, height=50, wall_countdown=1000)
    generator.generate()
    assert generator.level is not None
    assert generator.noise_map is not None
    assert generator.elevation_map is not None
    assert generator.settlements is not None
    assert generator.roads_graph is not None
    assert len(generator.level) == 50
    assert len(generator.level[0]) == 50


def test_map_generator_plot_without_generate():
    """Test plotting without generating raises error."""
    generator = MapGenerator()
    with pytest.raises(ValueError, match="Map not generated yet"):
        generator.plot()
