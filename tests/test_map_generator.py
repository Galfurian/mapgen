"""Tests for map generator."""

from mapgen import generate_map


def test_generate_map() -> None:
    """Test map generation."""
    map_data = generate_map(width=50, height=50)
    assert map_data is not None
    assert map_data.settlements is not None
    assert map_data.roads is not None
    assert map_data.height == 50
    assert map_data.width == 50
