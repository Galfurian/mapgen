#!/usr/bin/env python3
"""Complete MapGen workflow example demonstrating generation, saving, loading, and visualization."""

import argparse
import os
import random
from pathlib import Path

import numpy as np

from mapgen import roads, settlements, terrain, visualization
from mapgen.map_data import MapData, Tile
from mapgen.map_generator import logger


def main() -> None:
    """Demonstrate the complete MapGen workflow."""
    parser = argparse.ArgumentParser(description="Generate a fantasy map")
    parser.add_argument(
        "--output",
        type=str,
        default="generated_map.json",
        help="Output filename (extension determines format: .json for data, .png for image)",
    )
    parser.add_argument(
        "-dsm",
        "--disable-smoothing",
        action="store_true",
        help="Disable terrain smoothing",
    )
    parser.add_argument(
        "-dst",
        "--disable-settlements",
        action="store_true",
        help="Disable settlement generation",
    )
    parser.add_argument(
        "-drd",
        "--disable-roads",
        action="store_true",
        help="Disable road generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for map generation (default: 42)",
    )
    parser.add_argument(
        "-si",
        "--smoothing-iterations",
        type=int,
        default=3,
        help="Number of smoothing iterations (default: 3)",
    )

    args = parser.parse_args()

    # Resolve enable/disable flags
    enable_smoothing = not args.disable_smoothing
    enable_settlements = not args.disable_settlements
    enable_roads = not args.disable_roads

    print("🗺️  Generating fantasy map...")
    print(f"   Smoothing: {'enabled' if enable_smoothing else 'disabled'}")
    print(f"   Settlements: {'enabled' if enable_settlements else 'disabled'}")
    print(f"   Roads: {'enabled' if enable_roads else 'disabled'}")

    # Generate the map
    map_data = _generate_map(
        enable_smoothing=enable_smoothing,
        enable_settlements=enable_settlements,
        enable_roads=enable_roads,
        seed=args.seed,
        smoothing_iterations=args.smoothing_iterations,
    )

    if map_data is None:
        print("   ❌ Map generation failed!")
        return

    print(f"   ✅ Generated {map_data.width}x{map_data.height} map")
    print(
        f"   📊 Terrain types: {len(set(tile.name for row in map_data.tiles_grid for tile in row))}"
    )
    if enable_settlements:
        print(
            f"   🏘️  Settlements: {len(map_data.settlements) if map_data.settlements else 0}"
        )
    if enable_roads:
        print(f"   🛣️  Roads: {len(map_data.roads)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_extension = output_path.suffix.lower()

    # Determine what to generate based on file extension
    generate_json = output_extension == ".json"
    generate_png = output_extension == ".png"

    # If no extension or unknown extension, default to JSON
    if not output_extension or (not generate_json and not generate_png):
        generate_json = True
        output_path = output_path.with_suffix(".json")

    # Save to JSON if requested
    if generate_json:
        json_path = output_path if generate_json else output_path.with_suffix(".json")
        print(f"Saving to JSON: {json_path}")
        map_data.save_to_json(str(json_path))
        json_size = os.path.getsize(json_path)
        print(f"   💾 JSON size: {json_size:,} bytes")

    # Generate PNG if requested
    if generate_png:
        png_path = output_path if generate_png else output_path.with_suffix(".png")
        print(f"Generating PNG: {png_path}")
        fig = visualization.plot_map(map_data)
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.get_figure().clear()  # Free memory
        png_size = os.path.getsize(png_path)
        print(f"   🖼️  PNG size: {png_size:,} bytes")

    print("🎉 Map generation completed successfully!")


def _generate_map(
    enable_smoothing: bool = True,
    enable_settlements: bool = True,
    enable_roads: bool = True,
    seed: int = 42,
    smoothing_iterations: int = 3,
) -> MapData:

    width: int = 150
    height: int = 100
    padding: int = 2
    scale: float = 50.0
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    settlement_density: float = 0.002
    min_settlement_radius: float = 0.5
    max_settlement_radius: float = 1.0

    logger.info(f"Starting map generation: {width}*{height}")

    # Set random seed for reproducible generation
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Using random seed: {seed}")

    tiles = _get_default_tiles()

    logger.debug("Initializing map level")
    map_data = MapData(
        tiles=tiles,
        grid=[[0 for _ in range(width)] for _ in range(height)],
    )

    logger.debug("Generating noise map")
    terrain.generate_noise_map(
        map_data,
        width,
        height,
        scale,
        octaves,
        persistence,
        lacunarity,
    )

    logger.debug("Applying terrain features")
    terrain.apply_terrain_features(
        map_data,
    )

    if enable_smoothing:
        logger.debug("Smoothing terrain")
        terrain.smooth_terrain(
            map_data,
            smoothing_iterations,
        )

    if enable_settlements:
        logger.debug("Generating settlements")
        settlements.generate_settlements(
            map_data,
            settlement_density,
            min_settlement_radius,
            max_settlement_radius,
        )
        logger.debug(f"Generated {len(map_data.settlements)} settlements")

    if enable_roads:
        logger.debug("Generating road network")
        roads.generate_roads(
            map_data,
        )
        logger.debug(f"Generated road network with {len(map_data.roads)} roads")

    logger.info("Map generation completed successfully")

    return map_data


def _get_default_tiles() -> list[Tile]:
    """
    Create the catalog of tiles used for map generation.

    Returns:
        list[Tile]:
            List of Tile instances used in the map.

    """
    tiles = [
        Tile(
            name="wall",
            description="Impassable wall",
            walkable=False,
            movement_cost=1.0,
            blocks_line_of_sight=True,
            buildable=False,
            habitability=0.0,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_influence=0.0,
            smoothing_weight=1.0,
            elevation_min=0.0,
            elevation_max=0.0,
            terrain_priority=0,
            smoothing_priority=1,
            symbol="#",
            color=(0.0, 0.0, 0.0),
            resources=[],
        ),
        Tile(
            name="floor",
            description="Open floor space",
            walkable=True,
            movement_cost=1.0,
            blocks_line_of_sight=False,
            buildable=True,
            habitability=0.3,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_influence=0.0,
            smoothing_weight=1.0,
            elevation_min=0.0,
            elevation_max=0.0,
            terrain_priority=0,
            smoothing_priority=0,
            diggable=True,
            symbol=".",
            color=(0.9, 0.9, 0.9),
            resources=[],
        ),
        Tile(
            name="water",
            description="Water terrain",
            walkable=True,
            movement_cost=2.0,
            blocks_line_of_sight=False,
            buildable=False,
            habitability=0.0,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_influence=-0.5,
            smoothing_weight=1.0,
            elevation_min=-1.0,
            elevation_max=0.03,
            terrain_priority=1,
            smoothing_priority=2,
            symbol="~",
            color=(0.2, 0.5, 1.0),
            resources=[],
        ),
        Tile(
            name="forest",
            description="Forest terrain",
            walkable=True,
            movement_cost=1.2,
            blocks_line_of_sight=False,
            buildable=True,
            habitability=0.7,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_influence=0.0,
            smoothing_weight=1.0,
            elevation_min=0.1,
            elevation_max=0.5,
            terrain_priority=3,
            smoothing_priority=4,
            symbol="F",
            color=(0.2, 0.6, 0.2),
            resources=["wood", "game"],
        ),
        Tile(
            name="plains",
            description="Open plains",
            walkable=True,
            movement_cost=1.0,
            blocks_line_of_sight=False,
            buildable=True,
            habitability=0.9,
            road_buildable=True,
            elevation_penalty=0.0,
            elevation_influence=0.0,
            smoothing_weight=1.0,
            elevation_min=0.03,
            elevation_max=0.1,
            terrain_priority=2,
            smoothing_priority=0,
            symbol=".",
            color=(0.8, 0.9, 0.6),
            resources=["grain", "herbs"],
        ),
        Tile(
            name="mountain",
            description="Mountain terrain",
            walkable=False,
            movement_cost=1.0,
            blocks_line_of_sight=True,
            buildable=False,
            habitability=0.1,
            road_buildable=False,
            elevation_penalty=0.0,
            elevation_influence=1.0,
            smoothing_weight=1.0,
            elevation_min=0.5,
            elevation_max=1.0,
            terrain_priority=4,
            smoothing_priority=3,
            symbol="^",
            color=(0.5, 0.4, 0.3),
            resources=["stone", "ore"],
        ),
    ]
    return tiles


if __name__ == "__main__":
    main()
