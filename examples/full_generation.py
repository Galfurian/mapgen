#!/usr/bin/env python3
"""Complete MapGen workflow example demonstrating generation, saving, loading, and visualization."""

import argparse
import os
import random
from pathlib import Path

import numpy as np

from mapgen import roads, settlements, terrain, visualization
from mapgen.map_data import MapData, Tile
from mapgen.map_generator import MapGenerator, logger


def main() -> None:
    """Demonstrate the complete MapGen workflow."""
    parser = argparse.ArgumentParser(description="Generate a fantasy map")
    parser.add_argument(
        "--output",
        type=str,
        default="generated_map.json",
        help="Output filename (extension determines format: .json for data, .png for image, .txt for ASCII)",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=150,
        help="Map width (default: 150)",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=100,
        help="Map height (default: 100)",
    )
    parser.add_argument(
        "-N",
        "--scale",
        type=float,
        default=50.0,
        help="Noise scale (default: 50.0)",
    )
    parser.add_argument(
        "-O",
        "--octaves",
        type=int,
        default=6,
        help="Noise octaves (default: 6)",
    )
    parser.add_argument(
        "-P",
        "--persistence",
        type=float,
        default=0.5,
        help="Noise persistence (default: 0.5)",
    )
    parser.add_argument(
        "-L",
        "--lacunarity",
        type=float,
        default=2.0,
        help="Noise lacunarity (default: 2.0)",
    )
    parser.add_argument(
        "-sd",
        "--settlement-density",
        type=float,
        default=0.002,
        help="Settlement density (default: 0.002)",
    )
    parser.add_argument(
        "-msr",
        "--min-settlement-radius",
        type=float,
        default=0.5,
        help="Minimum settlement radius (default: 0.5)",
    )
    parser.add_argument(
        "-mxsr",
        "--max-settlement-radius",
        type=float,
        default=1.0,
        help="Maximum settlement radius (default: 1.0)",
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
        width=args.width,
        height=args.height,
        scale=args.scale,
        octaves=args.octaves,
        persistence=args.persistence,
        lacunarity=args.lacunarity,
        settlement_density=args.settlement_density,
        min_settlement_radius=args.min_settlement_radius,
        max_settlement_radius=args.max_settlement_radius,
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
    generate_txt = output_extension == ".txt"

    # If no extension or unknown extension, default to JSON
    if not output_extension or (
        not generate_json and not generate_png and not generate_txt
    ):
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

    # Generate TXT (ASCII map) if requested
    if generate_txt:
        txt_path = output_path if generate_txt else output_path.with_suffix(".txt")
        print(f"Saving ASCII map to: {txt_path}")
        ascii_map = visualization.get_ascii_map(map_data)
        with open(txt_path, "w") as f:
            f.write(ascii_map)
        txt_size = os.path.getsize(txt_path)
        print(f"   📄 TXT size: {txt_size:,} bytes")

    print("🎉 Map generation completed successfully!")


def _generate_map(
    width: int = 150,
    height: int = 100,
    scale: float = 50.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    settlement_density: float = 0.002,
    min_settlement_radius: float = 0.5,
    max_settlement_radius: float = 1.0,
    enable_smoothing: bool = True,
    enable_settlements: bool = True,
    enable_roads: bool = True,
    seed: int = 42,
    smoothing_iterations: int = 3,
) -> MapData:

    logger.info(f"Starting map generation: {width}*{height}")

    # Set random seed for reproducible generation
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Using random seed: {seed}")

    tiles = MapGenerator.get_default_tiles()

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


if __name__ == "__main__":
    main()
