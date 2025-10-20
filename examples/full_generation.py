#!/usr/bin/env python3
"""Complete MapGen workflow example demonstrating generation, saving, loading, and visualization."""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np

from mapgen import roads, settlements, terrain, visualization
from mapgen.map_data import MapData
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set the logging level based on verbosity.
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Generate the map.
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
        enable_smoothing=not args.disable_smoothing,
        enable_settlements=not args.disable_settlements,
        enable_roads=not args.disable_roads,
        seed=args.seed,
        smoothing_iterations=args.smoothing_iterations,
    )

    if map_data is not None:
        logger.info("✅ Map generation succeeded!")
    else:
        logger.error("❌ Map generation failed!")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_extension = output_path.suffix.lower()

    # Save to JSON if requested.
    if output_extension == ".json":
        output_path = output_path.with_suffix(".json")

        logger.info(f"Saving map data to: {output_path}")
        map_data.save_to_json(str(output_path.with_suffix(".json")))
        json_size = os.path.getsize(output_path)
        logger.info(f"📦 JSON size: {json_size:,} bytes")

    elif output_extension == ".png":
        output_path = output_path.with_suffix(".png")

        logger.info(f"Saving map image to: {output_path}")
        fig = visualization.plot_map(map_data)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.get_figure().clear()
        png_size = os.path.getsize(output_path)
        logger.info(f"🖼️ PNG size: {png_size:,} bytes")

    elif output_extension == ".txt":
        output_path = output_path.with_suffix(".txt")

        logger.info(f"Saving ASCII map to: {output_path}")
        ascii_map = visualization.get_ascii_map(map_data)
        with open(output_path, "w") as f:
            f.write(ascii_map)
        txt_size = os.path.getsize(output_path)
        logger.info(f"📄 TXT size: {txt_size:,} bytes")

    logger.info("🎉 Map generation completed successfully!")


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

    logger.info("🗺️ Generating fantasy map...")

    # Set random seed for reproducible generation
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"  Using random seed: {seed}")

    tiles = MapGenerator.get_default_tiles()

    logger.debug("  Initializing map level...")
    map_data = MapData(
        tiles=tiles,
        grid=[[0 for _ in range(width)] for _ in range(height)],
    )

    logger.debug("  Generating noise map...")
    terrain.generate_noise_map(
        map_data,
        width,
        height,
        scale,
        octaves,
        persistence,
        lacunarity,
    )

    logger.debug("  Applying terrain features...")
    terrain.apply_terrain_features(
        map_data,
    )

    if enable_smoothing:
        logger.debug("  Smoothing terrain...")
        terrain.smooth_terrain(
            map_data,
            smoothing_iterations,
        )

    if enable_settlements:
        logger.debug("  Generating settlements...")
        settlements.generate_settlements(
            map_data,
            settlement_density,
            min_settlement_radius,
            max_settlement_radius,
        )
        logger.debug(f"  🏘️ Settlements: {len(map_data.settlements)}")

    if enable_roads:
        logger.debug("Generating road network")
        roads.generate_roads(
            map_data,
        )
        logger.debug(f"  🛣️ Roads: {len(map_data.roads)}")

    return map_data


if __name__ == "__main__":
    main()
