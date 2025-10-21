#!/usr/bin/env python3
"""
Complete MapGen workflow - generates a fantasy map with configurable features and all available visualizations.

This script generates a fantasy map with configurable generation features and automatically
creates all available visualizations based on what was generated.
"""

import argparse
import logging
from pathlib import Path

from mapgen import (
    MapGenerator,
    plot_elevation_map,
    plot_map,
    plot_rainfall_map,
    plot_3d_map,
    get_ascii_map,
    logger,
)


def main() -> None:
    """Generate a fantasy map with configurable features and all available visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate fantasy maps with configurable features and all visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate everything with default settings
  python full_generation.py output/

  # Generate with specific seed for reproducibility
  python full_generation.py output/ --seed 123

  # Generate without rainfall and rivers
  python full_generation.py output/ --disable-rainfall --disable-rivers

  # Generate minimal map (terrain only)
  python full_generation.py output/ --disable-settlements --disable-roads --disable-rainfall --disable-rivers
        """,
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for all generated files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )

    # Feature control flags
    parser.add_argument(
        "--disable-rainfall",
        action="store_true",
        help="Disable rainfall generation",
    )
    parser.add_argument(
        "--disable-smoothing",
        action="store_true",
        help="Disable terrain smoothing",
    )
    parser.add_argument(
        "--disable-settlements",
        action="store_true",
        help="Disable settlement generation",
    )
    parser.add_argument(
        "--disable-roads",
        action="store_true",
        help="Disable road network generation",
    )
    parser.add_argument(
        "--disable-rivers",
        action="store_true",
        help="Disable river generation",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the map with configurable features
    logger.info("🗺️ Generating fantasy map...")
    generator = MapGenerator(
        seed=args.seed,
        enable_rainfall=not args.disable_rainfall,
        enable_smoothing=not args.disable_smoothing,
        enable_settlements=not args.disable_settlements,
        enable_roads=not args.disable_roads,
        enable_rivers=not args.disable_rivers,
    )

    map_data = generator.generate()

    if map_data is None:
        logger.error("❌ Map generation failed!")
        return

    logger.info("✅ Map generation succeeded!")
    logger.info(f"📐 Map size: {map_data.width}×{map_data.height}")
    logger.info(f"🏘️ Settlements: {len(map_data.settlements)}")
    logger.info(f"🛣️ Roads: {len(map_data.roads)}")

    # Count terrain types
    terrain_counts = {}
    for y in range(map_data.height):
        for x in range(map_data.width):
            tile = map_data.get_terrain(x, y)
            terrain_counts[tile.name] = terrain_counts.get(tile.name, 0) + 1

    logger.info("🏔️ Terrain distribution:")
    for tile_name, count in sorted(terrain_counts.items()):
        percentage = (count / (map_data.width * map_data.height)) * 100
        logger.info(f"   {tile_name}: {count:,} tiles ({percentage:.1f}%)")

    # Generate ALL available visualizations automatically
    dpi = 300
    generated_files = 0

    # 1. Main terrain map (always generated)
    logger.info("🖼️ Generating main terrain map...")
    fig = plot_map(map_data)
    main_map_path = output_dir / f"terrain_map_seed_{args.seed}.png"
    fig.savefig(main_map_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.clear()
    logger.info(f"💾 Main map saved: {main_map_path}")
    generated_files += 1

    # 2. Elevation map (always available since elevation_map is always initialized)
    if map_data.elevation_map:
        logger.info("📊 Generating elevation map...")
        fig = plot_elevation_map(map_data, title=f"Elevation Map (Seed: {args.seed})")
        elevation_path = output_dir / f"elevation_map_seed_{args.seed}.png"
        fig.savefig(elevation_path, dpi=dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 Elevation map saved: {elevation_path}")
        generated_files += 1

    # 3. Rainfall map (available if rainfall was generated)
    if map_data.rainfall_map:
        logger.info("🌧️ Generating rainfall map...")
        fig = plot_rainfall_map(map_data, title=f"Rainfall Map (Seed: {args.seed})")
        rainfall_path = output_dir / f"rainfall_map_seed_{args.seed}.png"
        fig.savefig(rainfall_path, dpi=dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 Rainfall map saved: {rainfall_path}")
        generated_files += 1

    # 4. 3D visualization (always available)
    logger.info("🎲 Generating 3D visualization...")
    fig = plot_3d_map(map_data)
    map_3d_path = output_dir / f"terrain_3d_seed_{args.seed}.png"
    fig.savefig(map_3d_path, dpi=dpi, bbox_inches="tight")
    fig.clear()
    logger.info(f"💾 3D map saved: {map_3d_path}")
    generated_files += 1

    # 5. ASCII representation (always available)
    logger.info("📄 Generating ASCII map...")
    ascii_map = get_ascii_map(map_data)
    ascii_path = output_dir / f"terrain_ascii_seed_{args.seed}.txt"
    with open(ascii_path, "w") as f:
        f.write(ascii_map)
    logger.info(f"💾 ASCII map saved: {ascii_path}")
    generated_files += 1

    # 6. JSON data (always available)
    logger.info("📦 Saving map data as JSON...")
    json_path = output_dir / f"map_data_seed_{args.seed}.json"
    map_data.save_to_json(str(json_path))
    logger.info(f"💾 JSON data saved: {json_path}")
    generated_files += 1

    logger.info("🎉 Map generation and visualization finished!")
    logger.info(f"📁 All outputs saved to: {output_dir}")
    logger.info(f"📊 Generated {generated_files} files")


if __name__ == "__main__":
    main()
