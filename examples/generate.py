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
    generate_map,
    logger,
    get_ascii_layer,
    get_ascii_map,
    plot_map,
    plot_map_layer,
)
from mapgen.visualization import plot_3d_map


def main() -> None:
    """Generate a fantasy map with configurable features and all available visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate fantasy maps with configurable features and all visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate everything with default settings (recommended)
  python full_generation.py output/

  # Generate with specific seed for reproducibility
  python full_generation.py output/ --seed 123

  # Generate without settlements and roads (keep climate data)
  python full_generation.py output/ --disable-settlements --disable-roads

  # Generate minimal map (terrain + climate data only)
  python full_generation.py output/ --disable-settlements --disable-roads --disable-rivers --disable-vegetation
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
    parser.add_argument(
        "--width",
        type=int,
        default=150,
        help="Map width (default: 150)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=100,
        help="Map height (default: 100)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Padding around edges (default: 2)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=50.0,
        help="Noise scale (default: 50.0)",
    )
    parser.add_argument(
        "--octaves",
        type=int,
        default=6,
        help="Number of noise octaves (default: 6)",
    )
    parser.add_argument(
        "--persistence",
        type=float,
        default=0.5,
        help="Noise persistence (default: 0.5)",
    )
    parser.add_argument(
        "--lacunarity",
        type=float,
        default=2.0,
        help="Noise lacunarity (default: 2.0)",
    )
    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        default=5,
        help="Number of smoothing iterations (default: 5)",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=0.5,
        help="Sigma value for smoothing (default: 0.5)",
    )
    parser.add_argument(
        "--settlement-density",
        type=float,
        default=0.003,
        help="Density of settlements (default: 0.002)",
    )
    parser.add_argument(
        "--min-settlement-radius",
        type=float,
        default=0.5,
        help="Minimum settlement radius (default: 0.5)",
    )
    parser.add_argument(
        "--max-settlement-radius",
        type=float,
        default=1.0,
        help="Maximum settlement radius (default: 1.0)",
    )
    parser.add_argument(
        "--sea-level",
        type=float,
        default=0.00,
        help="Sea level for land/sea ratio (default: 0.0)",
    )
    parser.add_argument(
        "--rainfall-temp-weight",
        type=float,
        default=0.3,
        help="Weight for temperature influence on rainfall (default: 0.3)",
    )
    parser.add_argument(
        "--rainfall-humidity-weight",
        type=float,
        default=0.4,
        help="Weight for humidity influence on rainfall (default: 0.4)",
    )
    parser.add_argument(
        "--rainfall-orographic-weight",
        type=float,
        default=0.3,
        help="Weight for orographic influence on rainfall (default: 0.3)",
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
        "--disable-vegetation",
        action="store_true",
        help="Disable climate-driven vegetation placement",
    )
    parser.add_argument(
        "--min-source-elevation",
        type=float,
        default=0.6,
        help="Minimum elevation for river sources - higher = only tallest mountains (default: 0.6)",
    )
    parser.add_argument(
        "--min-source-rainfall",
        type=float,
        default=0.5,
        help="Minimum rainfall percentile for river sources (default: 0.5)",
    )
    parser.add_argument(
        "--min-river-length",
        type=int,
        default=10,
        help="Minimum path length to place a river (default: 10)",
    )
    parser.add_argument(
        "--forest-coverage",
        type=float,
        default=0.15,
        help="Target forest coverage ratio (default: 0.15)",
    )
    parser.add_argument(
        "--desert-coverage",
        type=float,
        default=0.10,
        help="Target desert coverage ratio (default: 0.10)",
    )
    parser.add_argument(
        "--max-lake-size",
        type=int,
        default=None,
        help="Maximum size for edge-connected water bodies to be classified as lakes (default: auto-calculated)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=900,
        help="DPI for output images (default: 900)",
    )
    parser.add_argument(
        "--dump-ascii",
        action="store_true",
        help="Dump ASCII representation of the main terrain map",
    )
    parser.add_argument(
        "--dump-3dmap",
        action="store_true",
        help="Dump 3D representation of the main terrain map",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Dump map data as JSON file",
    )
    parser.add_argument(
        "--dump-elevation",
        action="store_true",
        help="Dump elevation data, both as text and as a visualization",
    )
    parser.add_argument(
        "--dump-temperature",
        action="store_true",
        help="Dump temperature data, both as text and as a visualization",
    )
    parser.add_argument(
        "--dump-humidity",
        action="store_true",
        help="Dump humidity data, both as text and as a visualization",
    )
    parser.add_argument(
        "--dump-orographic",
        action="store_true",
        help="Dump orographic data, both as text and as a visualization",
    )
    parser.add_argument(
        "--dump-rainfall",
        action="store_true",
        help="Dump rainfall data, both as text and as a visualization",
    )
    parser.add_argument(
        "--dump-accumulation",
        action="store_true",
        help="Dump water accumulation data, both as text and as a visualization",
    )
    parser.add_argument(
        "--dump-all",
        action="store_true",
        help="Dump all available visualizations and data",
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

    map_data = generate_map(
        width=args.width,
        height=args.height,
        padding=args.padding,
        scale=args.scale,
        octaves=args.octaves,
        persistence=args.persistence,
        lacunarity=args.lacunarity,
        smoothing_iterations=args.smoothing_iterations,
        smoothing_sigma=args.smoothing_sigma,
        settlement_density=args.settlement_density,
        min_settlement_radius=args.min_settlement_radius,
        max_settlement_radius=args.max_settlement_radius,
        seed=args.seed,
        enable_settlements=not args.disable_settlements,
        enable_roads=not args.disable_roads,
        enable_rivers=not args.disable_rivers,
        enable_vegetation=not args.disable_vegetation,
        min_source_elevation=args.min_source_elevation,
        min_source_rainfall=args.min_source_rainfall,
        min_river_length=args.min_river_length,
        sea_level=args.sea_level,
        rainfall_temp_weight=args.rainfall_temp_weight,
        rainfall_humidity_weight=args.rainfall_humidity_weight,
        rainfall_orographic_weight=args.rainfall_orographic_weight,
        forest_coverage=args.forest_coverage,
        desert_coverage=args.desert_coverage,
        max_lake_size=args.max_lake_size,
    )

    logger.info("✅ Map generation succeeded!")
    logger.info(f"📐 Map size: {map_data.width}x{map_data.height}")
    logger.info(f"🏘️ Settlements: {len(map_data.settlements)}")
    logger.info(f"🛣️ Roads: {len(map_data.roads)}")
    logger.info(f"🚢 Water routes: {len(map_data.water_routes)}")

    # Count terrain types
    terrain_counts: dict[str, int] = {}
    for y in range(map_data.height):
        for x in range(map_data.width):
            tile = map_data.get_terrain(x, y)
            terrain_counts[tile.name] = terrain_counts.get(tile.name, 0) + 1

    logger.info("🏔️ Terrain distribution:")
    for tile_name, count in sorted(terrain_counts.items()):
        percentage = (count / (map_data.width * map_data.height)) * 100
        logger.info(f"   {tile_name}: {count:,} tiles ({percentage:.1f}%)")

    # Elevation map (always available since elevation_map is always initialized)
    if args.dump_elevation or args.dump_all:
        logger.info("📊 Generating elevation map...")
        fig = plot_map_layer(
            map_data.elevation_map,
            colormap="terrain",
            title=f"Elevation Map (Seed: {args.seed})",
            label="Elevation",
        )
        elevation_path = output_dir / f"seed_{args.seed}_layer_elevation.png"
        fig.savefig(elevation_path, dpi=args.dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 Elevation map saved: {elevation_path}")

        logger.info("📄 Generating ASCII elevation map...")
        ascii_elevation_map = get_ascii_layer(map_data.elevation_map)
        ascii_elevation_path = output_dir / f"seed_{args.seed}_ascii_map_elevation.txt"
        with open(ascii_elevation_path, "w") as f:
            f.write(ascii_elevation_map)
        logger.info(f"💾 ASCII elevation map saved: {ascii_elevation_path}")

    # Rainfall map (available if rainfall was generated)
    if args.dump_rainfall or args.dump_all:
        logger.info("🌧️ Generating rainfall map...")
        fig = plot_map_layer(
            map_data.rainfall_map,
            colormap="Blues",
            title=f"Rainfall Map (Seed: {args.seed})",
            label="Rainfall Intensity",
        )
        rainfall_path = output_dir / f"seed_{args.seed}_layer_rainfall.png"
        fig.savefig(rainfall_path, dpi=args.dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 Rainfall map saved: {rainfall_path}")

        logger.info("📄 Generating ASCII rainfall map...")
        ascii_rainfall_map = get_ascii_layer(map_data.rainfall_map)
        ascii_rainfall_path = output_dir / f"seed_{args.seed}_ascii_map_rainfall.txt"
        with open(ascii_rainfall_path, "w") as f:
            f.write(ascii_rainfall_map)
        logger.info(f"💾 ASCII rainfall map saved: {ascii_rainfall_path}")

    # Temperature map (available if temperature was generated)
    if args.dump_temperature or args.dump_all:
        logger.info("🌡️ Generating temperature map...")
        fig = plot_map_layer(
            map_data.temperature_map,
            colormap="RdYlBu_r",
            title=f"Temperature Map (Seed: {args.seed})",
            label="Temperature",
        )
        temperature_path = output_dir / f"seed_{args.seed}_layer_temperature.png"
        fig.savefig(temperature_path, dpi=args.dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 Temperature map saved: {temperature_path}")

        logger.info("📄 Generating ASCII temperature map...")
        ascii_temperature_map = get_ascii_layer(map_data.temperature_map)
        ascii_temperature_path = (
            output_dir / f"seed_{args.seed}_ascii_map_temperature.txt"
        )
        with open(ascii_temperature_path, "w") as f:
            f.write(ascii_temperature_map)
        logger.info(f"💾 ASCII temperature map saved: {ascii_temperature_path}")

    # Accumulation map (available if accumulation was generated)
    if args.dump_accumulation or args.dump_all:
        logger.info("💧 Generating accumulation map...")
        fig = plot_map_layer(
            map_data.accumulation_map,
            colormap="Blues",
            title=f"Water Accumulation Map (Seed: {args.seed})",
            label="Water Accumulation",
        )
        accumulation_path = output_dir / f"seed_{args.seed}_layer_accumulation.png"
        fig.savefig(accumulation_path, dpi=args.dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 Accumulation map saved: {accumulation_path}")

        logger.info("📄 Generating ASCII accumulation map...")
        ascii_accumulation_map = get_ascii_layer(map_data.accumulation_map)
        ascii_accumulation_path = (
            output_dir / f"seed_{args.seed}_ascii_map_accumulation.txt"
        )
        with open(ascii_accumulation_path, "w") as f:
            f.write(ascii_accumulation_map)
        logger.info(f"💾 ASCII accumulation map saved: {ascii_accumulation_path}")

    if args.dump_humidity or args.dump_all:
        logger.info("💧 Generating humidity map...")
        fig = plot_map_layer(
            map_data.humidity_map,
            colormap="Blues",
            title=f"Humidity Map (Seed: {args.seed})",
            label="Humidity",
        )
        humidity_path = output_dir / f"seed_{args.seed}_layer_humidity.png"
        fig.savefig(humidity_path, dpi=args.dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 Humidity map saved: {humidity_path}")

        logger.info("📄 Generating ASCII humidity map...")
        ascii_humidity_map = get_ascii_layer(map_data.humidity_map)
        ascii_humidity_path = output_dir / f"seed_{args.seed}_ascii_map_humidity.txt"
        with open(ascii_humidity_path, "w") as f:
            f.write(ascii_humidity_map)
        logger.info(f"💾 ASCII humidity map saved: {ascii_humidity_path}")

    if args.dump_orographic or args.dump_all:
        logger.info("⛰️ Generating orographic map...")
        fig = plot_map_layer(
            map_data.orographic_map,
            colormap="terrain",
            title=f"Orographic Map (Seed: {args.seed})",
            label="Orographic Factor",
        )
        orographic_path = output_dir / f"seed_{args.seed}_layer_orographic.png"
        fig.savefig(orographic_path, dpi=args.dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 Orographic map saved: {orographic_path}")

        logger.info("📄 Generating ASCII orographic map...")
        ascii_orographic_map = get_ascii_layer(map_data.orographic_map)
        ascii_orographic_path = (
            output_dir / f"seed_{args.seed}_ascii_map_orographic.txt"
        )
        with open(ascii_orographic_path, "w") as f:
            f.write(ascii_orographic_map)
        logger.info(f"💾 ASCII orographic map saved: {ascii_orographic_path}")

    if args.dump_ascii or args.dump_all:
        logger.info("📄 Generating ASCII map...")
        ascii_map = get_ascii_map(map_data)
        ascii_path = output_dir / f"seed_{args.seed}_ascii_map.txt"
        with open(ascii_path, "w") as f:
            f.write(ascii_map)
        logger.info(f"💾 ASCII map saved: {ascii_path}")

    if args.dump_3dmap or args.dump_all:
        logger.info("🎲 Generating 3D visualization...")
        fig = plot_3d_map(map_data)
        map_3d_path = output_dir / f"seed_{args.seed}_3d_map.png"
        fig.savefig(map_3d_path, dpi=args.dpi, bbox_inches="tight")
        fig.clear()
        logger.info(f"💾 3D map saved: {map_3d_path}")

    if args.dump_json or args.dump_all:
        logger.info("📦 Saving map data as JSON...")
        json_path = output_dir / f"seed_{args.seed}_map_data.json"
        map_data.save_to_json(str(json_path))
        logger.info(f"💾 JSON data saved: {json_path}")

    # Main terrain map (always generated)
    logger.info("🖼️ Generating main terrain map...")
    fig = plot_map(map_data)
    main_map_path = output_dir / f"seed_{args.seed}_map.png"
    fig.savefig(main_map_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.clear()
    logger.info(f"💾 Main map saved: {main_map_path}")

    logger.info("🎉 Map generation and visualization finished!")
    logger.info(f"📁 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
