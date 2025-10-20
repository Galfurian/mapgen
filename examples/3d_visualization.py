#!/usr/bin/env python3
"""3D Map Visualization Example - See your fantasy maps come to life in 3D!"""

import argparse
import logging
import random
from pathlib import Path

from mapgen import MapGenerator, plot_3d_map

# Suppress matplotlib font debug messages
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'  # Set default font to avoid font search


def main() -> None:
    """Generate and visualize a fantasy map in stunning 3D."""
    parser = argparse.ArgumentParser(description="Generate a 3D fantasy map")
    parser.add_argument(
        "--output",
        type=str,
        default="output/3d_map.png",
        help="Output filename for the 3D visualization",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=100,
        help="Map width (default: 100)",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=80,
        help="Map height (default: 80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--elevation-scale",
        type=float,
        default=3.0,
        help="Elevation exaggeration scale (default: 3.0) - higher values make mountains taller and settlements more visible",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="terrain",
        choices=["terrain", "viridis", "plasma", "inferno", "magma"],
        help="Color map for terrain (default: terrain)",
    )
    parser.add_argument(
        "--no-settlements",
        action="store_true",
        help="Hide settlements in 3D view",
    )
    parser.add_argument(
        "--no-roads",
        action="store_true",
        help="Hide roads in 3D view",
    )
    parser.add_argument(
        "--enable-legend",
        action="store_true",
        help="Show legend in 3D view (disabled by default for cleaner visualization)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    print("🗺️  Generating fantasy map...")
    # Create map generator with custom parameters
    generator = MapGenerator(
        width=args.width,
        height=args.height,
        scale=40.0,  # Good scale for 3D visualization
        octaves=5,
        persistence=0.5,
        lacunarity=2.0,
        settlement_density=0.001,  # Fewer settlements for cleaner 3D view
        min_settlement_radius=3,
        max_settlement_radius=8,
        smoothing_iterations=2,
    )

    # Generate the map
    map_data = generator.generate()
    print("✨ Map generation complete!")
    print(f"   📐 Dimensions: {map_data.width} x {map_data.height}")
    print(f"   🏘️  Settlements: {len(map_data.settlements)}")
    print(f"   🛣️  Roads: {len(map_data.roads)}")

    print("\n🎨 Creating stunning 3D visualization...")
    # Create 3D visualization
    fig = plot_3d_map(
        map_data,
        enable_settlements=not args.no_settlements,
        enable_roads=not args.no_roads,
        enable_legend=args.enable_legend,
        colormap=args.colormap,
        elevation_scale=args.elevation_scale,
    )

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the 3D visualization
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"💾 3D visualization saved to: {args.output}")

    print("\n🎉 Success! Your fantasy world is now in 3D!")
    print("   ✨ Settlements are now visible above mountains with connecting lines!")
    print("   � Legend disabled by default for clean 3D visualization (use --enable-legend to show)")
    print("   �💡 Tip: Try different elevation scales and colormaps!")
    print("   🎮 You can rotate, zoom, and explore the 3D map interactively!")


if __name__ == "__main__":
    main()