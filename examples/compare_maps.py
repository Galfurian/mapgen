#!/usr/bin/env python3
"""Script to compare two maps side by side."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

from mapgen import visualization
from mapgen.map_data import MapData


def main() -> None:
    """Compare two maps side by side."""
    parser = argparse.ArgumentParser(description="Compare two maps side by side")
    parser.add_argument(
        "map1",
        type=str,
        help="Path to the first map JSON file"
    )
    parser.add_argument(
        "map2",
        type=str,
        help="Path to the second map JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison.png",
        help="Output filename for the comparison image (default: comparison.png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for the output image (default: 200)"
    )

    args = parser.parse_args()

    # Check if files exist
    map1_path = Path(args.map1)
    map2_path = Path(args.map2)

    if not map1_path.exists():
        print(f"❌ Error: First map file '{map1_path}' does not exist")
        return

    if not map2_path.exists():
        print(f"❌ Error: Second map file '{map2_path}' does not exist")
        return

    # Load maps
    print("📂 Loading maps...")
    try:
        map1_data = MapData.load_from_json(str(map1_path))
        print(f"   ✅ Loaded first map: {map1_data.width}x{map1_data.height}")
        print(f"      Settlements: {len(map1_data.settlements) if map1_data.settlements else 0}")
        print(f"      Roads: {len(map1_data.roads)}")
    except Exception as e:
        print(f"❌ Error loading first map: {e}")
        return

    try:
        map2_data = MapData.load_from_json(str(map2_path))
        print(f"   ✅ Loaded second map: {map2_data.width}x{map2_data.height}")
        print(f"      Settlements: {len(map2_data.settlements) if map2_data.settlements else 0}")
        print(f"      Roads: {len(map2_data.roads)}")
    except Exception as e:
        print(f"❌ Error loading second map: {e}")
        return

    # Create side-by-side comparison
    print("🎨 Generating comparison visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot first map
    visualization.plot_base_terrain(ax1, map1_data)
    visualization.plot_contour_lines(ax1, map1_data)
    visualization.plot_roads(ax1, map1_data)
    visualization.plot_settlements(ax1, map1_data)
    ax1.set_title(f"Map 1: {map1_path.name}", fontsize=14, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot second map
    visualization.plot_base_terrain(ax2, map2_data)
    visualization.plot_contour_lines(ax2, map2_data)
    visualization.plot_roads(ax2, map2_data)
    visualization.plot_settlements(ax2, map2_data)
    ax2.set_title(f"Map 2: {map2_path.name}", fontsize=14, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Adjust layout
    plt.tight_layout()

    # Save the comparison
    output_path = Path(args.output)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)  # Free memory

    file_size = os.path.getsize(output_path)
    print(f"✅ Comparison saved to: {output_path} ({file_size:,} bytes)")


if __name__ == "__main__":
    main()
