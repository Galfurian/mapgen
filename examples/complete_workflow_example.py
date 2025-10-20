#!/usr/bin/env python3
"""Complete MapGen workflow example demonstrating generation, saving, loading, and visualization."""

import os
from pathlib import Path

from mapgen import MapGenerator, visualization


def main() -> None:
    """Demonstrate the complete MapGen workflow."""
    print("🗺️  MapGen Complete Workflow Example")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Generate a map
    print("Generating fantasy map...")
    generator = MapGenerator(
        width=150,
        height=100,
        padding=2,
        scale=50.0,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        smoothing_iterations=5,
        settlement_density=0.002,
        min_settlement_radius=0.5,
        max_settlement_radius=1.0,
        seed=42,
    )

    map_data = generator.generate()

    if map_data is None:
        print("   ❌ Map generation failed!")
        return

    print(f"   ✅ Generated {generator.width}x{generator.height} map")
    print(f"   📊 Terrain: {len(map_data.tiles)} types")
    print(f"   🏘️  Settlements: {len(map_data.settlements)}")
    print(f"   🛣️  Roads: {len(map_data.roads)}")

    # Generate PNG from original map.
    print("Generating PNG from original map...")
    map_path_original = output_dir / "original_map.png"
    fig_original = visualization.plot_map(map_data)
    fig_original.savefig(
        map_path_original,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    fig_original.get_figure().clear()  # Free memory
    png_original_size = os.path.getsize(map_path_original)
    print(f"   🖼️  Original PNG size: {png_original_size:,} bytes")

    # Save the map to JSON
    map_json_path = output_dir / "complete_workflow_map.json"
    print(f"Saving to JSON: {map_json_path}")
    map_data.save_to_json(str(map_json_path))
    json_size = os.path.getsize(map_json_path)
    print(f"   💾 JSON size: {json_size:,} bytes")


if __name__ == "__main__":
    main()
