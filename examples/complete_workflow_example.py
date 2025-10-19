#!/usr/bin/env python3
"""Complete MapGen workflow example demonstrating generation, saving, loading, and visualization."""

import os
from pathlib import Path

from mapgen import MapGenerator
from mapgen.map_data import MapData


def main() -> None:
    """Demonstrate the complete MapGen workflow."""
    print("🗺️  MapGen Complete Workflow Example")
    print("=" * 50)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Generate a map
    print("\n1️⃣  Generating fantasy map...")
    generator = MapGenerator(
        width=100,
        height=80,
        scale=60.0,
        octaves=6,
        sea_level=0.02,
        mountain_level=0.6,
        forest_threshold=0.15,
        settlement_density=0.0015,
    )

    map_data = generator.generate()
    print(f"   ✅ Generated {map_data.width}x{map_data.height} map")
    print(f"   📊 Terrain: {len(set(tile.name for row in map_data.grid for tile in row))} types")
    print(f"   🏘️  Settlements: {len(map_data.settlements) if map_data.settlements else 0}")
    print(f"   🛣️  Roads: {len(map_data.roads_graph.edges) if map_data.roads_graph else 0}")

    # Step 2: Save to JSON
    json_path = output_dir / "complete_workflow_map.json"
    print(f"\n2️⃣  Saving to JSON: {json_path}")
    map_data.save_to_json(str(json_path))
    json_size = os.path.getsize(json_path)
    print(f"   💾 JSON size: {json_size:,} bytes")

    # Step 3: Generate PNG from original map
    png_original_path = output_dir / "original_map.png"
    print(f"\n3️⃣  Generating PNG from original map...")
    fig_original = generator.plot(map_data)
    fig_original.savefig(png_original_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig_original.get_figure().clear()  # Free memory
    png_original_size = os.path.getsize(png_original_path)
    print(f"   🖼️  Original PNG size: {png_original_size:,} bytes")

    # Step 4: Load map from JSON
    print(f"\n4️⃣  Loading map from JSON...")
    loaded_map_data = MapData.load_from_json(str(json_path))
    print(f"   ✅ Loaded {loaded_map_data.width}x{loaded_map_data.height} map")
    print(f"   📊 Terrain types preserved: {len(set(tile.name for row in loaded_map_data.grid for tile in row))}")
    print(f"   🏘️  Settlements loaded: {len(loaded_map_data.settlements) if loaded_map_data.settlements else 0}")
    print(f"   🛣️  Roads loaded: {len(loaded_map_data.roads_graph.edges) if loaded_map_data.roads_graph else 0}")

    # Step 5: Generate PNG from loaded map
    png_loaded_path = output_dir / "loaded_map.png"
    print(f"\n5️⃣  Generating PNG from loaded map...")
    fig_loaded = generator.plot(loaded_map_data)
    fig_loaded.savefig(png_loaded_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig_loaded.get_figure().clear()  # Free memory
    png_loaded_size = os.path.getsize(png_loaded_path)
    print(f"   🖼️  Loaded PNG size: {png_loaded_size:,} bytes")

    # Step 6: Verify data integrity
    print(f"\n6️⃣  Verifying data integrity...")
    integrity_ok = True

    # Check dimensions
    if loaded_map_data.width != map_data.width or loaded_map_data.height != map_data.height:
        print("   ❌ Dimensions don't match!")
        integrity_ok = False

    # Check terrain
    terrain_mismatches = 0
    for y in range(min(map_data.height, loaded_map_data.height)):
        for x in range(min(map_data.width, loaded_map_data.width)):
            if map_data.get_terrain(x, y).name != loaded_map_data.get_terrain(x, y).name:
                terrain_mismatches += 1

    if terrain_mismatches > 0:
        print(f"   ❌ {terrain_mismatches} terrain mismatches!")
        integrity_ok = False

    # Check settlements
    if len(map_data.settlements or []) != len(loaded_map_data.settlements or []):
        print("   ❌ Settlement count doesn't match!")
        integrity_ok = False

    # Check roads
    if len(map_data.roads_graph.edges if map_data.roads_graph else []) != len(loaded_map_data.roads_graph.edges if loaded_map_data.roads_graph else []):
        print("   ❌ Road count doesn't match!")
        integrity_ok = False

    if integrity_ok:
        print("   ✅ All data preserved perfectly!")
    else:
        print("   ⚠️  Some data integrity issues detected")

    # Step 7: Summary
    print(f"\n🎉 Workflow completed successfully!")
    print(f"   📁 Files created:")
    print(f"      JSON: {json_path} ({json_size:,} bytes)")
    print(f"      Original PNG: {png_original_path} ({png_original_size:,} bytes)")
    print(f"      Loaded PNG: {png_loaded_path} ({png_loaded_size:,} bytes)")
    print(f"\n💡 The JSON file contains the complete map data:")
    print(f"   - Terrain grid ({map_data.width}x{map_data.height})")
    print(f"   - Noise map for visual shading")
    print(f"   - Elevation map for road generation")
    print(f"   - {len(map_data.settlements) if map_data.settlements else 0} settlements with names and positions")
    print(f"   - Road network with {len(map_data.roads_graph.edges) if map_data.roads_graph else 0} connections")
    print(f"\n🚀 You can now load any .json file and generate visualizations!")


if __name__ == "__main__":
    main()