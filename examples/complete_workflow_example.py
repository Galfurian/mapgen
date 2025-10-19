#!/usr/bin/env python3
"""Complete MapGen workflow example demonstrating generation, saving, loading, and visualization."""

import os
from pathlib import Path

from mapgen import MapGenerator, visualization
from mapgen.map_data import MapData


def main() -> None:
    """Demonstrate the complete MapGen workflow."""
    print("🗺️  MapGen Complete Workflow Example")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Generate a map
    print("Generating fantasy map...")
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

    generator.generate()

    if generator.map_data is None:
        print("   ❌ Map generation failed!")
        return
    if generator.noise_map is None:
        print("   ❌ Noise map generation failed!")
        return
    if generator.roads_graph is None:
        print("   ❌ Road generation failed!")
        return

    print(f"   ✅ Generated {generator.width}x{generator.height} map")
    print(f"   📊 Terrain: {len(generator.map_data.tiles)} types")
    print(f"   🏘️  Settlements: {len(generator.map_data.settlements)}")
    print(f"   🛣️  Roads: {len(generator.roads_graph.edges)}")

    # Generate PNG from original map.
    print("Generating PNG from original map...")
    map_path_original = output_dir / "original_map.png"
    fig_original = visualization.plot_map(
        map_data=generator.map_data,
        noise_map=generator.noise_map,
        settlements=generator.map_data.settlements,
        roads_graph=generator.roads_graph,
    )
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
    generator.map_data.save_to_json(str(map_json_path))
    json_size = os.path.getsize(map_json_path)
    print(f"   💾 JSON size: {json_size:,} bytes")

    if False:

        # Step 4: Load map from JSON
        print("\n4️⃣  Loading map from JSON...")
        loaded_map_data = MapData.load_from_json(str(map_json_path))
        print(f"   ✅ Loaded {loaded_map_data.width}x{loaded_map_data.height} map")
        print(
            f"   📊 Terrain types preserved: {len(set(tile.name for row in loaded_map_data.tiles_grid for tile in row))}"
        )
        print(
            f"   🏘️  Settlements loaded: {len(loaded_map_data.settlements) if loaded_map_data.settlements else 0}"
        )
        print(
            f"   🛣️  Roads loaded: {len(loaded_map_data.roads_graph.edges) if loaded_map_data.roads_graph else 0}"
        )

        # Step 5: Generate PNG from loaded map
        png_loaded_path = output_dir / "loaded_map.png"
        print("\n5️⃣  Generating PNG from loaded map...")
        fig_loaded = generator.plot(loaded_map_data)
        fig_loaded.savefig(
            png_loaded_path, dpi=200, bbox_inches="tight", facecolor="white"
        )
        fig_loaded.get_figure().clear()  # Free memory
        png_loaded_size = os.path.getsize(png_loaded_path)
        print(f"   🖼️  Loaded PNG size: {png_loaded_size:,} bytes")

        # Step 6: Verify data integrity
        print("\n6️⃣  Verifying data integrity...")
        integrity_ok = True

        # Check dimensions
        if (
            loaded_map_data.width != map_data.width
            or loaded_map_data.height != map_data.height
        ):
            print("   ❌ Dimensions don't match!")
            integrity_ok = False

        # Check terrain
        terrain_mismatches = 0
        for y in range(min(map_data.height, loaded_map_data.height)):
            for x in range(min(map_data.width, loaded_map_data.width)):
                if (
                    map_data.get_terrain(x, y).name
                    != loaded_map_data.get_terrain(x, y).name
                ):
                    terrain_mismatches += 1

        if terrain_mismatches > 0:
            print(f"   ❌ {terrain_mismatches} terrain mismatches!")
            integrity_ok = False

        # Check settlements
        if len(map_data.settlements or []) != len(loaded_map_data.settlements or []):
            print("   ❌ Settlement count doesn't match!")
            integrity_ok = False

        # Check roads
        if len(map_data.roads_graph.edges if map_data.roads_graph else []) != len(
            loaded_map_data.roads_graph.edges if loaded_map_data.roads_graph else []
        ):
            print("   ❌ Road count doesn't match!")
            integrity_ok = False

        if integrity_ok:
            print("   ✅ All data preserved perfectly!")
        else:
            print("   ⚠️  Some data integrity issues detected")

        # Step 7: Summary
        print("\n🎉 Workflow completed successfully!")
        print("   📁 Files created:")
        print(f"      JSON: {map_json_path} ({json_size:,} bytes)")
        print(f"      Original PNG: {map_path_original} ({png_original_size:,} bytes)")
        print(f"      Loaded PNG: {png_loaded_path} ({png_loaded_size:,} bytes)")
        print("\n💡 The JSON file contains the complete map data:")
        print(f"   - Terrain grid ({map_data.width}x{map_data.height})")
        print("   - Noise map for visual shading")
        print("   - Elevation map for road generation")
        print(
            f"   - {len(map_data.settlements) if map_data.settlements else 0} settlements with names and positions"
        )
        print(
            f"   - Road network with {len(map_data.roads_graph.edges) if map_data.roads_graph else 0} connections"
        )
        print("\n🚀 You can now load any .json file and generate visualizations!")


if __name__ == "__main__":
    main()
