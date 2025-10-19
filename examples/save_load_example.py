#!/usr/bin/env python3
"""Example script demonstrating how to save and load maps in JSON format."""

import os
from pathlib import Path

from mapgen import MapGenerator


def main() -> None:
    """Demonstrate saving and loading maps in lossless JSON format."""
    print("MapGen JSON Save/Load Example")
    print("=" * 40)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Generate a map
    print("Generating map...")
    generator = MapGenerator(width=50, height=50)
    generator.generate()

    assert generator.map_data is not None
    print(f"Generated map: {generator.map_data.width}x{generator.map_data.height}")
    print(f"Settlements: {len(generator.settlements) if generator.settlements else 0}")

    # Save the map to JSON
    json_path = output_dir / "example_map.json"
    print(f"\nSaving map to: {json_path}")
    generator.map_data.save_to_json(str(json_path))

    # Get file size
    file_size = os.path.getsize(json_path)
    print(f"File size: {file_size:,} bytes")

    # Load the map back
    print("\nLoading map from JSON...")
    from mapgen.map_data import MapData
    loaded_map = MapData.load_from_json(str(json_path))

    print(f"Loaded map: {loaded_map.width}x{loaded_map.height}")

    # Verify data integrity
    print("\nVerifying data integrity...")
    mismatches = 0
    for y in range(min(generator.map_data.height, loaded_map.height)):
        for x in range(min(generator.map_data.width, loaded_map.width)):
            orig_tile = generator.map_data.get_terrain(x, y)
            load_tile = loaded_map.get_terrain(x, y)

            # Compare key properties
            if (orig_tile.name != load_tile.name or
                orig_tile.walkable != load_tile.walkable or
                orig_tile.symbol != load_tile.symbol):
                mismatches += 1

    if mismatches == 0:
        print("✅ All data preserved perfectly!")
    else:
        print(f"⚠️  Found {mismatches} mismatches")

    # Show a sample of the JSON structure
    print("\nJSON structure preview:")
    with open(json_path, 'r') as f:
        lines = f.readlines()[:20]  # First 20 lines
        for i, line in enumerate(lines, 1):
            print(f"{i:2d}: {line.rstrip()}")
            if i >= 50:  # Show first 10 lines, then ellipsis
                print("    ...")
                break

    print(f"\nFull JSON saved to: {json_path}")
    print("You can open this file in any text editor or JSON viewer!")


if __name__ == "__main__":
    main()