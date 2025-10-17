#!/usr/bin/env python3
"""Basic example of using MapGen to generate a fantasy map."""

from mapgen import MapGenerator


def main() -> None:
    """Generate and display a fantasy map."""
    print("Generating fantasy map...")

    # Create generator with custom parameters
    generator = MapGenerator(
        width=150,
        height=100,
        scale=50.0,
        octaves=6,
        sea_level=0.03,
        mountain_level=0.5,
        forest_threshold=0.1,
        settlement_density=0.002,
    )

    # Generate the map
    generator.generate()

    assert generator.settlements is not None
    assert generator.roads_graph is not None

    print(f"Generated map with {len(generator.settlements)} settlements")
    print(f"Road network has {len(generator.roads_graph.edges)} roads")

    # Plot the map
    fig = generator.plot()

    # Save to file (since we can't display in headless environment)
    fig.savefig("fantasy_map.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Map saved as 'fantasy_map.png'")

    # You can also access the raw data:
    # generator.map - Map object with terrain types
    # generator.noise_map - numpy array of noise values
    # generator.elevation_map - numpy array of elevation
    # generator.settlements - list of settlement dicts
    # generator.roads_graph - networkx graph of roads


if __name__ == "__main__":
    main()
