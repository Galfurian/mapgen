# MapGen

A Python library for procedural generation of fantasy maps using Perlin noise and cellular automata.

## Features

- **Procedural Terrain Generation**: Generate diverse landscapes with mountains, forests, plains, and water using Perlin noise.
- **Settlement Placement**: Automatically place settlements in suitable terrain locations.
- **Road Network Generation**: Create realistic road networks connecting settlements using A* pathfinding.
- **Visualization**: Plot generated maps with terrain colors, contour lines, settlements, and roads.
- **Customizable Parameters**: Fine-tune generation parameters for different map styles.

## Installation

```bash
pip install mapgen
```

For development:

```bash
git clone https://github.com/Galfurian/mapgen.git
cd mapgen
pip install -e ".[dev]"
```

## Quick Start

```python
from mapgen import MapGenerator

# Create a generator with default parameters
generator = MapGenerator(width=150, height=100)

# Generate the map
map_data = generator.generate()

# Plot the map
fig = generator.plot(map_data)
fig.show()
```

## Saving and Loading Maps

```python
from mapgen.map_data import MapData

# Save generated map to JSON
map_data.save_to_json("my_map.json")

# Load map from JSON
loaded_map = MapData.load_from_json("my_map.json")

# Plot loaded map
fig = generator.plot(loaded_map)
```

## Parameters

The `MapGenerator` class accepts various parameters to customize map generation:

- `width`, `height`: Map dimensions
- `wall_countdown`: Amount of digging for initial terrain carving
- `scale`, `octaves`, `persistence`, `lacunarity`: Perlin noise parameters
- `sea_level`, `mountain_level`, `forest_threshold`: Terrain type thresholds
- `smoothing_iterations`: Terrain smoothing passes
- `settlement_density`: Probability of settlement placement
- `min_settlement_radius`, `max_settlement_radius`: Settlement size range

## API Reference

### MapGenerator

Main class for generating maps.

#### Methods

- `generate() -> MapData`: Generate and return the complete map data
- `plot(map_data: MapData) -> Figure`: Return a matplotlib figure of the given map data

### MapData

Container for all map data including terrain, settlements, roads, and visualization data.

#### MapData Methods

- `save_to_json(filepath: str)`: Save map data to a JSON file
- `load_from_json(filepath: str) -> MapData`: Load map data from a JSON file (class method)

## Web Interface

MapGen includes an interactive web interface built with Streamlit:

```bash
# Install with web interface
pip install -e ".[web]"

# Run the web interface
streamlit run web/app.py
```

The web interface provides:

- Interactive parameter tuning with sliders
- Real-time map generation and visualization
- Save/load functionality for maps
- Map statistics and metrics

## Development

Run tests:

```bash
pytest
```

Format code:

```bash
black src/mapgen tests
isort src/mapgen tests
```

## License

MIT License. See LICENSE file.

## Inspiration

This project is inspired by [fantasy-map](https://github.com/Dmukherjeetextiles/fantasy-map) but has been completely reorganized and improved for better modularity and maintainability.
