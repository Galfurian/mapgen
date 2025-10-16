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
generator.generate()

# Plot the map
fig = generator.plot()
fig.show()
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

- `generate()`: Generate the complete map
- `plot()`: Return a matplotlib figure of the generated map

## Examples

See the `examples/` directory for more detailed usage examples.

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
