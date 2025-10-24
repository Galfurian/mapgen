# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Flora-based vegetation system with climate-driven placement
- New `flora.py` module for vegetation algorithms
- New `tile_collections.py` module for organizing tiles by placement method
- `TileCollections` dataclass with separate collections for base_terrain, vegetation, water_features, and infrastructure
- `get_default_tile_collections()` function for organized tile catalog
- Climate-driven forest placement based on rainfall and elevation
- Desert placement in arid regions
- Grassland placement in moderate climate conditions
- New `enable_vegetation` parameter in `generate_map()` to toggle vegetation system
- New `apply_base_terrain()` function in terrain module for base terrain assignment

### Changed

- Forests now placed using climate data (rainfall + elevation) instead of pure elevation
- Terrain generation now separated into distinct phases: base terrain → vegetation → water features → infrastructure
- `apply_terrain_features()` now focuses only on base terrain (sea, coast, plains, mountains)
- Map generation pipeline restructured for better separation of concerns
- Vegetation tiles no longer compete with base terrain via priority system

### Improved

- Better ecological accuracy with climate-based biome placement
- More realistic vegetation distribution following environmental rules
- Improved code organization with clear separation of placement phases
- Enhanced extensibility - easy to add new vegetation types or biomes
- Better performance with specialized algorithms for each placement type
- Fixed plains elevation range (0.0-0.5) to prevent coastal clustering of forests

## [0.1.0] - 2025-10-16

### Added

- Initial release of MapGen library
- Procedural terrain generation using Perlin noise
- Cellular automata terrain smoothing
- Settlement generation with random names
- Road network generation using A* pathfinding
- Visualization with matplotlib
- Comprehensive test suite
- Example scripts
- Modern Python packaging with pyproject.toml

### Features

- Customizable map parameters (size, noise settings, terrain thresholds)
- Modular architecture with separate modules for terrain, settlements, roads, and visualization
- Type hints and documentation
- Inspired by fantasy-map but completely reorganized and improved
