# MapGen Architecture Restructure: Flora-Based Vegetation System

## Overview

This document outlines the restructuring of MapGen to implement a **flora-based vegetation system** that separates base terrain placement from vegetation algorithms, enabling more realistic and sophisticated biome generation.

## Current Architecture Problems

1. **Mixed Concerns**: Base terrain (elevation-driven) and vegetation (climate-driven) are handled in the same system
2. **Limited Realism**: Forests compete with base terrain via priority system rather than following ecological rules
3. **Poor Extensibility**: Adding new vegetation types requires modifying core terrain logic
4. **Monolithic Tile List**: All tiles in one collection makes organization difficult

## Target Architecture

### Phase 1: Base Terrain (terrain.py)

- **Input**: Elevation map only
- **Output**: Sea, coast, plains, mountains
- **Algorithm**: Elevation-based tile assignment
- **Tiles**: `base_terrain_tiles` collection

### Phase 2: Flora/Vegetation (flora.py) *[NEW]*

- **Input**: Base terrain + climate maps (rainfall, temperature)
- **Output**: Forests, grasslands, deserts, tundra
- **Algorithm**: Climate-driven placement algorithms
- **Tiles**: `vegetation_tiles` collection

### Phase 3: Water Features (rivers.py, lakes.py)

- **Input**: Terrain + hydrological data
- **Output**: Rivers, lakes, wetlands
- **Algorithm**: Flow-based algorithms
- **Tiles**: `water_tiles` collection

### Phase 4: Infrastructure (roads.py, settlements.py)

- **Input**: Complete terrain
- **Output**: Roads, settlements, structures
- **Algorithm**: Pathfinding and placement algorithms
- **Tiles**: `infrastructure_tiles` collection

## Implementation Plan

### Step 1: Create Tile Collections Structure

**File**: `src/mapgen/tile_collections.py`

```python
from dataclasses import dataclass
from typing import List
from .map_data import Tile

@dataclass
class TileCollections:
    """Organized collections of tiles by placement method and type."""

    # Base terrain tiles (elevation-driven)
    base_terrain: List[Tile] = field(default_factory=list)

    # Vegetation tiles (climate-based)
    vegetation: List[Tile] = field(default_factory=list)

    # Water feature tiles (algorithm-based)
    water_features: List[Tile] = field(default_factory=list)

    # Infrastructure tiles (algorithm-based)
    infrastructure: List[Tile] = field(default_factory=list)

    @property
    def all_tiles(self) -> List[Tile]:
        """Get all tiles in a single list for backward compatibility."""
        return (self.base_terrain + self.vegetation +
                self.water_features + self.infrastructure)
```

### Step 2: Create Flora Module

**File**: `src/mapgen/flora.py`

```python
"""Flora and vegetation placement algorithms."""

import logging
from .map_data import MapData, Tile

logger = logging.getLogger(__name__)

def place_vegetation(
    map_data: MapData,
    vegetation_tiles: List[Tile],
) -> None:
    """
    Place vegetation based on climate conditions.

    This runs after base terrain placement and can override
    base terrain tiles with appropriate vegetation.
    """
    # Implementation will include:
    # - Forest placement based on rainfall + elevation
    # - Desert placement in dry areas
    # - Grassland placement in moderate conditions
    # - Tundra placement in cold areas (future)
    pass

def _place_forests(map_data: MapData, forest_tiles: List[Tile]) -> None:
    """Place forests using seed-based growth algorithm."""
    pass

def _place_deserts(map_data: MapData, desert_tiles: List[Tile]) -> None:
    """Place deserts in arid regions."""
    pass
```

### Step 3: Refactor Map Generator

**File**: `src/mapgen/map_generator.py`

#### Changes

1. Replace `get_default_tiles()` with `get_default_tile_collections()`
2. Update generation pipeline:

```python
def generate_map(...) -> MapData:
    # ... existing setup code ...

    # Phase 1: Generate base terrain
    terrain.apply_base_terrain(map_data, tile_collections.base_terrain)

    # Phase 2: Place vegetation
    flora.place_vegetation(map_data, tile_collections.vegetation)

    # Phase 3: Generate water features
    if enable_rivers:
        rivers.generate_rivers(map_data)
    # ... lakes, etc.

    # Phase 4: Generate infrastructure
    if enable_settlements:
        settlements.generate_settlements(map_data)
    if enable_roads:
        roads.generate_roads(map_data)

    return map_data
```

### Step 4: Update Terrain Module

**File**: `src/mapgen/terrain.py`

#### Rename and refactor

- `apply_terrain_features()` → `apply_base_terrain()`
- Remove vegetation tiles from processing
- Focus only on elevation-based base terrain

```python
def apply_base_terrain(
    map_data: MapData,
    base_terrain_tiles: List[Tile],
) -> None:
    """Apply base terrain tiles based on elevation only."""
    # Process only base terrain tiles (sea, coast, plains, mountains)
    # No vegetation placement here
```

### Step 5: Update Tile Definitions

**File**: `src/mapgen/map_generator.py`

#### Reorganize `get_default_tile_collections()`

```python
def get_default_tile_collections() -> TileCollections:
    collections = TileCollections()

    # Base terrain tiles
    collections.base_terrain.extend([
        Tile(name="sea", ...),
        Tile(name="coast", ...),
        Tile(name="plains", ...),
        Tile(name="mountain", ...),
    ])

    # Vegetation tiles
    collections.vegetation.extend([
        Tile(name="forest", ...),
        Tile(name="grassland", ...),
        Tile(name="desert", ...),
    ])

    # Water features
    collections.water_features.extend([
        Tile(name="river", ...),
        Tile(name="lake", ...),
    ])

    # Infrastructure
    collections.infrastructure.extend([
        # Roads, walls, etc.
    ])

    return collections
```

### Step 6: Update MapData Class

**File**: `src/mapgen/map_data.py`

#### Add tile collections support

- Update `MapData` to accept `TileCollections` instead of flat tile list
- Maintain backward compatibility with `all_tiles` property
- Update tile indexing and lookup methods

### Step 7: Migrate Forest Placement

1. Remove forest tile from base terrain
2. Add forest tile to vegetation collection
3. Implement forest placement algorithm in `flora.py`
4. Update terrain assignment to exclude forests

### Step 8: Update Tests

**Files**: `tests/*.py`

1. Update test tile creation to use new collections
2. Add tests for flora placement algorithms
3. Ensure backward compatibility where needed

### Step 9: Update Examples and Documentation

1. Update example scripts to use new API
2. Update README with new architecture explanation
3. Add documentation for flora algorithms

## Migration Benefits

### Realism Improvements

- **Climate-Driven Vegetation**: Forests grow based on rainfall, not elevation priority
- **Biome Diversity**: Different vegetation types can coexist in same elevation ranges
- **Ecological Accuracy**: Vegetation follows real environmental rules

### Architectural Improvements

- **Separation of Concerns**: Each placement phase has clear responsibilities
- **Modularity**: Easy to add new vegetation types or algorithms
- **Extensibility**: New biome types don't require core terrain changes
- **Better Performance**: Specialized algorithms for each placement type

### Performance Benefits

- **Specialized Algorithms**: Each placement type uses appropriate algorithms
- **Parallel Processing**: Phases could potentially run in parallel
- **Optimized Collections**: Smaller, focused tile lists improve lookup speed

## Risk Assessment

### High Risk

- **Breaking Changes**: Complete restructure affects all users
- **Complex Migration**: Many files need coordinated changes
- **Performance Impact**: Additional processing phases

### Mitigation Strategies

- **Incremental Migration**: Implement one collection at a time
- **Backward Compatibility**: Maintain old API during transition
- **Comprehensive Testing**: Test each phase independently
- **Feature Flags**: Allow gradual rollout of new features

## Success Criteria

1. **Functionality**: All existing maps generate correctly
2. **Realism**: Vegetation placement follows climate rules
3. **Performance**: No significant slowdown vs current system
4. **Maintainability**: Code is more organized and extensible
5. **Testability**: Each component can be tested independently

## Timeline Estimate

- **Phase 1** (Tile Collections): 2-3 days
- **Phase 2** (Flora Module): 3-4 days
- **Phase 3** (Migration): 4-5 days
- **Phase 4** (Testing & Polish): 2-3 days
- **Total**: 11-15 days

## Alternative Approaches

### Option A: Minimal Change

- Keep current system but improve forest placement logic
- **Pros**: Faster implementation, backward compatible
- **Cons**: Still mixed concerns, limited extensibility

### Option B: Hybrid Approach

- Implement flora system but keep forests in terrain for now
- **Pros**: Gradual migration, less risky
- **Cons**: Still not fully clean architecture

### Option C: Full Restructure (Recommended)

- Complete separation as outlined above
- **Pros**: Clean architecture, maximum realism
- **Cons**: Most complex implementation

## Decision

**Recommendation**: Proceed with full restructure (Option C) for long-term maintainability and realism benefits. The architectural improvements justify the implementation complexity.
