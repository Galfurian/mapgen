# MapGen Web Interface

Interactive web interface for MapGen using Streamlit.

## Features

- **Interactive Parameter Tuning**: Adjust all map generation parameters with sliders
- **Real-time Map Generation**: Generate maps instantly with live visualization
- **Save/Load Maps**: Export maps as JSON or PNG, load existing maps
- **Map Statistics**: View settlement counts, road lengths, and other metrics

## Installation

1. Install the main MapGen package with web interface:

```bash
pip install -e ".[web]"
```

Or install web dependencies separately:

```bash
pip install streamlit>=1.28.0
```

## Usage

Run the web interface:

```bash
streamlit run web/app.py
```

Then open your browser to `http://localhost:8501`

## Interface Overview

- **Left Sidebar**: Parameter controls for map generation
- **Main Area**: Map visualization and statistics
- **Right Panel**: Save/load controls and instructions

## Parameters

- **Basic**: Width, height, random seed
- **Noise**: Scale, octaves, persistence, lacunarity for terrain generation
- **Terrain**: Sea level, mountain level, forest threshold
- **Settlements**: Density and placement settings
- **Options**: Enable/disable settlements, roads, water routes

## Saving and Loading

- **JSON Export**: Saves complete map data including terrain, settlements, roads
- **PNG Export**: Saves the current visualization as an image
- **JSON Import**: Load previously saved maps for viewing or further processing
