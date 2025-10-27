"""
Streamlit web interface for MapGen.

Run with: streamlit run web/app.py
"""

from typing import Any
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
import os
import math

from mapgen import generate_map
from mapgen.map_data import MapData


@st.cache_data
def create_interactive_map(
    map_data: MapData,
    selection: dict | None = None,
) -> go.Figure:
    """Create an interactive Plotly map with zoom and pan controls."""
    # Create base terrain image
    rgb_values = np.zeros((map_data.height, map_data.width, 3))
    points_x = []
    points_y = []
    point_colors = []

    for y in range(map_data.height):
        for x in range(map_data.width):
            # Store point coordinates.
            points_x.append(x)
            points_y.append(y)
            # Get the terrain tile.
            tile = map_data.get_terrain(x, y)
            # Compute shade factor based on elevation.
            shade_factor = 0.5 + 0.5 * map_data.get_elevation(x, y)
            # Generate shaded color.
            shaded_color = tuple(c * shade_factor for c in tile.color)
            # Convert to 0-255 range for Plotly
            rgb_values[y, x, :] = tuple(int(c * 255) for c in shaded_color)
            # Store point color for selection.
            point_colors.append(
                "rgb("
                f"{int(shaded_color[0] * 255) * 0.875},"
                f"{int(shaded_color[1] * 255) * 0.875},"
                f"{int(shaded_color[2] * 255) * 0.875})"
            )

    # Highlight selected tiles
    if selection and "point_indices" in selection:
        for idx in selection["point_indices"]:
            x = idx % map_data.width
            y = idx // map_data.width
            # Blend with red for highlight
            original = rgb_values[y, x]
            rgb_values[y, x] = (original * 0.85 + np.array([255, 0, 0]) * 0.15).astype(
                np.uint8
            )

    # Create the figure
    fig = go.Figure()

    # Add terrain as image
    fig.add_trace(go.Image(z=rgb_values.astype(np.uint8), hoverinfo="skip"))
    
    # Add selectable points at tile centers with tile colors.
    fig.add_trace(
        go.Scatter(
            x=points_x,
            y=points_y,
            mode="markers",
            marker=dict(color=point_colors, size=3, opacity=0.8),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add roads
    for road in map_data.roads:
        x_coords = [pos.x for pos in road.path]
        y_coords = [pos.y for pos in road.path]

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines+markers",
                line=dict(color="brown", width=2),
                marker=dict(color="brown", size=2),
                name="Roads",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add water routes
    for water_route in map_data.water_routes:
        x_coords = [pos.x for pos in water_route.path]
        y_coords = [pos.y for pos in water_route.path]

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(color="blue", size=2),
                name="Water Routes",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add settlements
    settlement_x = []
    settlement_y = []
    settlement_names = []
    settlement_colors = []

    for settlement in map_data.settlements:
        settlement_x.append(settlement.position.x)
        settlement_y.append(settlement.position.y)
        settlement_names.append(settlement.name)
        settlement_colors.append("blue" if settlement.is_harbor else "white")

    if settlement_x:
        fig.add_trace(
            go.Scatter(
                x=settlement_x,
                y=settlement_y,
                mode="markers+text",
                marker=dict(
                    color=settlement_colors,
                    size=[s.radius * 10 for s in map_data.settlements],
                    line=dict(color="black", width=1),
                ),
                text=settlement_names,
                textposition="top center",
                textfont=dict(size=10, color="white"),
                name="Settlements",
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Configure layout with proper zoom controls
    fig.update_layout(
        title="Interactive Fantasy Map",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            autorange=True,
            fixedrange=False,  # Allow horizontal panning/zooming
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            autorange=True,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=False,  # Allow vertical panning/zooming
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        dragmode="select",
        hovermode="closest",
        clickmode="event+select",
    )

    # Enable zoom controls - Plotly has built-in zoom controls that appear on hover
    # The zoom in/out buttons will appear in the top-right corner when hovering over the plot
    fig.update_layout(
        modebar=dict(
            add=[
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "zoom2d",
            ]
        )
    )

    return fig


STORAGE_FILE = ".web.storage"


def save_session_data():
    """Save relevant session state to file."""
    # Only save our widget settings, not internal state or large objects
    settings_to_save: dict[str, Any] = {}
    for key, value in st.session_state.items():
        settings_to_save[str(key)] = value
    try:
        with open(STORAGE_FILE, "w") as f:
            json.dump(settings_to_save, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save settings: {e}")


def load_session_data() -> None:
    """Load settings from file into session state."""
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE, "r") as f:
                st.session_state.update(json.load(f))
    except Exception:
        pass


def get_setting(key: str, default=None) -> Any:
    """Get a setting value from session state, setting default if not exists."""
    return st.session_state.get(key, default)


def get_widget_kwargs(key: str, default, **fixed_kwargs):
    """Build kwargs for a widget, conditionally including 'value' if key not in session_state."""
    kwargs = {"key": key, **fixed_kwargs}
    if key not in st.session_state:
        kwargs["value"] = default
    return kwargs


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="MapGen - Fantasy Map Generator", page_icon="🗺️", layout="wide"
    )

    st.title("🗺️ MapGen - Fantasy Map Generator")
    st.markdown(
        "Generate procedural fantasy maps with settlements, roads, and terrain."
    )

    # Load previous settings.
    load_session_data()

    # Sidebar with parameters
    st.sidebar.header("⚙️ Parameters")

    # Basic parameters
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        width = st.number_input(
            "Width",
            **get_widget_kwargs(
                "width",
                240,
                min_value=1,
                max_value=9999,
                help="Map width in tiles",
                on_change=save_session_data,
            ),
        )

    with col2:
        height = st.number_input(
            "Height",
            **get_widget_kwargs(
                "height",
                120,
                min_value=1,
                max_value=9999,
                help="Map height in tiles",
                on_change=save_session_data,
            ),
        )
    with col3:
        seed = st.number_input(
            "Seed",
            **get_widget_kwargs(
                "seed",
                42,
                min_value=0,
                help="Random seed for reproducible results",
                on_change=save_session_data,
            ),
        )

    # Noise parameters
    st.sidebar.subheader("🌊 Noise Settings")
    scale = st.sidebar.number_input(
        "Scale",
        **get_widget_kwargs(
            "scale",
            50,
            min_value=1,
            max_value=1000,
            help="Noise scale factor",
            on_change=save_session_data,
        ),
    )
    octaves = st.sidebar.number_input(
        "Octaves",
        **get_widget_kwargs(
            "octaves",
            6,
            min_value=1,
            max_value=20,
            help="Noise octaves",
            on_change=save_session_data,
        ),
    )
    persistence = st.sidebar.number_input(
        "Persistence",
        **get_widget_kwargs(
            "persistence",
            0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Noise persistence",
            on_change=save_session_data,
        ),
    )
    lacunarity = st.sidebar.number_input(
        "Lacunarity",
        **get_widget_kwargs(
            "lacunarity",
            2.0,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            help="Noise lacunarity",
            on_change=save_session_data,
        ),
    )

    # Terrain thresholds
    st.sidebar.subheader("🏔️ Terrain Thresholds")
    sea_level = st.sidebar.number_input(
        "Sea Level",
        **get_widget_kwargs(
            "sea_level",
            -0.25,
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            help="Elevation level for sea (controls land/sea ratio)",
            on_change=save_session_data,
        ),
    )

    # Other parameters
    st.sidebar.subheader("🏘️ Settlements & Roads")
    settlement_density = st.sidebar.number_input(
        "Settlement Density",
        **get_widget_kwargs(
            "settlement_density",
            0.003,
            min_value=0.0001,
            max_value=0.1,
            step=0.0001,
            format="%.4f",
            help="Settlement placement probability",
            on_change=save_session_data,
        ),
    )
    smoothing_iterations = st.sidebar.number_input(
        "Smoothing",
        **get_widget_kwargs(
            "smoothing_iterations",
            5,
            min_value=0,
            max_value=50,
            help="Terrain smoothing iterations",
            on_change=save_session_data,
        ),
    )

    # Generation options
    st.sidebar.subheader("🎯 Options")
    generate_settlements = st.sidebar.checkbox(
        "Generate Settlements",
        **get_widget_kwargs(
            "generate_settlements",
            True,
            on_change=save_session_data,
        ),
    )
    generate_roads = st.sidebar.checkbox(
        "Generate Roads",
        **get_widget_kwargs(
            "generate_roads",
            True,
            on_change=save_session_data,
        ),
    )
    generate_rivers = st.sidebar.checkbox(
        "Generate Rivers",
        **get_widget_kwargs(
            "generate_rivers",
            True,
            on_change=save_session_data,
        ),
    )
    generate_vegetation = st.sidebar.checkbox(
        "Generate Vegetation",
        **get_widget_kwargs(
            "generate_vegetation",
            True,
            on_change=save_session_data,
        ),
    )

    st.sidebar.subheader("🍪 Settings")
    if st.sidebar.button("Clear Saved Settings"):
        try:
            if os.path.exists(STORAGE_FILE):
                os.remove(STORAGE_FILE)
            st.rerun()
        except Exception as e:
            st.error(f"Could not clear settings: {e}")

    st.session_state.map_data = get_setting("map_data", None)

    # Main content area
    if st.button("🎲 Generate Map", type="primary", width="stretch"):
        with st.spinner("Generating map..."):
            try:
                # Generate map directly with parameters
                map_data = generate_map(
                    width=int(width or 240),
                    height=int(height or 120),
                    scale=scale,
                    octaves=int(octaves or 6),
                    persistence=float(persistence or 0.5),
                    lacunarity=float(lacunarity or 2.0),
                    sea_level=float(sea_level or -0.25),
                    settlement_density=float(settlement_density or 0.003),
                    smoothing_iterations=int(smoothing_iterations or 5),
                    seed=int(seed or 42),
                    enable_settlements=bool(generate_settlements),
                    enable_roads=bool(generate_roads),
                    enable_rivers=bool(generate_rivers),
                    enable_vegetation=bool(generate_vegetation),
                )

                # Store in session state
                st.session_state.map_data = map_data.model_dump()

                save_session_data()

            except Exception as e:
                st.error(f"Error generating map: {str(e)}")
                return

    # Create interactive Plotly map.
    if st.session_state.map_data is not None:
        map_data: MapData = MapData.model_validate(st.session_state.map_data)
        # Get the current selection from session state.
        selection = st.session_state.get("my_chart", {}).get("selection", None)
        # First, create the interactive map figure.
        fig = create_interactive_map(map_data, selection)
        # Use full container width and enable scroll zoom for map zooming.
        st.plotly_chart(
            fig,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "sendDataToCloud",
                    "lasso2d",
                    "zoom",
                    "resetAxes",
                ],
                "selectionMode": "box",
            },
            on_select="rerun",
            key="my_chart",
        )


if __name__ == "__main__":
    main()
