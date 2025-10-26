"""
Streamlit web interface for MapGen.

Run with: streamlit run web/app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from mapgen import generate_map, plot_map
from mapgen.map_data import MapData


def create_interactive_map(map_data: MapData) -> go.Figure:
    """Create an interactive Plotly map with zoom and pan controls."""
    # Create base terrain image
    rgb_values = np.zeros((map_data.height, map_data.width, 3))

    for y in range(map_data.height):
        for x in range(map_data.width):
            tile = map_data.get_terrain(x, y)
            # Compute shade factor based on elevation.
            shade_factor = 0.5 + 0.5 * map_data.get_elevation(x, y)
            # Generate shaded color.
            shaded_color = tuple(c * shade_factor for c in tile.color)
            # Convert to 0-255 range for Plotly
            rgb_values[y, x, :] = tuple(int(c * 255) for c in shaded_color)

    # Create the figure
    fig = go.Figure()

    # Add terrain as image
    fig.add_trace(go.Image(z=rgb_values.astype(np.uint8), hoverinfo="skip"))

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
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            autorange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        dragmode="pan",
        hovermode="closest",
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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="MapGen - Fantasy Map Generator", page_icon="🗺️", layout="wide"
    )

    st.title("🗺️ MapGen - Fantasy Map Generator")
    st.markdown(
        "Generate procedural fantasy maps with settlements, roads, and terrain."
    )

    # Sidebar with parameters
    st.sidebar.header("⚙️ Parameters")

    # Basic parameters
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        width = st.number_input(
            "Width",
            min_value=1,
            max_value=9999,
            value=240,
            help="Map width in tiles",
        )
    with col2:
        height = st.number_input(
            "Height",
            min_value=1,
            max_value=9999,
            value=120,
            help="Map height in tiles",
        )
    with col3:
        seed = st.number_input(
            "Seed",
            value=42,
            min_value=0,
            help="Random seed for reproducible results",
        )

    # Noise parameters
    st.sidebar.subheader("🌊 Noise Settings")
    scale = st.sidebar.number_input(
        "Scale",
        min_value=1,
        max_value=1000,
        value=50,
        help="Noise scale factor",
    )
    octaves = st.sidebar.number_input(
        "Octaves",
        min_value=1,
        max_value=20,
        value=6,
        help="Noise octaves",
    )
    persistence = st.sidebar.number_input(
        "Persistence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Noise persistence",
    )
    lacunarity = st.sidebar.number_input(
        "Lacunarity",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Noise lacunarity",
    )

    # Terrain thresholds
    st.sidebar.subheader("🏔️ Terrain Thresholds")
    sea_level = st.sidebar.number_input(
        "Sea Level",
        min_value=-1.0,
        max_value=1.0,
        value=-0.25,
        step=0.01,
        help="Elevation level for sea (controls land/sea ratio)",
    )

    # Other parameters
    st.sidebar.subheader("🏘️ Settlements & Roads")
    settlement_density = st.sidebar.number_input(
        "Settlement Density",
        min_value=0.0001,
        max_value=0.1,
        value=0.003,
        step=0.0001,
        format="%.4f",
        help="Settlement placement probability",
    )
    smoothing_iterations = st.sidebar.number_input(
        "Smoothing",
        min_value=0,
        max_value=50,
        value=5,
        help="Terrain smoothing iterations",
    )

    # Generation options
    st.sidebar.subheader("🎯 Options")
    generate_settlements = st.sidebar.checkbox("Generate Settlements", value=True)
    generate_roads = st.sidebar.checkbox("Generate Roads", value=True)
    generate_rivers = st.sidebar.checkbox("Generate Rivers", value=True)
    generate_vegetation = st.sidebar.checkbox("Generate Vegetation", value=True)

    # Display options
    st.sidebar.subheader("👁️ Display")
    show_zoom_controls = st.sidebar.checkbox(
        "Interactive Zoom",
        value=True,
        help="Enable Google Maps-style zoom and pan controls",
    )

    # Main content area
    if st.button("🎲 Generate Map", type="primary", width="stretch"):
        with st.spinner("Generating map..."):
            try:
                # Generate map directly with parameters
                map_data = generate_map(
                    width=width,
                    height=height,
                    scale=scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    sea_level=sea_level,
                    settlement_density=settlement_density,
                    smoothing_iterations=smoothing_iterations,
                    seed=seed,
                    enable_settlements=generate_settlements,
                    enable_roads=generate_roads,
                    enable_rivers=generate_rivers,
                    enable_vegetation=generate_vegetation,
                )

                # Store in session state
                st.session_state.map_data = map_data

                st.success("Map generated successfully!")

            except Exception as e:
                st.error(f"Error generating map: {str(e)}")
                return

    # Display map if available
    if "map_data" in st.session_state:
        map_data = st.session_state.map_data

        if show_zoom_controls:
            # Create interactive Plotly map
            fig = create_interactive_map(map_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create static matplotlib map
            fig = plot_map(map_data)
            st.pyplot(fig, use_container_width=True)

        # Map statistics
        st.subheader("📊 Map Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)

        with stats_col1:
            st.metric("Settlements", len(map_data.settlements))
            st.metric("Roads", len(map_data.roads))

        with stats_col2:
            st.metric("Water Routes", len(map_data.water_routes))
            total_road_tiles = sum(len(road.path) for road in map_data.roads)
            st.metric("Road Tiles", total_road_tiles)

        with stats_col3:
            total_water_tiles = sum(len(route.path) for route in map_data.water_routes)
            st.metric("Water Route Tiles", total_water_tiles)


if __name__ == "__main__":
    main()
