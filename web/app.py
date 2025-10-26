"""
Streamlit web interface for MapGen.

Run with: streamlit run web/app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import os

from mapgen import generate_map, plot_map
from mapgen.map_data import MapData


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
            "Width", min_value=1, max_value=9999, value=150, help="Map width in tiles"
        )
    with col2:
        height = st.number_input(
            "Height", min_value=1, max_value=9999, value=100, help="Map height in tiles"
        )
    with col3:
        seed = st.number_input(
            "Seed", value=42, min_value=0, help="Random seed for reproducible results"
        )

    # Noise parameters
    st.sidebar.subheader("🌊 Noise Settings")
    scale = st.sidebar.number_input(
        "Scale", min_value=1, max_value=1000, value=50, help="Noise scale factor"
    )
    octaves = st.sidebar.number_input(
        "Octaves", min_value=1, max_value=20, value=6, help="Noise octaves"
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

        # Create the plot
        fig = plot_map(map_data)

        # Display the plot
        st.pyplot(fig, width="stretch")

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
