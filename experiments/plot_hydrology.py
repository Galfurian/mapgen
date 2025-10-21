"""
Script to plot hydrological features: accumulation, lakes, and river sources.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
import os


def plot_hydrology(map_data, out_path):
    """
    Plot accumulation map with lakes and river sources overlaid.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot accumulation
    accumulation = np.array(map_data['accumulation_map'])
    im = ax.imshow(np.log1p(accumulation), cmap="Blues", interpolation="nearest", alpha=0.7)
    ax.set_title("Hydrological Map: Accumulation, Lakes, and River Sources")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot lakes
    for lake in map_data.get('lakes', []):
        center_x, center_y = lake['center']
        size = lake['size']
        # Plot lake as circle
        circle = Circle((center_x, center_y), radius=np.sqrt(size) / 2, color='cyan', alpha=0.5, fill=True)
        ax.add_patch(circle)
        # Mark center
        ax.plot(center_x, center_y, 'co', markersize=5)

    # Find river sources (assuming rivers are in map_data, but since not saved, simulate)
    # For now, just mark lake centers as potential sources
    for lake in map_data.get('lakes', []):
        center_x, center_y = lake['center']
        ax.plot(center_x, center_y, 'r*', markersize=10, label='River Source' if lake == map_data.get('lakes', [])[0] else "")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log(Accumulation + 1)")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot hydrological features from map data.")
    parser.add_argument("input_json", help="Path to map_data JSON file")
    parser.add_argument("--output", default=None, help="Output image path (optional)")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.input_json))[0]
        out_path = f"output/hydrology_{base}.png"

    plot_hydrology(data, out_path)
    print(f"Hydrology plot saved to {out_path}")


if __name__ == "__main__":
    main()