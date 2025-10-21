"""
Script to compute and plot water accumulation (runoff) from rainfall and elevation.
Optimized for speed using a queue-based topological order (no recursion).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import argparse
import os


def compute_accumulation(elevation, rainfall):
    height, width = elevation.shape
    flow_to = np.full((height, width, 2), -1, dtype=int)
    in_degree = np.zeros((height, width), dtype=int)
    for y in range(height):
        for x in range(width):
            min_elev = elevation[y, x]
            min_pos = (x, y)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if elevation[ny, nx] < min_elev:
                        min_elev = elevation[ny, nx]
                        min_pos = (nx, ny)
            flow_to[y, x] = min_pos
    for y in range(height):
        for x in range(width):
            tx, ty = flow_to[y, x]
            if (tx, ty) != (x, y):
                in_degree[ty, tx] += 1
    queue = deque()
    for y in range(height):
        for x in range(width):
            if in_degree[y, x] == 0:
                queue.append((x, y))
    accumulation = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            accumulation[y, x] = rainfall[y, x]
    while queue:
        x, y = queue.popleft()
        tx, ty = flow_to[y, x]
        if (tx, ty) != (x, y):
            accumulation[ty, tx] += accumulation[y, x]
            in_degree[ty, tx] -= 1
            if in_degree[ty, tx] == 0:
                queue.append((tx, ty))
    return accumulation


def plot_accumulation(accumulation, out_path):
    plt.figure(figsize=(10, 7))
    plt.imshow(np.log1p(accumulation), cmap="Blues", interpolation="nearest")
    plt.title("Log Water Accumulation (Runoff)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="log(Accumulation + 1)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot water accumulation (runoff) from rainfall and elevation."
    )
    parser.add_argument("input_json", help="Path to map_data JSON file")
    parser.add_argument("--output", default=None, help="Output image path (optional)")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)
    elevation = np.array(data["elevation_map"])
    rainfall = np.array(data["rainfall_map"])
    accumulation = compute_accumulation(elevation, rainfall)

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.input_json))[0]
        out_path = f"output/accumulation_map_{base}.png"
    plot_accumulation(accumulation, out_path)

    print(
        f"Accumulation stats: min={accumulation.min():.3f}, max={accumulation.max():.3f}, mean={accumulation.mean():.3f}"
    )


if __name__ == "__main__":
    main()
