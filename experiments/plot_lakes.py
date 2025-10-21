"""
Script to detect and plot lakes (reservoirs) in a generated map.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Load map data
with open("output/map_data_seed_345.json") as f:
    data = json.load(f)

tiles = data["tiles"]
grid = data["grid"]
height = len(grid)
width = len(grid[0])

# Identify water tile indices (fresh, still water)
water_indices = set()
for i, tile in enumerate(tiles):
    if tile.get("is_water") and not tile.get("is_salt_water") and not tile.get("is_flowing_water"):
        water_indices.add(i)

# Flood fill to find lakes (connected regions of still fresh water)
lake_map = np.full((height, width), -1, dtype=int)
lake_id = 0
lakes = []
visited = set()
for y in range(height):
    for x in range(width):
        if grid[y][x] in water_indices and (x, y) not in visited:
            # Start a new lake
            queue = deque()
            queue.append((x, y))
            lake_pixels = []
            while queue:
                cx, cy = queue.popleft()
                if (cx, cy) in visited:
                    continue
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                if grid[cy][cx] not in water_indices:
                    continue
                visited.add((cx, cy))
                lake_map[cy, cx] = lake_id
                lake_pixels.append((cx, cy))
                # 4-way neighbors
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                        queue.append((nx, ny))
            if lake_pixels:
                lakes.append(lake_pixels)
                lake_id += 1

# Plot lakes
plt.figure(figsize=(10, 7))
lake_img = np.full((height, width), -1, dtype=int)
for lid, pixels in enumerate(lakes):
    for x, y in pixels:
        lake_img[y, x] = lid
plt.imshow(lake_img, cmap="tab20", interpolation="nearest")
plt.title(f"Detected Lakes (Reservoirs): {len(lakes)} found")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Lake ID")
plt.tight_layout()
plt.savefig("output/lakes_map_seed_345.png")
plt.show()

print(f"Detected {len(lakes)} lakes.")
for i, lake in enumerate(lakes):
    print(f"Lake {i}: {len(lake)} tiles")
