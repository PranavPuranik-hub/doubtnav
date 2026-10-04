import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.planning import CostmapBuilder, CameraConfig

def generate_synthetic_inputs(img_w=512, img_h=384):
    """
    Generate a synthetic group mask, uncertainty map, and depth map.
    Simulates a scene with:
      - A road (traversable)
      - A person (obstacle) in the middle
      - Sky/unknown in the top half
    """
    # 0: ignore, 1: traversable, 2: risky, 3: obstacle, 4: non_traversable
    group_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    uncertainty = np.zeros((img_h, img_w), dtype=np.float32)
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)

    # Intrinsics
    fx = 0.9 * img_w
    fy = fx
    cx = img_w / 2.0
    cy = img_h / 2.0
    camera_height = 0.5  # metres

    for v in range(img_h):
        for u in range(img_w):
            if v < cy:
                # Sky (ignore group 0)
                group_mask[v, u] = 0
                uncertainty[v, u] = 0.8
                depth_map[v, u] = 20.0  # Far away
            else:
                # Ground plane
                # y_c = (v - cy) * z / fy = camera_height => z = camera_height * fy / (v - cy)
                z = camera_height * fy / (max(1, v - cy))
                
                x = (u - cx) * z / fx
                
                # Default to traversable road
                group_mask[v, u] = 1
                depth_map[v, u] = z
                
                # Place an obstacle (e.g., person) at z=3.0, x=0.5
                if 2.8 < z < 3.2 and 0.2 < x < 0.8:
                    group_mask[v, u] = 3
                    depth_map[v, u] = 3.0  # Standing upright, const depth
                    uncertainty[v, u] = 0.1

    return group_mask, uncertainty, depth_map

def main():
    print("Generating synthetic inputs...")
    cam = CameraConfig(img_w=512, img_h=384)
    group_mask, uncertainty, depth_map = generate_synthetic_inputs(cam.img_w, cam.img_h)

    print("Building costmap...")
    builder = CostmapBuilder(cam, vehicle_radius=0.3, decay=0.0) # decay 0.0 to see instant results
    
    # Run update a few times to simulate temporal fusion if decay > 0, 
    # but with decay 0.0 it updates immediately.
    costmap = builder.update(group_mask, uncertainty, depth_map)

    # Plot
    print("Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    ax = axes[0, 0]
    im = ax.imshow(group_mask, cmap='tab10', vmin=0, vmax=4)
    ax.set_title("Group Mask (Camera View)")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(depth_map, cmap='plasma', vmin=0, vmax=10)
    ax.set_title("Depth Map (metres)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(uncertainty, cmap='Reds', vmin=0, vmax=1)
    ax.set_title("Uncertainty")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    # In BEV, robot is at bottom center (row=99, col=50)
    # Extent: left=-5m, right=5m, bottom=0m, top=10m
    im = ax.imshow(costmap, cmap='viridis', origin='upper', 
                   extent=[-5, 5, 0, 10], vmin=0, vmax=100)
    ax.set_title("BEV Costmap (Inflated)")
    ax.plot(0, 0, 'r^', markersize=12, label="Robot")
    ax.legend()
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    os.makedirs("data/outputs", exist_ok=True)
    out_path = "data/outputs/costmap_viz.png"
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
