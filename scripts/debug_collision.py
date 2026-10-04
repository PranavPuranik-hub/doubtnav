import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.sim import Simulator
from backend.planning import DWAPlanner, DWAConfig, astar, _GRID_SIZE, _CELL_RES

def debug_seed(seed):
    sim = Simulator(seed=seed)
    dwa = DWAPlanner(DWAConfig(
        max_v=1.2,
        max_omega=1.5,
        emergency_cost=9999.0,
        heading_weight=1.5,
        velocity_weight=2.5,
        cost_weight=1.0,
    ))
    
    prev_cost = None
    global_path = []
    
    for step in range(1500):
        raw_cost, _ = sim.get_costmap()
        costmap = raw_cost.copy()
        if prev_cost is not None:
            costmap = 0.7 * costmap + 0.3 * prev_cost
        prev_cost = costmap.copy()

        if step % 10 == 0 or not global_path:
            small = sim.world.reshape(60, 10, 60, 10).max(axis=(1, 3))
            sr, sc = int(sim.robot.x), int(sim.robot.y)
            gr, gc = int(sim.goal[0]), int(sim.goal[1])
            cells = astar(small, (sr, sc), (gr, gc))
            global_path = [(r + 0.5, c + 0.5) for r, c in cells]

        wp_world = sim.goal
        for wx, wy in global_path:
            if np.hypot(wx - sim.robot.x, wy - sim.robot.y) > 4.0:
                wp_world = (wx, wy)
                break
                
        dx = wp_world[0] - sim.robot.x
        dy = wp_world[1] - sim.robot.y
        cos_y = np.cos(sim.robot.yaw)
        sin_y = np.sin(sim.robot.yaw)
        fwd  =  dx * cos_y + dy * sin_y
        lat  = -dx * sin_y + dy * cos_y
        
        lr = _GRID_SIZE - 1 - int(round(fwd / _CELL_RES))
        lc = _GRID_SIZE // 2 + int(round(lat / _CELL_RES))
        wp_local = (max(0, min(_GRID_SIZE - 1, lr)), max(0, min(_GRID_SIZE - 1, lc)))

        cmd = dwa.plan(costmap, _GRID_SIZE - 1, _GRID_SIZE // 2, 0.0, wp_local[0], wp_local[1], speed_cap=1.0)
        
        res = sim.step(cmd.v, cmd.omega)
        if res.collision:
            print(f"Seed {seed} collided at step {step}!")
            # Render and save
            h, w = costmap.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[costmap < 5] = [90, 80, 40]
            img[(costmap >= 5) & (costmap < 80)] = [80, 80, 80]
            img[costmap >= 80] = [70, 70, 220]
            
            # Robot is at bottom center (99, 50)
            cv2.circle(img, (50, 99), 3, (0, 255, 0), -1)
            # Planned trajectory?
            cv2.imwrite(f"collision_seed_{seed}.png", img)
            break

if __name__ == "__main__":
    for s in range(4):
        debug_seed(s)
