"""
tests/test_planning.py
----------------------
Tests for CostmapBuilder, A* planner, DWA planner, and Simulator.
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.planning import CostmapBuilder, CameraConfig, astar, DWAPlanner, DWAConfig
from backend.sim import Simulator

def test_costmap_builder():
    cam = CameraConfig(img_w=512, img_h=384)
    builder = CostmapBuilder(cam, vehicle_radius=0.1)
    
    # Fake group mask: bottom half is traversable (1), top half is unknown/ignore (0)
    group_mask = np.zeros((384, 512), dtype=np.uint8)
    group_mask[192:, :] = 1
    
    uncertainty = np.zeros((384, 512), dtype=np.float32)

    # Fake depth map: a flat ground plane at y = 0.5m
    # In camera frame, y_c = (v - cy) * z / fy -> z = y_c_world * fy / (v - cy)
    depth_map = np.full((384, 512), 1.0, dtype=np.float32)
    # Give the bottom center a depth of 1.0m (which corresponds to row 90 in a 100x100 grid with 0.1m cells)
    # Wait, distance 1.0m means cell row = 99 - 1.0/0.1 = 89
    
    costmap = builder.update(group_mask, uncertainty, depth_map)
    
    assert costmap.shape == (100, 100)
    # The area 1.0m ahead of the robot (cell 89, 50) is seen as traversable (cost=1).
    # Since cost starts at 50 and fuses with decay 0.85: 50 * 0.85 + 1 * 0.15 = 42.65
    assert costmap[89, 50] < 45.0
    
def test_astar_planner():
    costmap = np.ones((100, 100), dtype=np.float64)
    # Create a wall
    costmap[50, 20:80] = 100.0
    
    start = (90, 50)
    goal = (10, 50)
    
    path = astar(costmap, start, goal, cost_threshold=80.0)
    
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal
    
    # Path should not go through the wall
    for r, c in path:
        assert costmap[r, c] < 80.0

def test_dwa_planner():
    costmap = np.ones((100, 100), dtype=np.float64)
    dwa = DWAPlanner(DWAConfig())
    
    # Robot at (90, 50) facing UP (-row)
    cmd = dwa.plan(costmap, 90, 50, 0.0, 80, 50)
    
    # Should move forward (v > 0)
    assert cmd.v > 0.0
    assert not cmd.emergency_stop
    
    # Emergency stop test
    costmap[85:90, 48:53] = 100.0
    cmd_estop = dwa.plan(costmap, 90, 50, 0.0, 80, 50)
    assert cmd_estop.emergency_stop
    
def test_simulator():
    sim = Simulator(seed=42)
    assert not sim.is_done
    assert not sim.had_collision
    
    # Spawn obstacle right in front of the robot
    sim.spawn_obstacle(sim.robot.x + 1.0, sim.robot.y, radius=0.5)
    
    # Move forward
    sim.step(1.0, 0.0) # 0.1s step, moves 0.1m
    
    # Get costmap
    cmap, unc = sim.get_costmap()
    assert cmap.shape == (100, 100)
    assert unc.shape == (100, 100)
