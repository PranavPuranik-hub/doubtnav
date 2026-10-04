"""
planning.py – CostmapBuilder, A* global planner, DWA local planner.

CostmapBuilder
--------------
  Projects segmentation group_mask + uncertainty onto a 10 m × 10 m local grid
  (0.1 m cells → 100 × 100) using ground-plane inverse-perspective mapping.
  Cell cost = class_cost + 20 * uncertainty.  Obstacles inflated by vehicle
  radius.  Temporal fusion via exponential decay.

A* global planner
-----------------
  8-connected grid search returning a list of (row, col) waypoints.

DWA local planner
-----------------
  Samples (v, ω) candidates, simulates forward, selects the best trajectory
  considering cost, heading-to-goal, and velocity.  Scales by confidence
  speed_cap.  Emergency-stops when forward cost exceeds a threshold.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRID_SIZE   = 100            # cells  (10 m / 0.1 m)
_CELL_RES    = 0.1            # metres per cell
_GRID_EXTENT = 10.0           # metres  (side length)

CLASS_COSTS = {
    0: 1,      # ignore  → treat as traversable-ish
    1: 1,      # traversable
    2: 8,      # risky
    3: 100,    # obstacle
    4: 100,    # non_traversable
}

_UNKNOWN_COST = 50.0          # cells never observed
_DECAY        = 0.85          # exponential-decay factor per frame


# ---------------------------------------------------------------------------
# CostmapBuilder
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Camera parameters for inverse-perspective mapping."""
    height: float = 0.5        # metres above ground
    pitch:  float = 0.3        # radians downward tilt (positive = looking down)
    hfov:   float = 1.2        # horizontal field-of-view (radians)
    img_w:  int   = 512
    img_h:  int   = 384


class CostmapBuilder:
    """
    Maintains a 100 × 100 local costmap (10 m × 10 m, 0.1 m cells).

    The robot is always at cell (99, 50) – bottom-centre – looking "up" (row 0).
    Row index decreases as distance from the robot increases.

    Parameters
    ----------
    cam          : CameraConfig for intrinsics
    vehicle_radius : inflation radius in metres
    decay        : exponential-decay factor for temporal fusion
    """

    def __init__(
        self,
        cam: CameraConfig | None = None,
        vehicle_radius: float = 0.3,
        decay: float = _DECAY,
    ):
        self.cam = cam or CameraConfig()
        self.vehicle_radius = vehicle_radius
        self.decay = decay

        self._inflate_cells = max(1, int(math.ceil(vehicle_radius / _CELL_RES)))

        # Costmap: high initial cost (unknown)
        self.costmap = np.full((_GRID_SIZE, _GRID_SIZE), _UNKNOWN_COST, dtype=np.float64)

        # Precompute (u, v) grid for vectorised depth projection
        v, u = np.indices((self.cam.img_h, self.cam.img_w))
        self.u = u
        self.v = v
        
        # Intrinsics
        self.fx = 0.9 * self.cam.img_w
        self.fy = self.fx
        self.cx = self.cam.img_w / 2.0
        self.cy = self.cam.img_h / 2.0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        group_mask: np.ndarray,
        uncertainty: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Project a new frame's segmentation into the costmap using a depth map
        (or synthetic ground-plane depth if not provided), with temporal fusion
        and obstacle inflation.

        Parameters
        ----------
        group_mask  : (H, W) uint8, values in {0..4}
        uncertainty : (H, W) float 0..1
        depth_map   : (H, W) float, distance in metres (optional)

        Returns the updated costmap (100 × 100 float64).
        """
        # Decay existing costmap towards unknown
        self.costmap = self.decay * self.costmap + (1.0 - self.decay) * _UNKNOWN_COST

        if depth_map is None:
            # Synthesize ground plane depth
            v_diff = np.maximum(1.0, self.v - self.cy)
            depth_map = (self.cam.height * self.fy / v_diff).astype(np.float32)
            # Far clip for sky/above horizon
            depth_map[self.v < self.cy] = 50.0

        # Resize masks if they don't match the configured dimensions
        if group_mask.shape != (self.cam.img_h, self.cam.img_w):
            group_mask = cv2.resize(
                group_mask, (self.cam.img_w, self.cam.img_h),
                interpolation=cv2.INTER_NEAREST
            )
            uncertainty = cv2.resize(
                uncertainty.astype(np.float32),
                (self.cam.img_w, self.cam.img_h),
                interpolation=cv2.INTER_LINEAR
            )
            depth_map = cv2.resize(
                depth_map.astype(np.float32),
                (self.cam.img_w, self.cam.img_h),
                interpolation=cv2.INTER_LINEAR
            )

        # Vectorised projection from depth
        # Z is forward, X is lateral (right positive)
        z = depth_map
        x = (self.u - self.cx) * z / self.fx

        # Map to grid
        cols = np.round(x / _CELL_RES).astype(np.int32) + _GRID_SIZE // 2
        rows = _GRID_SIZE - 1 - np.round(z / _CELL_RES).astype(np.int32)

        # Filter points within the local grid
        valid = (z >= 0.05) & (z <= _GRID_EXTENT) & \
                (rows >= 0) & (rows < _GRID_SIZE) & \
                (cols >= 0) & (cols < _GRID_SIZE)

        rows = rows[valid]
        cols = cols[valid]
        classes = group_mask[valid]
        unc = uncertainty[valid]

        # Build a fresh observation costmap
        obs = np.full((_GRID_SIZE, _GRID_SIZE), np.nan, dtype=np.float64)
        obs_count = np.zeros((_GRID_SIZE, _GRID_SIZE), dtype=np.float64)

        # Compute per-pixel cost
        class_cost_arr = np.array([CLASS_COSTS.get(c, 50) for c in classes], dtype=np.float64)
        pixel_cost = class_cost_arr + 20.0 * unc

        # Average into grid cells
        np.add.at(obs_count, (rows, cols), 1.0)
        obs_filled = np.zeros_like(obs)
        np.add.at(obs_filled, (rows, cols), pixel_cost)

        observed = obs_count > 0
        obs_filled[observed] /= obs_count[observed]

        # Fuse: observed cells override decayed value
        fused = (
            self.decay * self.costmap[observed] +
            (1.0 - self.decay) * obs_filled[observed]
        )
        self.costmap[observed] = np.maximum(fused, obs_filled[observed])

        # Inflate obstacles
        self._inflate()

        return self.costmap.copy()

    # ------------------------------------------------------------------
    def _inflate(self):
        """Dilate high-cost cells (≥80) by vehicle radius."""
        obstacle_mask = (self.costmap >= 80).astype(np.uint8)
        kernel_size = 2 * self._inflate_cells + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        inflated = cv2.dilate(obstacle_mask, kernel)
        # Set newly inflated cells to a high cost
        new_obstacles = (inflated > 0) & (self.costmap < 80)
        self.costmap[new_obstacles] = 90.0

    # ------------------------------------------------------------------
    def update_from_raw(
        self,
        raw_costmap: np.ndarray,
    ) -> np.ndarray:
        """
        Directly fuse a pre-computed 100 × 100 costmap (used by the simulator).
        """
        fused = self.decay * self.costmap + (1.0 - self.decay) * raw_costmap
        self.costmap = np.maximum(fused, raw_costmap)
        self._inflate()
        return self.costmap.copy()

    def reset(self):
        self.costmap = np.full((_GRID_SIZE, _GRID_SIZE), _UNKNOWN_COST, dtype=np.float64)


# ---------------------------------------------------------------------------
# A* global planner
# ---------------------------------------------------------------------------

def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(
    costmap: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cost_threshold: float = 80.0,
) -> List[Tuple[int, int]]:
    """
    A* on an 8-connected grid.

    Parameters
    ----------
    costmap        : (N, N) float array – cell traversal cost
    start, goal    : (row, col) tuples
    cost_threshold : cells with cost ≥ this are impassable

    Returns
    -------
    List of (row, col) from start to goal, or empty list if no path.
    """
    rows, cols = costmap.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return []
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return []
    if costmap[start[0], start[1]] >= cost_threshold:
        return []
    if costmap[goal[0], goal[1]] >= cost_threshold:
        return []

    open_set: list = []
    heapq.heappush(open_set, (0.0, start))
    came_from: dict = {}
    g_score: dict = {start: 0.0}
    closed_set = set()

    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]
    diag_cost = math.sqrt(2)

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc in neighbours:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if costmap[nr, nc] >= cost_threshold:
                continue

            move_cost = diag_cost if (dr != 0 and dc != 0) else 1.0
            tent_g = g_score[current] + move_cost * costmap[nr, nc]
            nkey = (nr, nc)

            if tent_g < g_score.get(nkey, float('inf')):
                g_score[nkey] = tent_g
                f_score = tent_g + _heuristic(nkey, goal)
                came_from[nkey] = current
                heapq.heappush(open_set, (f_score, nkey))

    return []  # no path


# ---------------------------------------------------------------------------
# DWA local planner
# ---------------------------------------------------------------------------

@dataclass
class DWAConfig:
    """Tuning knobs for the Dynamic Window Approach planner."""
    max_v:          float = 1.0       # m/s
    max_omega:      float = 1.5       # rad/s
    v_samples:      int   = 11
    omega_samples:  int   = 21
    dt:             float = 0.1       # simulation timestep (s)
    predict_time:   float = 1.5       # lookahead (s)
    heading_weight: float = 1.0
    velocity_weight:float = 0.5
    cost_weight:    float = 2.0
    emergency_cost: float = 70.0      # forward-cost threshold → e-stop


@dataclass
class DWAOutput:
    v:     float
    omega: float
    emergency_stop: bool = False


class DWAPlanner:
    """
    Dynamic Window Approach local planner.

    Given a costmap, robot pose, and a local goal, samples (v, ω) pairs,
    simulates differential-drive trajectories, and returns the best command
    scaled by the confidence speed_cap.
    """

    def __init__(self, cfg: DWAConfig | None = None):
        self.cfg = cfg or DWAConfig()

    # ------------------------------------------------------------------
    def plan(
        self,
        costmap: np.ndarray,
        robot_row: int,
        robot_col: int,
        robot_yaw: float,
        goal_row: int,
        goal_col: int,
        speed_cap: float = 1.0,
        raw_costmap: Optional[np.ndarray] = None,
    ) -> DWAOutput:
        """
        Returns (v, omega) command.

        costmap     : 100 × 100 local grid
        robot_row/col : current cell
        robot_yaw   : radians (0 = up in grid, i.e. row-decreasing)
        goal_row/col : local waypoint cell
        speed_cap   : confidence multiplier (1.0 / 0.6 / 0.3)
        raw_costmap : raw costmap without uncertainty, for e-stop check
        """
        cfg = self.cfg
        max_v = cfg.max_v * speed_cap
        max_w = cfg.max_omega

        # ── Emergency-stop check ───────────────────────────────────────
        eval_costmap = raw_costmap if raw_costmap is not None else costmap
        fwd_cost = self._forward_cost(eval_costmap, robot_row, robot_col, robot_yaw)
        if fwd_cost > cfg.emergency_cost:
            return DWAOutput(v=0.0, omega=0.0, emergency_stop=True)

        # ── Sample dynamic window ──────────────────────────────────────
        # Never return v=0. For CAUTIOUS/DEAD_RECKONING, enforce min 0.3 * speed_cap * max_v
        min_v = 0.3 * speed_cap * cfg.max_v if speed_cap < 1.0 else 0.1 * cfg.max_v
        
        vs = np.linspace(min_v, max_v, cfg.v_samples)
        ws = np.linspace(-max_w, max_w, cfg.omega_samples)

        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0

        for v in vs:
            for w in ws:
                score = self._score_trajectory(
                    costmap, robot_row, robot_col, robot_yaw,
                    goal_row, goal_col, v, w,
                )
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        return DWAOutput(v=float(best_v), omega=float(best_w))

    # ------------------------------------------------------------------
    def _score_trajectory(
        self, costmap, r, c, yaw, gr, gc, v, w,
    ) -> float:
        cfg = self.cfg
        steps = int(cfg.predict_time / cfg.dt)
        cr, cc, cyaw = float(r), float(c), float(yaw)
        total_cost = 0.0

        for _ in range(steps):
            cyaw += w * cfg.dt
            # In grid: row decreases when moving "forward" (yaw=0 → up)
            cr -= v * cfg.dt * math.cos(cyaw) / _CELL_RES
            cc += v * cfg.dt * math.sin(cyaw) / _CELL_RES

            ir, ic = int(round(cr)), int(round(cc))
            if not (0 <= ir < _GRID_SIZE and 0 <= ic < _GRID_SIZE):
                return -1e6   # out of bounds
            cell_cost = costmap[ir, ic]
            if cell_cost >= 80:
                return -1e6   # collision
            total_cost += cell_cost

        # Final heading error
        heading_err = abs(math.atan2(gc - cc, -(gr - cr)) - cyaw)
        heading_err = min(heading_err, 2 * math.pi - heading_err)

        heading_score = math.pi - heading_err
        velocity_score = v
        cost_score = -total_cost / max(steps, 1)

        return (
            cfg.heading_weight  * heading_score +
            cfg.velocity_weight * velocity_score +
            cfg.cost_weight     * cost_score
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _forward_cost(costmap, r, c, yaw, look_cells: int = 5) -> float:
        """Average cost of cells directly ahead."""
        costs = []
        for i in range(1, look_cells + 1):
            nr = r - int(round(i * math.cos(yaw)))
            nc = c + int(round(i * math.sin(yaw)))
            if 0 <= nr < _GRID_SIZE and 0 <= nc < _GRID_SIZE:
                costs.append(costmap[nr, nc])
            else:
                costs.append(100.0)
        return float(np.mean(costs)) if costs else 100.0
