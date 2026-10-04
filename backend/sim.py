"""
sim.py – Procedural top-down 60 × 60 m simulator.

World
-----
  A 60 × 60 m arena (0.1 m cells → 600 × 600 grid) with procedurally placed
  rocks, trees, a ditch (non-traversable water band), a start A and goal B.

Robot
-----
  Differential-drive kinematics: x += v·cos(θ)·dt, y += v·sin(θ)·dt, θ += ω·dt.

Simulated camera
----------------
  Extracts a 10 × 10 m (100 × 100 cell) window centred on the robot and
  oriented by its heading.  Adds Gaussian noise and marks cells beyond a
  "visibility radius" as unknown (high uncertainty).

API
---
  spawn_obstacle(x, y, radius)  – inject a surprise rock at runtime.
  step(v, omega)                – advance one timestep.
  get_costmap()                 – simulated camera observation.
  is_collision()                – True if the robot cell overlaps an obstacle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# World constants
# ---------------------------------------------------------------------------

WORLD_SIZE      = 60.0     # metres
CELL_RES        = 0.1      # metres per cell
WORLD_CELLS     = int(WORLD_SIZE / CELL_RES)  # 600

LOCAL_SIZE      = 100      # 10 m local window (cells)
LOCAL_EXTENT    = 10.0     # metres

# Cost values in the ground-truth world grid
COST_FREE       =   1.0
COST_RISKY      =   8.0
COST_OBSTACLE   = 100.0
COST_WATER      = 100.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    x:   float = 5.0       # metres
    y:   float = 5.0
    yaw: float = 0.0       # radians (0 = +X direction)


@dataclass
class SimResult:
    done:       bool  = False      # reached goal
    collision:  bool  = False
    dist_to_goal: float = float('inf')
    steps:      int   = 0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    Procedural 2-D world simulator.

    Parameters
    ----------
    seed           : RNG seed for procedural generation
    start          : (x, y) in metres
    goal           : (x, y) in metres
    dt             : simulation timestep (s)
    goal_radius    : metres – how close counts as "arrived"
    max_steps      : safety limit
    robot_radius   : metres – for collision checking
    visibility     : metres – simulated camera visibility radius
    noise_std      : standard deviation of cost noise
    """

    def __init__(
        self,
        seed: int = 0,
        start: Tuple[float, float] = (5.0, 5.0),
        goal:  Tuple[float, float] = (55.0, 55.0),
        dt: float = 0.1,
        goal_radius: float = 1.5,
        max_steps: int = 1500,
        robot_radius: float = 0.3,
        visibility: float = 8.0,
        noise_std: float = 2.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.goal = goal
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.robot_radius = robot_radius
        self.visibility = visibility
        self.noise_std = noise_std

        self.robot = RobotState(x=start[0], y=start[1], yaw=math.atan2(
            goal[1] - start[1], goal[0] - start[0]
        ))

        self._steps = 0
        self._collision = False

        # Build world grid
        self.world = np.full((WORLD_CELLS, WORLD_CELLS), COST_FREE, dtype=np.float64)
        self._generate_world()

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------
    def _generate_world(self):
        rng = self.rng

        # Boundary walls (2 m thick)
        wall_cells = int(2.0 / CELL_RES)
        self.world[:wall_cells, :] = COST_OBSTACLE
        self.world[-wall_cells:, :] = COST_OBSTACLE
        self.world[:, :wall_cells] = COST_OBSTACLE
        self.world[:, -wall_cells:] = COST_OBSTACLE

        # Ditch (water band) – horizontal, somewhere in the middle third
        ditch_row = int(rng.uniform(20, 40) / CELL_RES)
        ditch_width = int(rng.uniform(1.5, 3.0) / CELL_RES)
        # Leave a gap for a crossing
        gap_col_start = int(rng.uniform(15, 45) / CELL_RES)
        gap_width = int(rng.uniform(4.0, 8.0) / CELL_RES)

        self.world[ditch_row:ditch_row + ditch_width, :] = COST_WATER
        self.world[
            ditch_row:ditch_row + ditch_width,
            gap_col_start:gap_col_start + gap_width
        ] = COST_FREE

        # Rocks (circular obstacles)
        n_rocks = rng.integers(8, 16)
        for _ in range(n_rocks):
            cx = rng.uniform(8, 52)
            cy = rng.uniform(8, 52)
            r  = rng.uniform(0.5, 2.0)
            self._place_circle(cx, cy, r, COST_OBSTACLE)

        # Trees (smaller circles, with risky border)
        n_trees = rng.integers(6, 12)
        for _ in range(n_trees):
            cx = rng.uniform(8, 52)
            cy = rng.uniform(8, 52)
            r  = rng.uniform(0.3, 1.0)
            self._place_circle(cx, cy, r + 0.3, COST_RISKY)  # risky zone
            self._place_circle(cx, cy, r, COST_OBSTACLE)      # trunk

        # Clear start and goal areas
        self._clear_area(self.robot.x, self.robot.y, 3.0)
        self._clear_area(self.goal[0], self.goal[1], 3.0)

    def _place_circle(self, cx: float, cy: float, r: float, cost: float):
        cr, cc = int(cx / CELL_RES), int(cy / CELL_RES)
        rc = int(r / CELL_RES)
        for dr in range(-rc, rc + 1):
            for dc in range(-rc, rc + 1):
                if dr * dr + dc * dc <= rc * rc:
                    rr, cc2 = cr + dr, cc + dc
                    if 0 <= rr < WORLD_CELLS and 0 <= cc2 < WORLD_CELLS:
                        self.world[rr, cc2] = max(self.world[rr, cc2], cost)

    def _clear_area(self, cx: float, cy: float, r: float):
        cr, cc = int(cx / CELL_RES), int(cy / CELL_RES)
        rc = int(r / CELL_RES)
        for dr in range(-rc, rc + 1):
            for dc in range(-rc, rc + 1):
                if dr * dr + dc * dc <= rc * rc:
                    rr, cc2 = cr + dr, cc + dc
                    if 0 <= rr < WORLD_CELLS and 0 <= cc2 < WORLD_CELLS:
                        self.world[rr, cc2] = COST_FREE

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------
    def spawn_obstacle(self, x: float, y: float, radius: float = 1.0):
        """Inject a surprise obstacle at (x, y)."""
        self._place_circle(x, y, radius, COST_OBSTACLE)

    def step(self, v: float, omega: float) -> SimResult:
        """Advance the robot one timestep with differential-drive kinematics."""
        self._steps += 1

        # Kinematics
        self.robot.yaw += omega * self.dt
        self.robot.x  += v * math.cos(self.robot.yaw) * self.dt
        self.robot.y  += v * math.sin(self.robot.yaw) * self.dt

        # Clamp to world
        self.robot.x = max(0.3, min(WORLD_SIZE - 0.3, self.robot.x))
        self.robot.y = max(0.3, min(WORLD_SIZE - 0.3, self.robot.y))

        # Collision check
        if self._check_collision():
            self._collision = True

        # Goal check
        dist = math.hypot(self.robot.x - self.goal[0],
                          self.robot.y - self.goal[1])

        done = dist < self.goal_radius or self._steps >= self.max_steps

        return SimResult(
            done=done,
            collision=self._collision,
            dist_to_goal=dist,
            steps=self._steps,
        )

    def _check_collision(self) -> bool:
        cr = int(self.robot.x / CELL_RES)
        cc = int(self.robot.y / CELL_RES)
        rc = int(self.robot_radius / CELL_RES)
        for dr in range(-rc, rc + 1):
            for dc in range(-rc, rc + 1):
                if dr * dr + dc * dc <= rc * rc:
                    rr, cc2 = cr + dr, cc + dc
                    if 0 <= rr < WORLD_CELLS and 0 <= cc2 < WORLD_CELLS:
                        if self.world[rr, cc2] >= COST_OBSTACLE:
                            return True
        return False

    # ------------------------------------------------------------------
    # Simulated camera costmap
    # ------------------------------------------------------------------
    def get_costmap(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a 100 × 100 local costmap + uncertainty map as seen by a
        simulated forward-facing "camera".
        """
        local = np.full((LOCAL_SIZE, LOCAL_SIZE), 50.0, dtype=np.float64)
        unc   = np.ones((LOCAL_SIZE, LOCAL_SIZE), dtype=np.float64)

        cos_y = math.cos(self.robot.yaw)
        sin_y = math.sin(self.robot.yaw)
        
        # Precompute noise for the whole grid
        noise_grid = self.rng.normal(0, self.noise_std, (LOCAL_SIZE, LOCAL_SIZE))

        # Vectorized coordinate computation
        lr_idx = np.arange(LOCAL_SIZE)[:, None]
        lc_idx = np.arange(LOCAL_SIZE)[None, :]
        
        fwd = (LOCAL_SIZE - 1 - lr_idx) * CELL_RES
        lat = (lc_idx - LOCAL_SIZE // 2) * CELL_RES
        
        wx = self.robot.x + fwd * cos_y - lat * sin_y
        wy = self.robot.y + fwd * sin_y + lat * cos_y
        
        wr = (wx / CELL_RES).astype(np.int32)
        wc = (wy / CELL_RES).astype(np.int32)
        
        dist = np.hypot(fwd, lat)
        
        valid = (wr >= 0) & (wr < WORLD_CELLS) & (wc >= 0) & (wc < WORLD_CELLS) & (dist < self.visibility)
        
        wr_valid = wr[valid]
        wc_valid = wc[valid]
        
        gt_cost = self.world[wr_valid, wc_valid]
        local[valid] = np.maximum(1.0, gt_cost + noise_grid[valid])
        unc[valid] = np.minimum(1.0, 0.05 + 0.1 * (dist[valid] / self.visibility))

        return local, unc

    # ------------------------------------------------------------------
    # Global planning helpers (world-frame)
    # ------------------------------------------------------------------
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x / CELL_RES), int(y / CELL_RES)

    def cell_to_world(self, r: int, c: int) -> Tuple[float, float]:
        return (r + 0.5) * CELL_RES, (c + 0.5) * CELL_RES

    @property
    def dist_to_goal(self) -> float:
        return math.hypot(self.robot.x - self.goal[0],
                          self.robot.y - self.goal[1])

    @property
    def is_done(self) -> bool:
        return (self.dist_to_goal < self.goal_radius
                or self._steps >= self.max_steps)

    @property
    def had_collision(self) -> bool:
        return self._collision


# ---------------------------------------------------------------------------
# Full run helper
# ---------------------------------------------------------------------------

def run_episode(
    seed: int = 0,
    verbose: bool = False,
) -> SimResult:
    """
    Run a complete episode: build sim → plan globally on ground-truth
    (cheating for route, but costmap is noisy) → DWA locally.

    Returns the final SimResult.
    """
    from backend.planning import (
        CostmapBuilder, astar, DWAPlanner, DWAConfig,
        _GRID_SIZE, _CELL_RES,
    )

    sim = Simulator(seed=seed)

    # DWA planner
    dwa = DWAPlanner(DWAConfig(
        max_v=1.2,
        max_omega=1.5,
        predict_time=1.2,
        emergency_cost=65.0,
        heading_weight=1.5,
        velocity_weight=2.5,
        cost_weight=1.0,
    ))

    result = SimResult()
    prev_costmap = None

    while not sim.is_done:
        if sim._steps % 100 == 0 and verbose:
            print(f"Step {sim._steps}: pos=({sim.robot.x:.1f},{sim.robot.y:.1f})")
        # ── Get simulated camera observation ──────────────────────────
        raw_cost, raw_unc = sim.get_costmap()

        # Temporal smoothing (simple EMA with previous)
        if prev_costmap is not None:
            fused = 0.7 * raw_cost + 0.3 * prev_costmap
        else:
            fused = raw_cost.copy()
        prev_costmap = fused.copy()

        # ── A* on GLOBAL costmap for route planning ───────────────────
        # Plan globally on the ground-truth world (simulate having a map)
        if not hasattr(sim, "_global_path") or sim._steps % 10 == 0:
            # Subsample world to 1m grid (60x60) for instant A*
            small_world = sim.world.reshape(60, 10, 60, 10).max(axis=(1, 3))
            
            start_r, start_c = int(sim.robot.x), int(sim.robot.y)
            goal_r, goal_c = int(sim.goal[0]), int(sim.goal[1])
            
            if not hasattr(sim, "_global_path"):
                path_cells = astar(small_world, (start_r, start_c), (goal_r, goal_c))
                sim._global_path = [(r + 0.5, c + 0.5) for r, c in path_cells]
        
        path_world = getattr(sim, "_global_path", [])
        
        # Find a lookahead point on the global path ~3-5m away
        lookahead = 4.0
        wp_world = sim.goal
        if path_world:
            for wx, wy in path_world:
                d = math.hypot(wx - sim.robot.x, wy - sim.robot.y)
                if d > lookahead:
                    wp_world = (wx, wy)
                    break

        # Convert waypoint to LOCAL grid coordinates for DWA
        dx = wp_world[0] - sim.robot.x
        dy = wp_world[1] - sim.robot.y
        cos_y = math.cos(sim.robot.yaw)
        sin_y = math.sin(sim.robot.yaw)
        fwd =  dx * cos_y + dy * sin_y
        lat = -dx * sin_y + dy * cos_y

        goal_lr = _GRID_SIZE - 1 - int(round(fwd / _CELL_RES))
        goal_lc = _GRID_SIZE // 2 + int(round(lat / _CELL_RES))

        # Clamp to local grid
        goal_lr = max(0, min(_GRID_SIZE - 1, goal_lr))
        goal_lc = max(0, min(_GRID_SIZE - 1, goal_lc))

        robot_r = _GRID_SIZE - 1
        robot_c = _GRID_SIZE // 2
        wp = (goal_lr, goal_lc)

        # ── DWA step ──────────────────────────────────────────────────
        cmd = dwa.plan(
            fused,
            robot_r, robot_c,
            0.0,     # in local frame, robot always faces "up" (row-decreasing)
            wp[0], wp[1],
            speed_cap=1.0,
        )

        if cmd.v == 0.0 and cmd.omega == 0.0:
            if verbose:
                fwd = dwa._forward_cost(fused, robot_r, robot_c, 0.0)
                print(f"Stuck at step {sim._steps}, pos=({sim.robot.x:.2f},{sim.robot.y:.2f}), fwd_cost={fwd:.2f}, wp={wp}, e_stop={cmd.emergency_stop}")
            break

        # Convert DWA (v, omega) from local frame back to world
        v = cmd.v if not cmd.emergency_stop else 0.0
        omega = cmd.omega if not cmd.emergency_stop else 0.0

        result = sim.step(v, omega)

        if result.collision:
            if verbose:
                print(f"  [seed={seed}] COLLISION at step {result.steps}")
            break

        if result.dist_to_goal < sim.goal_radius:
            result.done = True
            break

    if verbose:
        status = "SUCCESS" if result.dist_to_goal < sim.goal_radius else "FAIL"
        coll = "COLLISION" if result.collision else "clean"
        print(
            f"  seed={seed:>3}  {status:>7}  {coll:<10}  "
            f"steps={result.steps:>4}  dist={result.dist_to_goal:.2f}"
        )

    return result


def run_benchmark(n_seeds: int = 20, verbose: bool = True):
    """Run n_seeds episodes and report aggregate stats."""
    successes = 0
    collisions = 0
    total_steps = 0

    print(f"\n{'='*60}")
    print(f"  DoubtNav Sim Benchmark – {n_seeds} seeds")
    print(f"{'='*60}")

    for seed in range(n_seeds):
        res = run_episode(seed=seed, verbose=verbose)
        if res.dist_to_goal < 1.5:
            successes += 1
        if res.collision:
            collisions += 1
        total_steps += res.steps

    rate = successes / n_seeds * 100
    print(f"\n{'='*60}")
    print(f"  Success rate : {successes}/{n_seeds} = {rate:.0f}%")
    print(f"  Collisions   : {collisions}")
    print(f"  Avg steps    : {total_steps / n_seeds:.0f}")
    print(f"{'='*60}\n")

    return successes, collisions, total_steps


if __name__ == "__main__":
    run_benchmark(20, verbose=True)
