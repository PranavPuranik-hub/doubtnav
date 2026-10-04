"""
backend/streamer.py
-------------------
Streaming engine for DoubtNav.
Generates WebSocket frame payloads at ~10 FPS:
  - Base64 JPEG overlay frame
  - Base64 JPEG uncertainty heatmap (soft teal-to-amber colormap)
  - Base64 JPEG costmap with planned path
  - Trajectory points
  - Confidence score, mode, speed cap
  - FPS and event log entries
Handles both 'video' and 'sim' session modes, frame degradations, and simulator controls.
"""

from __future__ import annotations

import base64
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.localization import (
    ConfidenceMonitor,
    Mode,
    VisualOdometry,
    apply_degradations,
)
from backend.planning import (
    CameraConfig,
    CostmapBuilder,
    DWAPlanner,
    DWAConfig,
    _GRID_SIZE,
    astar,
)
from backend.sim import Simulator


# ---------------------------------------------------------------------------
# Colormap & Rendering Helpers
# ---------------------------------------------------------------------------

def _build_teal_to_amber_palette() -> np.ndarray:
    """
    Build a 256x3 BGR palette for soft teal-to-amber transition.
      Uncertainty 0.0 -> Soft Teal  RGB(32, 178, 170) -> BGR(170, 178, 32)
      Uncertainty 0.5 -> Soft Cream RGB(230, 215, 140) -> BGR(140, 215, 230)
      Uncertainty 1.0 -> Soft Amber RGB(245, 160, 32)  -> BGR(32, 160, 245)
    """
    palette = np.zeros((256, 3), dtype=np.uint8)
    c_teal = np.array([170.0, 178.0, 32.0])   # BGR
    c_mid  = np.array([140.0, 215.0, 230.0])  # BGR
    c_amber= np.array([32.0, 160.0, 245.0])   # BGR

    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            alpha = t / 0.5
            color = (1.0 - alpha) * c_teal + alpha * c_mid
        else:
            alpha = (t - 0.5) / 0.5
            color = (1.0 - alpha) * c_mid + alpha * c_amber
        palette[i] = np.clip(color, 0, 255).astype(np.uint8)
    return palette

_TEAL_AMBER_PALETTE = _build_teal_to_amber_palette()


def colorize_uncertainty(uncertainty: np.ndarray) -> np.ndarray:
    """
    Apply soft teal-to-amber colormap to uncertainty map (HxW in [0..1]).
    Returns HxWx3 BGR image.
    """
    u_norm = np.clip(uncertainty, 0.0, 1.0)
    u_uint8 = (u_norm * 255.0).astype(np.uint8)
    return _TEAL_AMBER_PALETTE[u_uint8]


def render_costmap_with_path(
    costmap: np.ndarray,
    planned_path: Optional[List[Tuple[int, int]]] = None,
    target_size: int = 300,
) -> np.ndarray:
    """
    Render 100x100 costmap (0..100) into a 300x300 BGR image with:
      - Colorized terrain (dark slate traversable, amber risky, coral obstacle)
      - Planned A* path drawn in glowing cyan
      - Robot marker (bottom center) and lookahead target
    """
    # Create base BGR image from cost values
    # Cost: 1 = traversable, 8 = risky, 50 = unknown, 90/100 = obstacle
    h, w = costmap.shape
    base_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # Palette (BGR)
    color_traversable = np.array([90, 80, 40], dtype=np.uint8)    # dark slate-teal
    color_risky       = np.array([30, 150, 220], dtype=np.uint8)   # warm amber
    color_unknown     = np.array([80, 80, 80], dtype=np.uint8)     # neutral gray
    color_obstacle    = np.array([70, 70, 220], dtype=np.uint8)    # coral red

    c = np.clip(costmap, 0.0, 100.0)

    # Interpolate colors based on ranges
    mask_trav = c <= 5.0
    mask_risk = (c > 5.0) & (c <= 30.0)
    mask_unk  = (c > 30.0) & (c < 80.0)
    mask_obs  = c >= 80.0

    base_bgr[mask_trav] = color_traversable
    base_bgr[mask_risk] = color_risky
    base_bgr[mask_unk]  = color_unknown
    base_bgr[mask_obs]  = color_obstacle

    # Upscale to target size for crisp display
    img = cv2.resize(base_bgr, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    # Subtle 1m grid lines (every 10 cells -> target_size // 10 pixels)
    step_px = target_size // 10
    for g in range(1, 10):
        pos = g * step_px
        cv2.line(img, (pos, 0), (pos, target_size), (45, 45, 45), 1)
        cv2.line(img, (0, pos), (target_size, pos), (45, 45, 45), 1)

    scale = target_size / float(_GRID_SIZE)

    # Draw planned path
    if planned_path and len(planned_path) > 1:
        pts = []
        for r, col in planned_path:
            px = int((col + 0.5) * scale)
            py = int((r + 0.5) * scale)
            pts.append([px, py])
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_arr], isClosed=False, color=(255, 240, 0), thickness=2)

        # Highlight lookahead point
        if len(pts) > 5:
            lp = pts[min(8, len(pts) - 1)]
            cv2.circle(img, (lp[0], lp[1]), 5, (0, 255, 255), -1)
            cv2.circle(img, (lp[0], lp[1]), 8, (255, 255, 255), 1)

    # Draw Robot at cell (99, 50)
    rx = int(50.5 * scale)
    ry = int(99.0 * scale) - 4
    # Triangle pointing up
    tri_pts = np.array([
        [rx, ry - 8],
        [rx - 6, ry + 6],
        [rx + 6, ry + 6]
    ], dtype=np.int32)
    cv2.fillPoly(img, [tri_pts], (0, 255, 0))
    cv2.polylines(img, [tri_pts], isClosed=True, color=(255, 255, 255), thickness=1)

    return img


def to_base64_jpeg(img_bgr: np.ndarray, quality: int = 80) -> str:
    """Encode BGR image to base64 JPEG string."""
    ret, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return ""
    return base64.b64encode(buf).decode("ascii")


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Coordinates perception, localization, planning, and simulation
    to produce ~10 FPS streaming updates for the frontend.
    """

    def __init__(self, clips_dir: str = "data/clips"):
        self.clips_dir = os.path.abspath(clips_dir)
        self.mode: str = "video"       # 'video' or 'sim'
        self.clip_name: str = "driving_sample.mp4"
        self.degradation: str = "none" # 'none', 'night', 'fog', 'glare', 'motion_blur'

        # Pipeline components
        self._perceiver = None         # Lazy loaded
        self._vo = VisualOdometry(width=512, height=384)
        self._monitor = ConfidenceMonitor()
        self._costmap_builder = CostmapBuilder(CameraConfig(img_w=512, img_h=384), vehicle_radius=0.3)
        self._dwa = DWAPlanner(DWAConfig(max_v=1.2, max_omega=1.5, emergency_cost=65.0))
        self._sim = Simulator(seed=42)

        # Video capture
        self._cap: Optional[cv2.VideoCapture] = None
        self._init_video_cap()

        # State tracking
        self.last_mode: Mode = Mode.NORMAL
        self.event_log: deque = deque(maxlen=60)
        self.pose_history: deque = deque(maxlen=300)
        self.frame_count: int = 0
        self.last_time: float = time.time()
        self.measured_fps: float = 10.0

        # Cached outputs for fallback / rate matching
        self._last_uncertainty: np.ndarray = np.zeros((384, 512), dtype=np.float32)
        self._last_group_mask: np.ndarray = np.zeros((384, 512), dtype=np.uint8)
        self._last_costmap: np.ndarray = np.full((_GRID_SIZE, _GRID_SIZE), 50.0, dtype=np.float64)

        self.add_event("Session initialized: mode='video', degradation='none'", level="info")

    # ------------------------------------------------------------------
    # Event Log
    # ------------------------------------------------------------------
    def add_event(self, message: str, level: str = "info"):
        ts_str = datetime.now().strftime("%H:%M:%S")
        self.event_log.append({
            "timestamp": time.time(),
            "time_str": ts_str,
            "message": message,
            "level": level,
        })

    # ------------------------------------------------------------------
    # Clips & Session Configuration
    # ------------------------------------------------------------------
    def get_available_clips(self) -> List[Dict[str, Any]]:
        clips = []
        if os.path.isdir(self.clips_dir):
            for fname in sorted(os.listdir(self.clips_dir)):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    fpath = os.path.join(self.clips_dir, fname)
                    clips.append({
                        "name": fname,
                        "size_bytes": os.path.getsize(fpath),
                    })
        return clips

    def _init_video_cap(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        target_clip = os.path.join(self.clips_dir, self.clip_name)
        if not os.path.isfile(target_clip):
            clips = self.get_available_clips()
            if clips:
                self.clip_name = clips[0]["name"]
                target_clip = os.path.join(self.clips_dir, self.clip_name)

        if os.path.isfile(target_clip):
            self._cap = cv2.VideoCapture(target_clip)
        else:
            self._cap = None

    def set_session(
        self,
        mode: Optional[str] = None,
        clip: Optional[str] = None,
        degradation: Optional[str] = None,
    ) -> Dict[str, Any]:
        changed = False

        if mode and mode in ("video", "sim") and mode != self.mode:
            self.mode = mode
            self.pose_history.clear()
            self.add_event(f"Switched session mode to '{self.mode}'", level="info")
            changed = True

        if clip and clip != self.clip_name:
            self.clip_name = clip
            self.pose_history.clear()
            self._init_video_cap()
            self.add_event(f"Selected video clip '{self.clip_name}'", level="info")
            changed = True

        if degradation and degradation in ("none", "night", "fog", "glare", "motion_blur"):
            if degradation != self.degradation:
                self.degradation = degradation
                self.add_event(f"Applied degradation toggle: '{self.degradation}'", level="warning" if degradation != "none" else "info")
                changed = True

        return {
            "status": "ok",
            "mode": self.mode,
            "clip": self.clip_name,
            "degradation": self.degradation,
        }

    # ------------------------------------------------------------------
    # Simulator Controls
    # ------------------------------------------------------------------
    def sim_spawn_obstacle(self, x: Optional[float] = None, y: Optional[float] = None, radius: float = 1.0) -> Dict[str, Any]:
        if x is None or y is None:
            # Spawn dynamically 8 meters in front of the robot
            dist = 8.0
            rx = self._sim.robot.x
            ry = self._sim.robot.y
            ryaw = self._sim.robot.yaw
            x = rx + dist * np.cos(ryaw)
            y = ry + dist * np.sin(ryaw)
            
        self._sim.spawn_obstacle(x, y, radius)
        self.add_event(f"Spawned obstacle at x={x:.1f}m, y={y:.1f}m (r={radius:.1f}m)", level="warning")
        return {"status": "ok", "x": x, "y": y, "radius": radius}

    def sim_set_goal(self, x: float, y: float) -> Dict[str, Any]:
        self._sim.goal = (float(x), float(y))
        self.add_event(f"Updated navigation goal to ({x:.1f}, {y:.1f})", level="info")
        return {"status": "ok", "goal": [x, y]}

    def sim_reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        s = seed if seed is not None else 42
        self._sim = Simulator(seed=s)
        self._costmap_builder.reset()
        self.pose_history.clear()
        self.add_event(f"Simulator reset with seed {s}", level="info")
        return {"status": "ok", "seed": s}

    # ------------------------------------------------------------------
    # Lazy Perceiver Loading
    # ------------------------------------------------------------------
    @property
    def perceiver(self):
        if self._perceiver is None:
            self.add_event("Loading SegFormer segmentation model (nvidia/segformer-b0-finetuned-ade-512-512)...", level="info")
            from backend.perception import Perceiver
            self._perceiver = Perceiver()
            self.add_event("Perceiver model ready.", level="info")
        return self._perceiver

    # ------------------------------------------------------------------
    # Frame Pipeline
    # ------------------------------------------------------------------
    def get_next_frame(self) -> Dict[str, Any]:
        """Produce a complete frame payload at ~10 FPS."""
        t_now = time.time()
        dt = max(0.001, t_now - self.last_time)
        self.last_time = t_now
        self.frame_count += 1
        inst_fps = 1.0 / dt
        self.measured_fps = 0.85 * self.measured_fps + 0.15 * inst_fps

        if self.mode == "sim":
            return self._step_simulation()
        else:
            return self._step_video()

    def _apply_current_degradation(self, frame_bgr: np.ndarray) -> np.ndarray:
        night = (self.degradation == "night")
        fog = (self.degradation == "fog")
        glare = (self.degradation == "glare")
        blur = (self.degradation == "motion_blur")
        return apply_degradations(
            frame_bgr,
            night=night,
            fog=fog,
            glare=glare,
            motion_blur=blur,
            night_factor=0.08,
            fog_intensity=0.65,
            blur_kernel=27,
        )

    # ------------------------------------------------------------------
    # Video Mode Processing
    # ------------------------------------------------------------------
    def _step_video(self) -> Dict[str, Any]:
        # 1. Read Frame
        frame = None
        if self._cap is not None and self._cap.isOpened():
            ret, f = self._cap.read()
            if not ret or f is None:
                # Loop back
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f = self._cap.read()
            if ret and f is not None:
                frame = cv2.resize(f, (512, 384))

        if frame is None:
            # Synthetic canvas fallback
            frame = np.full((384, 512, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (100, 200), (412, 384), (90, 80, 50), -1)
            cv2.putText(frame, "No Video Clip Available", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # 2. Apply Degradation
        degraded = self._apply_current_degradation(frame)

        # 3. Visual Odometry & Confidence Monitor
        vo_res = self._vo.update(degraded)
        mon_res = self._monitor.update(vo_res)

        # Check for mode transition events
        if mon_res.mode != self.last_mode:
            if mon_res.mode == Mode.DEAD_RECKONING:
                reason = "low light" if self.degradation == "night" else ("low features/blur" if vo_res.num_features < 50 else "loss of visual tracking")
                self.add_event(f"Switched to dead-reckoning: {reason} (score={mon_res.score:.2f})", level="error")
            elif mon_res.mode == Mode.CAUTIOUS:
                self.add_event(f"Switched to CAUTIOUS mode (score={mon_res.score:.2f}, speed cap={mon_res.speed_cap:.1f}x)", level="warning")
            elif mon_res.mode == Mode.NORMAL:
                self.add_event(f"Recovered to NORMAL mode: tracking stable (score={mon_res.score:.2f})", level="info")
            self.last_mode = mon_res.mode

        # 4. Perception
        # Run perceiver inference (or reuse every other frame if CPU bound)
        try:
            group_mask, uncertainty, overlay = self.perceiver.predict(degraded)
            self._last_uncertainty = uncertainty
            self._last_group_mask = group_mask
        except Exception as e:
            # Fallback if torch error
            group_mask = np.ones((384, 512), dtype=np.uint8)
            uncertainty = np.full((384, 512), 0.2, dtype=np.float32)
            overlay = degraded.copy()

        # 5. Costmap & Planning
        costmap = self._costmap_builder.update(group_mask, uncertainty)
        self._last_costmap = costmap

        # Plan local path from robot (row 99, col 50) towards forward goal (row 10, col 50)
        start_cell = (99, 50)
        target_cell = (10, 50)
        path = astar(costmap, start_cell, target_cell, cost_threshold=75.0)

        # Update historical pose for trajectory map
        if self.frame_count % 3 == 0:  # Sample to save bandwidth
            self.pose_history.append({"x": round(float(mon_res.pose.x), 2), "y": round(float(mon_res.pose.y), 2)})
        trajectory = list(self.pose_history)

        # 6. Render Heatmaps & Visualizations
        unc_heatmap = colorize_uncertainty(uncertainty)
        costmap_viz = render_costmap_with_path(costmap, planned_path=path)

        # Overlay HUD annotations on overlay
        hud = overlay.copy()
        mode_color = (0, 220, 0) if mon_res.mode == Mode.NORMAL else ((0, 200, 255) if mon_res.mode == Mode.CAUTIOUS else (0, 0, 240))
        cv2.putText(hud, f"MODE: {mon_res.mode.value}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(hud, f"CONF: {mon_res.score:.2f} | CAP: {mon_res.speed_cap:.1f}x", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

        return {
            "mode": mon_res.mode.value,
            "confidence": round(float(mon_res.score), 3),
            "speed_cap": round(float(mon_res.speed_cap), 2),
            "fps": round(float(self.measured_fps), 1),
            "overlay": to_base64_jpeg(hud),
            "uncertainty_heatmap": to_base64_jpeg(unc_heatmap),
            "costmap": to_base64_jpeg(costmap_viz),
            "trajectory": trajectory,
            "telemetry": {
                "x": round(float(mon_res.pose.x), 3),
                "y": round(float(mon_res.pose.y), 3),
                "yaw_deg": round(float(np.degrees(mon_res.pose.yaw)), 1),
                "features": int(vo_res.num_features),
                "inlier_ratio": round(float(vo_res.inlier_ratio), 3),
                "brightness": round(float(vo_res.brightness), 1),
                "blur": round(float(vo_res.blur), 1),
                "step": self.frame_count,
            },
            "event_log": list(self.event_log),
        }

    # ------------------------------------------------------------------
    # Simulation Mode Processing
    # ------------------------------------------------------------------
    def _step_simulation(self) -> Dict[str, Any]:
        sim = self._sim
        raw_cost, raw_unc = sim.get_costmap()

        # Apply costmap builder fusion
        costmap = self._costmap_builder.update_from_raw(raw_cost)

        # Plan local goal
        dx = sim.goal[0] - sim.robot.x
        dy = sim.goal[1] - sim.robot.y
        cos_y = np.cos(sim.robot.yaw)
        sin_y = np.sin(sim.robot.yaw)
        fwd =  dx * cos_y + dy * sin_y
        lat = -dx * sin_y + dy * cos_y

        goal_lr = max(0, min(_GRID_SIZE - 1, _GRID_SIZE - 1 - int(round(fwd / 0.1))))
        goal_lc = max(0, min(_GRID_SIZE - 1, _GRID_SIZE // 2 + int(round(lat / 0.1))))

        robot_r = 99
        robot_c = 50
        path = astar(costmap, (robot_r, robot_c), (goal_lr, goal_lc), cost_threshold=80.0)

        wp = path[min(6, len(path) - 1)] if path else (goal_lr, goal_lc)

        # DWA Planner step
        cmd = self._dwa.plan(costmap, robot_r, robot_c, 0.0, wp[0], wp[1], speed_cap=1.0)
        sim_res = sim.step(cmd.v if not cmd.emergency_stop else 0.0, cmd.omega if not cmd.emergency_stop else 0.0)

        if cmd.emergency_stop:
            self.add_event("Emergency stop triggered: forward obstacle proximity", level="error")
        if sim_res.collision:
            self.add_event(f"COLLISION detected at ({sim.robot.x:.1f}, {sim.robot.y:.1f})", level="error")
        if sim_res.done and sim_res.dist_to_goal < sim.goal_radius:
            self.add_event(f"Goal reached! Final distance: {sim_res.dist_to_goal:.2f}m", level="info")

        # Confidence in sim is high unless near noisy/unknown boundaries
        mean_unc = float(np.mean(raw_unc[80:, 40:60]))
        conf = max(0.2, 1.0 - mean_unc)
        mode = "NORMAL" if conf >= 0.6 else ("CAUTIOUS" if conf >= 0.35 else "DEAD_RECKONING")

        # Synthetic camera view for overlay
        # Sliced representation of the local terrain
        overlay_view = np.zeros((384, 512, 3), dtype=np.uint8)
        # Sky
        overlay_view[:190, :] = [180, 140, 80]
        # Ground
        overlay_view[190:, :] = [70, 110, 80]
        # Draw horizon line
        cv2.line(overlay_view, (0, 190), (512, 190), (120, 120, 120), 2)
        # Draw forward path projection
        cv2.putText(overlay_view, "SIMULATOR SYNTHETIC FEED", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(overlay_view, f"POS: ({sim.robot.x:.1f}, {sim.robot.y:.1f}) | GOAL DIST: {sim_res.dist_to_goal:.1f}m", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

        # Render heatmaps
        unc_large = cv2.resize(raw_unc, (512, 384))
        unc_heatmap = colorize_uncertainty(unc_large)
        costmap_viz = render_costmap_with_path(costmap, planned_path=path)

        if self.frame_count % 3 == 0:
            self.pose_history.append({"x": round(sim.robot.x, 2), "y": round(sim.robot.y, 2)})
        trajectory = list(self.pose_history)

        return {
            "mode": mode,
            "confidence": round(conf, 3),
            "speed_cap": 1.0 if mode == "NORMAL" else (0.6 if mode == "CAUTIOUS" else 0.3),
            "fps": round(float(self.measured_fps), 1),
            "overlay": to_base64_jpeg(overlay_view),
            "uncertainty_heatmap": to_base64_jpeg(unc_heatmap),
            "costmap": to_base64_jpeg(costmap_viz),
            "trajectory": trajectory,
            "telemetry": {
                "x": round(sim.robot.x, 2),
                "y": round(sim.robot.y, 2),
                "yaw_deg": round(float(np.degrees(sim.robot.yaw)), 1),
                "dist_to_goal": round(sim_res.dist_to_goal, 2),
                "v": round(cmd.v, 2),
                "omega": round(cmd.omega, 2),
                "step": sim_res.steps,
            },
            "event_log": list(self.event_log),
        }
