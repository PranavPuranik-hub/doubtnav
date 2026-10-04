"""
scripts/demo_localization.py
----------------------------
Quick sanity-check for VisualOdometry + ConfidenceMonitor.
Runs 60 synthetic frames and prints a live table, then 20 night-degraded
frames to force DEAD_RECKONING, then 10 clean frames to show recovery.
No video file needed.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from backend.localization import (
    VisualOdometry, ConfidenceMonitor, Mode,
    degrade_night, apply_degradations,
)

W, H = 512, 384

# ── Build a canvas with structure so ORB finds corners ─────────────────────
rng = np.random.default_rng(0)
CANVAS = rng.integers(0, 256, (H + 400, W + 400, 3), dtype=np.uint8)
for y in range(0, H + 400, 60):
    for x in range(0, W + 400, 60):
        cv2.rectangle(CANVAS, (x, y), (x+40, y+40), (255,255,255), 2)
        cv2.rectangle(CANVAS, (x+10, y+10), (x+30, y+30), (0,0,0), 2)

def get_frame(shift_x=0, shift_y=0):
    x0, y0 = shift_x % 200, shift_y % 200
    return CANVAS[y0:y0+H, x0:x0+W].copy()

# ── Init ───────────────────────────────────────────────────────────────────
vo = VisualOdometry(width=W, height=H)
cm = ConfidenceMonitor()

MODE_COLOR = {
    Mode.NORMAL:         "\033[92m",   # green
    Mode.CAUTIOUS:       "\033[93m",   # yellow
    Mode.DEAD_RECKONING: "\033[91m",   # red
}
RESET = "\033[0m"

HDR = (f"{'#':>4}  {'Phase':<12}  {'Mode':<16}  "
       f"{'Score':>6}  {'Cap':>4}  "
       f"{'Feats':>6}  {'Inlier':>7}  "
       f"{'Bright':>7}  {'Blur':>8}  "
       f"{'x':>7}  {'y':>7}  {'yaw_deg':>8}")
print(HDR)
print("-" * 120)

phases = (
    [(get_frame(i*2, 0),      "clean")    for i in range(60)] +
    [(degrade_night(get_frame((60+i)*2, 0), factor=0.04), "NIGHT") for i in range(20)] +
    [(get_frame((80+i)*2, 0), "recovery") for i in range(15)]
)

for idx, (frame, phase) in enumerate(phases):
    vr = vo.update(frame)
    mr = cm.update(vr)

    col = MODE_COLOR[mr.mode]
    mode_str = f"{col}{mr.mode.value:<14}{RESET}"

    print(
        f"{idx+1:>4}  {phase:<12}  {mr.mode.value:<16}  "
        f"{mr.score:>6.3f}  {mr.speed_cap:>4.1f}  "
        f"{vr.num_features:>6}  {vr.inlier_ratio:>7.3f}  "
        f"{vr.brightness:>7.1f}  {vr.blur:>8.1f}  "
        f"{mr.pose.x:>7.4f}  {mr.pose.y:>7.4f}  {mr.pose.yaw*57.3:>8.2f}"
    )

print("\nDone. Demo complete.")
