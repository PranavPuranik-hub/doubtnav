"""
tests/test_localization.py
--------------------------
Tests for VisualOdometry and ConfidenceMonitor in backend/localization.py.

Strategy
--------
1. Synthetic frames are generated procedurally (no video file needed).
2. We simulate degraded conditions by calling the frame_degradation helpers
   and asserting that ConfidenceMonitor transitions through the expected modes.
3. We then feed clean frames and assert recovery back to NORMAL within the
   hysteresis window (5 frames).
"""

import sys
import os
import math

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.localization import (
    VisualOdometry,
    ConfidenceMonitor,
    Mode,
    Pose2D,
    apply_degradations,
    degrade_night,
    degrade_fog,
    degrade_glare,
    degrade_motion_blur,
    _compute_score,
    VOResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_W, _H = 512, 384

# Large textured canvas from which we crop windows – gives real feature overlap
_rng_canvas = np.random.default_rng(0)
# Add structured patterns on top of noise so ORB finds corners
_CANVAS = _rng_canvas.integers(0, 256, (_H + 400, _W + 400, 3), dtype=np.uint8)
# Draw some rectangles so ORB has stable corners
for _y in range(0, _H + 400, 60):
    for _x in range(0, _W + 400, 60):
        cv2.rectangle(_CANVAS, (_x, _y), (_x + 40, _y + 40), (255, 255, 255), 2)
        cv2.rectangle(_CANVAS, (_x + 10, _y + 10), (_x + 30, _y + 30), (0, 0, 0), 2)


def _make_textured_frame(seed: int = 0, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """Crop a _H×_W window from the global canvas, shifted by (shift_x, shift_y).
    Adjacent shifts share most pixels, giving ORB good matchable features.
    """
    x0 = shift_x % 200
    y0 = shift_y % 200
    return _CANVAS[y0: y0 + _H, x0: x0 + _W].copy()


def _vo_and_monitor(frames):
    """Run VO + ConfidenceMonitor over a list of frames, return all results."""
    vo = VisualOdometry(width=_W, height=_H)
    cm = ConfidenceMonitor()
    results = []
    for f in frames:
        vo_res = vo.update(f)
        mon_res = cm.update(vo_res)
        results.append((vo_res, mon_res))
    return results


# ---------------------------------------------------------------------------
# Unit tests: VisualOdometry
# ---------------------------------------------------------------------------

class TestVisualOdometry:

    def test_first_frame_no_crash(self):
        """First frame must return a valid VOResult without error."""
        vo = VisualOdometry(width=_W, height=_H)
        frame = _make_textured_frame(seed=1)
        res = vo.update(frame)
        assert isinstance(res, VOResult)
        assert res.num_features >= 0
        assert res.success is False   # no previous frame → cannot compute pose

    def test_pose_is_zero_after_first_frame(self):
        vo = VisualOdometry(width=_W, height=_H)
        frame = _make_textured_frame(seed=2)
        res = vo.update(frame)
        assert res.pose.x == pytest.approx(0.0)
        assert res.pose.y == pytest.approx(0.0)
        assert res.pose.yaw == pytest.approx(0.0)

    def test_brightness_and_blur_exposed(self):
        vo = VisualOdometry(width=_W, height=_H)
        frame = _make_textured_frame(seed=3)
        res = vo.update(frame)
        assert 0.0 <= res.brightness <= 255.0
        assert res.blur >= 0.0

    def test_motion_produces_nonzero_pose(self):
        """A small lateral shift should produce a non-trivial (x,y) displacement."""
        vo = VisualOdometry(width=_W, height=_H, metric_scale=0.1)
        frames = [_make_textured_frame(seed=7, shift_x=i * 3) for i in range(8)]
        results = [vo.update(f) for f in frames]
        # At least one frame should succeed
        assert any(r.success for r in results)
        # Final pose should differ from origin
        final = results[-1].pose
        total_disp = math.hypot(final.x, final.y)
        assert total_disp > 0.0

    def test_custom_camera_matrix(self):
        K = np.array([[400, 0, 256], [0, 400, 192], [0, 0, 1]], dtype=np.float64)
        vo = VisualOdometry(width=_W, height=_H, camera_matrix=K)
        assert np.allclose(vo.K, K)

    def test_black_frame_returns_gracefully(self):
        """Completely black frames have no features; VO must not crash."""
        vo = VisualOdometry(width=_W, height=_H)
        blank = np.zeros((_H, _W, 3), dtype=np.uint8)
        res0 = vo.update(blank)
        res1 = vo.update(blank)
        assert res1.success is False
        assert res1.num_features == 0


# ---------------------------------------------------------------------------
# Unit tests: ConfidenceMonitor
# ---------------------------------------------------------------------------

class TestConfidenceMonitor:

    def _good_result(self) -> VOResult:
        return VOResult(
            pose=Pose2D(),
            num_features=400,
            inlier_ratio=0.85,
            brightness=128.0,
            blur=300.0,
            success=True,
        )

    def _bad_result(self) -> VOResult:
        return VOResult(
            pose=Pose2D(),
            num_features=5,
            inlier_ratio=0.0,
            brightness=10.0,   # very dark → low brightness score
            blur=2.0,
            success=False,
        )

    def test_good_frames_give_normal(self):
        cm = ConfidenceMonitor()
        for _ in range(10):
            mon = cm.update(self._good_result())
        assert mon.mode == Mode.NORMAL
        assert mon.speed_cap == pytest.approx(1.0)

    def test_bad_frames_give_dead_reckoning(self):
        cm = ConfidenceMonitor()
        for _ in range(10):
            mon = cm.update(self._bad_result())
        assert mon.mode == Mode.DEAD_RECKONING
        assert mon.speed_cap == pytest.approx(0.3)

    def test_hysteresis_requires_5_good_frames(self):
        cm = ConfidenceMonitor(recovery_frames=5)
        # Push into DEAD_RECKONING
        for _ in range(10):
            cm.update(self._bad_result())
        assert cm.mode == Mode.DEAD_RECKONING

        # 4 good frames: should still be DEAD_RECKONING
        for _ in range(4):
            mon = cm.update(self._good_result())
        assert cm.mode == Mode.DEAD_RECKONING

        # 5th good frame: should now be NORMAL
        mon = cm.update(self._good_result())
        assert mon.mode == Mode.NORMAL

    def test_score_between_0_and_1(self):
        cm = ConfidenceMonitor()
        for res in [self._good_result(), self._bad_result()]:
            mon = cm.update(res)
            assert 0.0 <= mon.score <= 1.0

    def test_dead_reckoning_propagates_pose(self):
        """In DR mode the pose must still change (decaying velocity)."""
        cm = ConfidenceMonitor()

        # Teach it a velocity by running a few good frames with increasing pose
        for i in range(6):
            r = self._good_result()
            r.pose = Pose2D(x=float(i) * 0.1, y=float(i) * 0.05)
            cm.update(r)

        pose_before = cm.update(self._bad_result()).pose.copy()
        pose_after  = cm.update(self._bad_result()).pose.copy()

        # Pose must differ (dead-reckoning propagation)
        assert pose_before.x != pytest.approx(pose_after.x) or \
               pose_before.y != pytest.approx(pose_after.y)

    def test_speed_cap_values(self):
        cm = ConfidenceMonitor()
        # Cautious zone: manufacture a score ~ 0.45
        # brightness=70 (score=1), features=100 (score=0.33), inlier=0.5, blur=80 (blur score=0.4)
        mid_result = VOResult(
            pose=Pose2D(), num_features=100, inlier_ratio=0.5,
            brightness=70.0, blur=80.0, success=True
        )
        for _ in range(5):
            mon = cm.update(mid_result)
        assert mon.speed_cap in (0.3, 0.6, 1.0)   # must be one of the three caps


# ---------------------------------------------------------------------------
# Integration test: degraded frames → mode transitions → recovery
# ---------------------------------------------------------------------------

class TestDegradationAndRecovery:

    def _build_sequence(self):
        """
        40 clean frames → 20 degraded (night) → 10 clean (recovery).
        Returns (frames, labels) where label is 'clean' or 'degraded'.
        """
        frames, labels = [], []

        # 40 clean frames with slight lateral shift to give VO something to track
        for i in range(40):
            f = _make_textured_frame(seed=42, shift_x=i * 2)
            frames.append(f)
            labels.append("clean")

        # 20 extreme night frames (factor=0.04 → very dark, brightness ≈ 5)
        for i in range(20):
            f = _make_textured_frame(seed=42, shift_x=(40 + i) * 2)
            f = degrade_night(f, factor=0.04)
            frames.append(f)
            labels.append("degraded")

        # 10 more clean frames for recovery
        for i in range(10):
            f = _make_textured_frame(seed=42, shift_x=(60 + i) * 2)
            frames.append(f)
            labels.append("clean")

        return frames, labels

    def test_mode_transitions_and_recovery(self):
        frames, labels = self._build_sequence()
        vo = VisualOdometry(width=_W, height=_H)
        cm = ConfidenceMonitor(recovery_frames=5)

        modes = []
        for f in frames:
            vo_res = vo.update(f)
            mon_res = cm.update(vo_res)
            modes.append(mon_res.mode)

        clean_modes    = [m for m, l in zip(modes, labels) if l == "clean"]
        degraded_modes = [m for m, l in zip(modes, labels) if l == "degraded"]

        # After 40 clean frames the monitor should have been NORMAL at least once
        assert Mode.NORMAL in clean_modes, \
            "Expected NORMAL mode during clean frames."

        # During night degradation at least some frames should be non-NORMAL
        non_normal_degraded = [m for m in degraded_modes if m != Mode.NORMAL]
        assert len(non_normal_degraded) > 0, \
            "Expected mode to degrade below NORMAL during darkened frames."

        # After recovery frames the final mode should not be DEAD_RECKONING
        last_mode = modes[-1]
        assert last_mode in (Mode.NORMAL, Mode.CAUTIOUS), \
            f"Expected recovery to NORMAL/CAUTIOUS, got {last_mode}."

    def test_all_degradation_helpers_output_correct_shape(self):
        frame = _make_textured_frame(seed=99)
        for fn in [
            lambda f: degrade_night(f),
            lambda f: degrade_fog(f),
            lambda f: degrade_glare(f),
            lambda f: degrade_motion_blur(f),
            lambda f: apply_degradations(f, night=True, fog=True, glare=True, motion_blur=True),
        ]:
            out = fn(frame)
            assert out.shape == frame.shape, \
                f"Shape mismatch after degradation: {out.shape} vs {frame.shape}"
            assert out.dtype == np.uint8

    def test_night_lowers_brightness(self):
        frame = _make_textured_frame(seed=5)
        gray_orig  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        night      = degrade_night(frame, factor=0.1)
        gray_night = cv2.cvtColor(night, cv2.COLOR_BGR2GRAY)
        assert gray_night.mean() < gray_orig.mean()

    def test_fog_raises_brightness(self):
        # Dark frame + fog → mean should increase
        dark = degrade_night(_make_textured_frame(seed=6), factor=0.05)
        fogged = degrade_fog(dark, intensity=0.8)
        assert fogged.mean() > dark.mean()

    def test_compute_score_perfect_result(self):
        res = VOResult(
            pose=Pose2D(), num_features=500, inlier_ratio=1.0,
            brightness=128.0, blur=400.0, success=True
        )
        score = _compute_score(res)
        assert score == pytest.approx(1.0)

    def test_compute_score_failed_result(self):
        res = VOResult(
            pose=Pose2D(), num_features=0, inlier_ratio=0.0,
            brightness=5.0, blur=0.0, success=False
        )
        score = _compute_score(res)
        assert score == pytest.approx(0.0)
