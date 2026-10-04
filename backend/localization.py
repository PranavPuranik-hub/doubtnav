"""
localization.py – Visual Odometry, Confidence Monitor, and Frame-Degradation helpers.

VisualOdometry
--------------
  ORB(2000) + BFMatcher ratio-test → findEssentialMat(RANSAC) → recoverPose
  Integrates a 2-D pose (x, y, yaw) with a configurable nominal metric scale.

ConfidenceMonitor
-----------------
  Aggregates VO quality signals into a 0..1 score and three modes:
    NORMAL        score >= 0.6   speed_cap = 1.0
    CAUTIOUS      0.35 <= score < 0.6   speed_cap = 0.6
    DEAD_RECKONING score < 0.35  speed_cap = 0.3   (pose propagated with decaying last velocity)
  Hysteresis: 5 consecutive "good" frames required to recover from DEAD_RECKONING.

frame_degradation
-----------------
  Helpers that simulate night / fog / glare / motion-blur; used by dashboard toggles
  and tests.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public constants / enums
# ---------------------------------------------------------------------------

class Mode(enum.Enum):
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    DEAD_RECKONING = "DEAD_RECKONING"


SPEED_CAPS: dict[Mode, float] = {
    Mode.NORMAL: 1.0,
    Mode.CAUTIOUS: 0.6,
    Mode.DEAD_RECKONING: 0.3,
}

_RECOVERY_FRAMES_NEEDED = 5   # hysteresis: consecutive good frames to leave DEAD_RECKONING


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class Pose2D:
    x: float = 0.0      # metres (right-positive)
    y: float = 0.0      # metres (forward-positive)
    yaw: float = 0.0    # radians (CCW-positive)

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.yaw)


@dataclass
class VOResult:
    """All diagnostics produced by VisualOdometry.update()."""
    pose: Pose2D
    num_features: int           # ORB keypoints in current frame
    inlier_ratio: float         # RANSAC inliers / matched pairs  (0..1)
    brightness: float           # mean gray value  (0..255)
    blur: float                 # variance of Laplacian  (higher = sharper)
    success: bool               # True if Essential-mat was solvable


@dataclass
class MonitorResult:
    score: float                # aggregate confidence  (0..1)
    mode: Mode
    speed_cap: float            # 1.0 / 0.6 / 0.3
    pose: Pose2D
    consecutive_good: int       # frames since last bad observation (useful for UI)


# ---------------------------------------------------------------------------
# VisualOdometry
# ---------------------------------------------------------------------------

class VisualOdometry:
    """
    Monocular Visual Odometry using ORB features.

    Parameters
    ----------
    width, height : frame resolution (used to build the default camera matrix)
    fx            : focal length in pixels; defaults to 0.9 * width
    camera_matrix : supply a custom 3x3 K matrix to override the default
    metric_scale  : nominal metres per VO unit (one "frame displacement")
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 384,
        fx: Optional[float] = None,
        camera_matrix: Optional[np.ndarray] = None,
        metric_scale: float = 0.05,
    ):
        self.metric_scale = metric_scale

        # ---- camera intrinsics ------------------------------------------------
        if camera_matrix is not None:
            self.K = camera_matrix.astype(np.float64)
        else:
            _fx = fx if fx is not None else 0.9 * width
            _fy = _fx
            _cx = width / 2.0
            _cy = height / 2.0
            self.K = np.array(
                [[_fx, 0.0, _cx],
                 [0.0, _fy, _cy],
                 [0.0,  0.0, 1.0]],
                dtype=np.float64,
            )

        # ---- ORB detector + BFMatcher ----------------------------------------
        self._orb = cv2.ORB_create(nfeatures=2000)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # ---- state -----------------------------------------------------------
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp: Optional[list] = None
        self._prev_des: Optional[np.ndarray] = None

        self.pose = Pose2D()

    # ------------------------------------------------------------------
    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    def _image_metrics(self, gray: np.ndarray) -> Tuple[float, float]:
        brightness = float(np.mean(gray))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur = float(lap.var())
        return brightness, blur

    # ------------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray) -> VOResult:
        gray = self._to_gray(frame_bgr)
        brightness, blur = self._image_metrics(gray)

        kp, des = self._orb.detectAndCompute(gray, None)
        num_features = len(kp)

        # ---- first frame: nothing to match against ----------------------------
        if self._prev_gray is None or des is None or self._prev_des is None or num_features < 8:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            return VOResult(
                pose=self.pose.copy(),
                num_features=num_features,
                inlier_ratio=0.0,
                brightness=brightness,
                blur=blur,
                success=False,
            )

        # ---- ratio-test matching ---------------------------------------------
        raw_matches = self._matcher.knnMatch(self._prev_des, des, k=2)
        good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

        if len(good) < 8:
            self._prev_gray, self._prev_kp, self._prev_des = gray, kp, des
            return VOResult(
                pose=self.pose.copy(),
                num_features=num_features,
                inlier_ratio=0.0,
                brightness=brightness,
                blur=blur,
                success=False,
            )

        # ---- extract point arrays --------------------------------------------
        pts_prev = np.float32(
            [self._prev_kp[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        pts_curr = np.float32(
            [kp[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        # ---- Essential matrix + RANSAC ---------------------------------------
        E, mask_e = cv2.findEssentialMat(
            pts_curr, pts_prev, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )

        if E is None or mask_e is None:
            self._prev_gray, self._prev_kp, self._prev_des = gray, kp, des
            return VOResult(
                pose=self.pose.copy(),
                num_features=num_features,
                inlier_ratio=0.0,
                brightness=brightness,
                blur=blur,
                success=False,
            )

        inlier_ratio = float(mask_e.sum()) / len(good)

        # ---- recover rotation + (unit) translation ---------------------------
        n_inliers, R, t, mask_r = cv2.recoverPose(
            E, pts_curr, pts_prev, self.K, mask=mask_e.copy()
        )

        if n_inliers < 6:
            self._prev_gray, self._prev_kp, self._prev_des = gray, kp, des
            return VOResult(
                pose=self.pose.copy(),
                num_features=num_features,
                inlier_ratio=inlier_ratio,
                brightness=brightness,
                blur=blur,
                success=False,
            )

        # ---- integrate 2-D pose (x-right, z-forward in camera frame) --------
        # t from recoverPose has shape (3, 1); flatten first
        t_flat = t.flatten()
        dx_cam = float(t_flat[0]) * self.metric_scale
        dz_cam = float(t_flat[2]) * self.metric_scale   # forward in camera coords

        # Extract yaw from rotation matrix (rotation about Y axis)
        dyaw = math.atan2(R[0, 2], R[0, 0])

        # Rotate displacement into world frame
        cos_y = math.cos(self.pose.yaw)
        sin_y = math.sin(self.pose.yaw)
        self.pose.x   += cos_y * dz_cam - sin_y * dx_cam
        self.pose.y   += sin_y * dz_cam + cos_y * dx_cam
        self.pose.yaw += dyaw

        # ---- advance state ---------------------------------------------------
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_des = des

        return VOResult(
            pose=self.pose.copy(),
            num_features=num_features,
            inlier_ratio=inlier_ratio,
            brightness=brightness,
            blur=blur,
            success=True,
        )


# ---------------------------------------------------------------------------
# Confidence score helper
# ---------------------------------------------------------------------------

def _compute_score(result: VOResult) -> float:
    """
    Aggregate VO quality signals into a single 0..1 confidence score.

    Sub-scores (each 0..1, equally weighted):
      - feature_score  : saturates at 300 features
      - inlier_score   : direct inlier ratio
      - brightness_score: penalise very dark (<30) or very bright (>220) frames
      - blur_score     : saturates at 200 variance-of-Laplacian units
    """
    if not result.success:
        return 0.0

    feature_score = min(result.num_features / 300.0, 1.0)
    inlier_score  = result.inlier_ratio

    # Brightness: ideal range 50-200
    br = result.brightness
    if br < 30 or br > 230:
        brightness_score = 0.0
    elif br < 50:
        brightness_score = (br - 30) / 20.0
    elif br > 210:
        brightness_score = (230 - br) / 20.0
    else:
        brightness_score = 1.0

    blur_score = min(result.blur / 200.0, 1.0)

    return (feature_score + inlier_score + brightness_score + blur_score) / 4.0


# ---------------------------------------------------------------------------
# ConfidenceMonitor
# ---------------------------------------------------------------------------

class ConfidenceMonitor:
    """
    Combines raw VO results into a navigation mode with hysteresis and
    dead-reckoning pose propagation.

    Parameters
    ----------
    recovery_frames : consecutive good frames needed to leave DEAD_RECKONING
    velocity_decay  : multiplier applied to the stored velocity each frame
                      while in DEAD_RECKONING (default 0.95)
    """

    _SCORE_NORMAL    = 0.85
    _SCORE_CAUTIOUS  = 0.60
    _HYSTERESIS      = 0.05
    _CALIBRATION_FRAMES = 20

    def __init__(
        self,
        recovery_frames: int = _RECOVERY_FRAMES_NEEDED,
        velocity_decay: float = 0.95,
    ):
        self._recovery_frames = recovery_frames
        self._velocity_decay  = velocity_decay

        self._mode             = Mode.NORMAL
        self._consecutive_good = 0

        # Last stable velocity (world-frame, metres per frame)
        self._vel_x:   float = 0.0
        self._vel_y:   float = 0.0
        self._vel_yaw: float = 0.0

        # Last known good pose (used as base for dead-reckoning)
        self._dr_pose = Pose2D()

        # Score history (for smoothing, optional)
        self._score_history: deque[float] = deque(maxlen=5)

        # Calibration
        self._calibration_scores: List[float] = []
        self._baseline_score: Optional[float] = None

    # ------------------------------------------------------------------
    def update(self, result: VOResult) -> MonitorResult:
        raw_score = _compute_score(result)

        # 1. Calibration phase
        if self._baseline_score is None:
            if result.success:
                self._calibration_scores.append(raw_score)
            if len(self._calibration_scores) >= self._CALIBRATION_FRAMES:
                self._baseline_score = float(np.mean(self._calibration_scores))
                if self._baseline_score < 0.01:
                    self._baseline_score = 1.0  # prevent div by zero
            # Return NORMAL during calibration
            self._dr_pose = result.pose.copy()
            return MonitorResult(
                score=1.0,
                mode=Mode.NORMAL,
                speed_cap=1.0,
                pose=result.pose.copy(),
                consecutive_good=self._consecutive_good
            )

        # 2. Normalization
        norm_score = raw_score / self._baseline_score
        
        # Light temporal smoothing
        self._score_history.append(norm_score)
        score = float(np.mean(self._score_history))

        # ---- mode transitions with hysteresis --------------------------------
        if self._mode == Mode.DEAD_RECKONING:
            if score >= self._SCORE_NORMAL + self._HYSTERESIS:
                self._consecutive_good += 1
            else:
                self._consecutive_good = 0

            if self._consecutive_good >= self._recovery_frames:
                self._mode = Mode.NORMAL
                self._consecutive_good = 0
        elif self._mode == Mode.CAUTIOUS:
            if score >= self._SCORE_NORMAL + self._HYSTERESIS:
                self._mode = Mode.NORMAL
                self._consecutive_good = 0
            elif score < self._SCORE_CAUTIOUS - self._HYSTERESIS:
                self._mode = Mode.DEAD_RECKONING
                self._consecutive_good = 0
        else:
            # NORMAL
            if score < self._SCORE_CAUTIOUS - self._HYSTERESIS:
                self._mode = Mode.DEAD_RECKONING
                self._consecutive_good = 0
            elif score < self._SCORE_NORMAL - self._HYSTERESIS:
                self._mode = Mode.CAUTIOUS
                self._consecutive_good = 0

        # ---- update velocity estimate / pose ---------------------------------
        pose: Pose2D

        if self._mode in (Mode.NORMAL, Mode.CAUTIOUS) and result.success:
            # Learn the stable velocity from the VO result
            prev_pose = self._dr_pose
            self._vel_x   = result.pose.x   - prev_pose.x
            self._vel_y   = result.pose.y   - prev_pose.y
            self._vel_yaw = result.pose.yaw - prev_pose.yaw
            self._dr_pose = result.pose.copy()
            pose = result.pose.copy()

        else:
            # DEAD_RECKONING: propagate with decaying last velocity
            self._vel_x   *= self._velocity_decay
            self._vel_y   *= self._velocity_decay
            self._vel_yaw *= self._velocity_decay

            self._dr_pose.x   += self._vel_x
            self._dr_pose.y   += self._vel_y
            self._dr_pose.yaw += self._vel_yaw
            pose = self._dr_pose.copy()

        return MonitorResult(
            score=score,
            mode=self._mode,
            speed_cap=SPEED_CAPS[self._mode],
            pose=pose,
            consecutive_good=self._consecutive_good,
        )

    # ------------------------------------------------------------------
    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def consecutive_good(self) -> int:
        return self._consecutive_good


# ---------------------------------------------------------------------------
# Frame-degradation helpers  (dashboard toggles / test fixtures)
# ---------------------------------------------------------------------------

def degrade_night(frame_bgr: np.ndarray, factor: float = 0.2) -> np.ndarray:
    """Simulate low-light by scaling pixel values down.
    factor=0 → black, factor=1 → original.
    """
    return np.clip(frame_bgr * factor, 0, 255).astype(np.uint8)


def degrade_fog(frame_bgr: np.ndarray, intensity: float = 0.6) -> np.ndarray:
    """Blend the frame with a white veil to simulate fog.
    intensity=0 → unchanged, intensity=1 → pure white.
    """
    fog_layer = np.full_like(frame_bgr, 220, dtype=np.uint8)
    return cv2.addWeighted(frame_bgr, 1.0 - intensity, fog_layer, intensity, 0)


def degrade_glare(
    frame_bgr: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: int = 80,
    strength: float = 180.0,
) -> np.ndarray:
    """Add a circular bright glare spot (sun / headlights)."""
    h, w = frame_bgr.shape[:2]
    if center is None:
        center = (w // 2, h // 3)

    glare = np.zeros((h, w), dtype=np.float32)
    cv2.circle(glare, center, radius, strength, -1)
    # Gaussian falloff
    glare = cv2.GaussianBlur(glare, (0, 0), radius // 2)

    out = frame_bgr.astype(np.float32)
    out[:, :, 0] += glare
    out[:, :, 1] += glare
    out[:, :, 2] += glare
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_motion_blur(frame_bgr: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """Apply horizontal motion blur."""
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(frame_bgr, -1, k)


def apply_degradations(
    frame_bgr: np.ndarray,
    night: bool = False,
    fog: bool = False,
    glare: bool = False,
    motion_blur: bool = False,
    night_factor: float = 0.15,
    fog_intensity: float = 0.55,
    blur_kernel: int = 25,
) -> np.ndarray:
    """
    Convenience wrapper used by dashboard toggles.
    Applies any combination of degradations in a stable order.
    """
    frame = frame_bgr.copy()
    if night:
        frame = degrade_night(frame, factor=night_factor)
    if fog:
        frame = degrade_fog(frame, intensity=fog_intensity)
    if glare:
        frame = degrade_glare(frame)
    if motion_blur:
        frame = degrade_motion_blur(frame, kernel_size=blur_kernel)
    return frame
