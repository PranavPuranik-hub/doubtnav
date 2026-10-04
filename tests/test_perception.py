import numpy as np
import cv2
import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.perception import Perceiver

def test_perceiver_synthetic_image():
    # Model initialization
    perceiver = Perceiver(device="cpu")
    
    # Create a synthetic image (e.g. 1920x1080 BGR)
    # This simulates a frame from a video or a camera
    img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    
    group_mask, uncertainty, overlay = perceiver.predict(img)
    
    # Check dimensions (should be 512x384 resize, wait OpenCV resize is (width, height), so shape is (384, 512))
    assert group_mask.shape == (384, 512), f"Expected mask shape (384, 512), got {group_mask.shape}"
    assert group_mask.dtype == np.uint8, "Group mask should be uint8"
    
    # Uncertainty
    assert uncertainty.shape == (384, 512), f"Expected uncertainty shape (384, 512), got {uncertainty.shape}"
    # Uncertainty is normalized between 0 and 1
    assert np.min(uncertainty) >= -1e-5, f"Uncertainty min should be >= 0, got {np.min(uncertainty)}"
    assert np.max(uncertainty) <= 1.0 + 1e-5, f"Uncertainty max should be <= 1, got {np.max(uncertainty)}"
    
    # Overlay
    assert overlay.shape == (384, 512, 3), f"Expected overlay shape (384, 512, 3), got {overlay.shape}"
    assert overlay.dtype == np.uint8, "Overlay should be uint8"
