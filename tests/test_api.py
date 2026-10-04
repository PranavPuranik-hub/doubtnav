"""
tests/test_api.py
-----------------
Tests for FastAPI application, REST endpoints, and WebSocket streaming.
"""

import base64
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app, session_manager


@pytest.fixture
def client():
    # Use TestClient
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_sec" in data
    assert "mode" in data
    assert "degradation" in data


def test_clips_endpoint(client):
    response = client.get("/clips")
    assert response.status_code == 200
    data = response.json()
    assert "clips" in data
    assert isinstance(data["clips"], list)


def test_session_configuration(client):
    # Set to sim mode
    resp1 = client.post("/session", json={"mode": "sim", "degradation": "none"})
    assert resp1.status_code == 200
    d1 = resp1.json()
    assert d1["mode"] == "sim"

    # Set degradation
    resp2 = client.post("/session", json={"degradation": "night"})
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert d2["degradation"] == "night"

    # Reset back to video mode
    resp3 = client.post("/session", json={"mode": "video", "degradation": "none"})
    assert resp3.status_code == 200
    d3 = resp3.json()
    assert d3["mode"] == "video"
    assert d3["degradation"] == "none"


def test_sim_endpoints(client):
    # Spawn obstacle
    r_obs = client.post("/sim/spawn_obstacle", json={"x": 12.5, "y": 15.0, "radius": 1.5})
    assert r_obs.status_code == 200
    d_obs = r_obs.json()
    assert d_obs["status"] == "ok"
    assert d_obs["x"] == 12.5
    assert d_obs["y"] == 15.0

    # Set goal
    r_goal = client.post("/sim/goal", json={"x": 48.0, "y": 48.0})
    assert r_goal.status_code == 200
    d_goal = r_goal.json()
    assert d_goal["status"] == "ok"
    assert d_goal["goal"] == [48.0, 48.0]

    # Reset sim
    r_reset = client.post("/sim/reset", json={"seed": 101})
    assert r_reset.status_code == 200
    d_reset = r_reset.json()
    assert d_reset["status"] == "ok"
    assert d_reset["seed"] == 101


def test_benchmark_endpoint(client):
    response = client.get("/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "success_rate" in data or "success_percentage" in data
    assert data["collisions"] == 0
    assert data["status"] == "PASSED"


def test_websocket_streaming_sim(client):
    # Switch to sim mode for quick frame generation without loading heavy video models
    client.post("/session", json={"mode": "sim"})

    with client.websocket_connect("/ws") as ws:
        # Receive frame payload
        msg = ws.receive_json()

        # Validate required fields
        assert "overlay" in msg
        assert "uncertainty_heatmap" in msg
        assert "costmap" in msg
        assert "trajectory" in msg
        assert "confidence" in msg
        assert "mode" in msg
        assert "fps" in msg
        assert "event_log" in msg

        # Validate base64 images
        for key in ("overlay", "uncertainty_heatmap", "costmap"):
            b64_str = msg[key]
            assert isinstance(b64_str, str)
            assert len(b64_str) > 100
            # Test decode
            decoded = base64.b64decode(b64_str)
            assert len(decoded) > 0

        # Validate types
        assert isinstance(msg["confidence"], (int, float))
        assert 0.0 <= msg["confidence"] <= 1.0
        assert msg["mode"] in ("NORMAL", "CAUTIOUS", "DEAD_RECKONING")
        assert isinstance(msg["fps"], (int, float))
        assert isinstance(msg["trajectory"], list)
        assert isinstance(msg["event_log"], list)
        assert len(msg["event_log"]) > 0

        # Disconnect cleanly by exiting context manager
